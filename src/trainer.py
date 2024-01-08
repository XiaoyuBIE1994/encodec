#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2023 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inria.fr)
License agreement in LICENSE.txt
"""

import os
import shutil
# import wandb
import math
import importlib
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import copy
from pesq import pesq

import torch
import torchaudio
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger

# import src.losses
from src import datasets, models, losses, utils
from src.encodec import EncodecModel
from src.encodec.utils import convert_audio

logger = get_logger(__name__) # avoid douplicated print, params defined by Hydra
accelerator = Accelerator(project_dir=HydraConfig.get().runtime.output_dir, log_with="tensorboard")

class Trainer(object):
    def __init__(self, cfg: DictConfig):

        # Init
        self.cfg = cfg
        self.project_dir = Path(accelerator.project_dir)
        self.device = accelerator.device
        self.amp = self.cfg.training.training_cfg.get('mix_train', False)
        logger.info('Init Trainer')

        # Backup code, only on main process
        if self.cfg.backup_code and accelerator.is_main_process:
            back_dir = self.project_dir / '.src'
            logger.info('Backup code at: {}'.format(back_dir))
            cwd = HydraConfig.get().runtime.cwd
            src_dir = Path(cwd) / 'src'
            if back_dir.exists():
                shutil.rmtree(back_dir)
            shutil.copytree(src=src_dir, dst=back_dir)

        # Checkpoint dir
        self.ckptdir = self.project_dir / 'checkpoints'
        self.ckpt_last = self.project_dir / 'ckpt_last'

        # Check resume
        if self.cfg.resume and not self.ckpt_last.is_dir():
            self.cfg.resume = False
            logger.info('Resume FAILED, no ckpt dir at: {}'.format(self.ckpt_last))

        # Wandb tracker
        # deprecate for the moment becasue wandb can't resume from a specific step
        # accelerator.init_trackers(project_name=self.cfg.project_name,
        #                           config={"learning_rate": 0.001, "batch_size": 32},
        #                           name=self.cfg.exp_name, resume=self.cfg.resume)
        # self.tracker = accelerator.get_tracker("wandb")

        # Tensorboard tracker
        accelerator.init_trackers(project_name='tb')
        self.tracker = accelerator.get_trahydracker("tensorboard")
        logger.info('Tracker backend: tensorboard')
        
        # Preparation
        self.train_loader = self._get_data(self.cfg.dataset.name, self.cfg.dataset.train_tsv,
                                        self.cfg.dataset.dataset_cfg, self.cfg.dataset.dataloader_cfg, is_train=True)
        self.val_loader = self._get_data(self.cfg.dataset.name, self.cfg.dataset.val_tsv,
                                        self.cfg.dataset.dataset_cfg, self.cfg.dataset.dataloader_cfg, is_train=False)
        testdata_cfg = copy.deepcopy(self.cfg.dataset.dataset_cfg)
        testloader_cfg = copy.deepcopy(self.cfg.dataset.dataloader_cfg)
        testdata_cfg.chunk_size = -1
        testloader_cfg.batch_size = accelerator.num_processes
        self.test_loader = self._get_data(self.cfg.dataset.name, self.cfg.dataset.test_tsv,
                                        testdata_cfg, testloader_cfg, is_train=False)
        self.fs = self.cfg.dataset.dataset_cfg.fs
        logger.info('Wav sampling rate: {} Hz'.format(self.fs))

        self.model = self._get_model(self.cfg.model.name, self.cfg.model.model_cfg)
        self.optimizer = self._get_optim(self.model.parameters(), self.cfg.training.optimizer)
        self.scheduler = self._get_scheduler(self.optimizer, self.cfg.training.scheduler)
        self.tm = utils.TrainMonitor() # training record

        self.backbone = self._get_backbone()
        self.convert_audio = torchaudio.transforms.Resample(self.fs, self.backbone.sample_rate)
        self.reconvert_audio = torchaudio.transforms.Resample(self.backbone.sample_rate, self.fs)
        self._acc_prepare()

        # Define the loss function
        self.loss_func = self._get_loss(self.cfg.training.loss)
                
        # Resume
        if self.cfg.resume:
            logger.info('Resume training from: {}'.format(self.ckpt_last))
            accelerator.load_state(self.ckpt_last)
            self.tm.nb_iter += 1
            logger.info('Training re-start from iter: {}'.format(self.tm.nb_iter))
        else:
            logger.info('Experiment workdir: {}'.format(self.project_dir))
            logger.info('num_processes: {}'.format(accelerator.num_processes))
            bs_per_gpu = self.cfg.dataset.dataloader_cfg.batch_size // accelerator.num_processes
            logger.info('batch size per gpu: {}'.format(bs_per_gpu))
            logger.info('mixed_precision: {}'.format(accelerator.mixed_precision))
            logger.info(OmegaConf.to_yaml(self.cfg))
            logger.info('Trainer init finish')


    def _get_data(self, dataset_name, tsv_filepath, dataset_cfg, dataloader_cfg, is_train=True):
        if is_train:
            shuffle = True
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        logger.info(f'Prepare data at: {tsv_filepath}')
        data_class = getattr(datasets, f'Dataset{dataset_name}')
        dataset = data_class(tsv_filepath=tsv_filepath, **dataset_cfg)
        dataloader = DataLoader(dataset=dataset, shuffle=shuffle, drop_last=drop_last, **dataloader_cfg)

        return dataloader


    def _get_model(self, model_name, model_cfg):
        logger.info("Instantiating network model: {}".format(model_name))
        net_class = getattr(models, f'Net{model_name}')
        model = net_class(**model_cfg)

        total_params = sum(p.numel() for p in model.parameters())
        logger.info('Total params: {:.2f} Mb'.format(total_params / 1024 ** 2))
        return model
    

    def _get_optim(self, params, optim_cfg):
        
        optimizer = getattr(torch.optim, optim_cfg.name, None)
        if optimizer:
            cfg = optim_cfg.config
            return optimizer(params, 
                             lr=cfg.min_lr,
                             weight_decay=cfg.weight_decay)
        else:
            logger.error(f"Optimizer not found: {optim_cfg.name}")
    

    def _get_scheduler(self, optimizer, sche_cfg):

        scheduler = getattr(torch.optim.lr_scheduler, sche_cfg.name, None)
        if scheduler:
            return scheduler(optimizer, **sche_cfg.config)
        else:
            logger.error(f"Scheduler not found: {sche_cfg.name}")


    def _get_loss(self, loss_cfg):
        #TODO enable multiple losses
        logger.info("Instantiating Loss, Train: {}, Val: {}".format(
            loss_cfg.train.loss_func, loss_cfg.val.loss_func
        ))

        loss_func = {
            'train': getattr(losses, loss_cfg.train.loss_func),
            'val':  getattr(losses, loss_cfg.val.loss_func),
        }
        return loss_func

    
    def _get_backbone(self):
        target_bandwidths = [1.5, 3., 6, 12., 24.]
        sample_rate = 24_000
        channels = 1
        model = EncodecModel._get_model(target_bandwidths, sample_rate, channels,
            causal=True, model_norm='weight_norm', audio_normalize=False, name='encodec_24khz')

        cwd = HydraConfig.get().runtime.cwd
        state_dict = torch.load(f'{cwd}/pretrained/encodec_24khz-d7cc33bc.th', map_location='cpu')
        model.load_state_dict(state_dict)
        model.cuda()
        model.eval()
        model.set_target_bandwidth(24.)

        return model


    def _acc_prepare(self):
        self.model = accelerator.prepare(self.model)
        self.optimizer = accelerator.prepare(self.optimizer)
        self.scheduler = accelerator.prepare(self.scheduler)
        self.train_loader = accelerator.prepare(self.train_loader)
        self.val_loader = accelerator.prepare(self.val_loader)
        self.test_loader = accelerator.prepare(self.test_loader)
        accelerator.register_for_checkpointing(self.tm)

        self.backbone.to(accelerator.device)
        self.convert_audio.to(accelerator.device)
        self.reconvert_audio.to(accelerator.device)

        logger.info('{} iterations per epoch'.format(len(self.train_loader)))


    def run(self):
        
        training_cfg = self.cfg.training.training_cfg
        warmup_iter = training_cfg.warm_iter
        total_iter = training_cfg.total_iter
        print_iter = training_cfg.print_iter
        eval_iter = training_cfg.eval_iter
        test_iter = training_cfg.test_iter
        early_stop = training_cfg.early_stop
        max_lr = self.cfg.training.optimizer.config.max_lr
        grad_clip = self.cfg.training.optimizer.config.grad_clip
        cur_lr = [group['lr'] for group in self.optimizer.param_groups][0]

        self.model.train()
        
        logger.info('Warm up...')
        while self.tm.nb_iter <= warmup_iter:
            for batch in self.train_loader:
                
                # train on step
                loss, grad_norm = self.train_one_step(batch, grad_clip)
                cur_lr = utils.warmup_learning_rate(self.optimizer, self.tm.nb_iter, warmup_iter, max_lr)

                # print log
                if self.tm.nb_iter % print_iter == 0:
                    # tracker
                    msg_dict = {'Lr': cur_lr, 
                                'Train/loss': loss,
                                'Train/grad_norm': grad_norm}
                    self.tracker.log(msg_dict, step=self.tm.nb_iter)
                    # logger
                    msg = "Warmup iter {:d} : lr {:.8f}\t loss: {:.4f}\t grad_norm: {:.4f}".format(
                        self.tm.nb_iter, cur_lr, loss, grad_norm)
                    logger.info(msg)

                # eval
                if self.tm.nb_iter % eval_iter == 0:
                    metrics = self.run_eval()
                    # save best val
                    if metrics['loss'] < self.tm.best_eval:
                        self.tm.best_eval = metrics['loss']
                        self.tm.best_iter = self.tm.nb_iter
                        logger.info("\t-->Loss improved!!! Save best!!!")
                        accelerator.save_state(output_dir=self.ckpt_last)
                    # early stop
                    else:
                        self.tm.early_stop += 1
                        if self.tm.early_stop >= early_stop:
                            logger.info(f"\t--> Validation no imporved for {early_stop} times")
                            logger.info(f"\t--> Best model saved at iter: {self.tm.best_iter}")
                            logger.info(f"\t--> Training finished by early stop")
                            logger.info(f"\t--> Final test, load best ckpt")
                            accelerator.load_state(self.ckpt_last)
                            self.run_test()
                            return

                # test set
                if self.tm.nb_iter % test_iter == 0:
                    self.run_test()

                self.tm.nb_iter += 1
                if self.tm.nb_iter > warmup_iter: 
                    break

        logger.info('Training...')
        while self.tm.nb_iter <= total_iter:
            for batch in self.train_loader:
                loss, grad_norm = self.train_one_step(batch, grad_clip)

                # print log
                if self.tm.nb_iter % print_iter == 0:
                    # tracker
                    msg_dict = {'Lr': cur_lr, 
                                'Train/loss': loss,
                                'Train/grad_norm': grad_norm}
                    self.tracker.log(msg_dict, step=self.tm.nb_iter)
                    # logger
                    msg = "Train iter {:d} : lr {:.8f}\t loss: {:.4f}\t grad_norm: {:.4f}".format(
                        self.tm.nb_iter, cur_lr, loss, grad_norm)
                    logger.info(msg)

                # eval
                if self.tm.nb_iter % eval_iter == 0:
                    metrics = self.run_eval()
                    # save best val
                    if metrics['loss'] < self.tm.best_eval:
                        self.tm.best_eval = metrics['loss']
                        self.tm.best_iter = self.tm.nb_iter
                        logger.info("\t-->Loss improved!!! Save best!!!")
                        accelerator.save_state(output_dir=self.ckpt_last)
                    # early stop
                    else:
                        self.tm.early_stop += 1
                        if self.tm.early_stop >= early_stop:
                            logger.info(f"\t--> Validation no imporved for {early_stop} times")
                            logger.info(f"\t--> Best model saved at iter: {self.tm.best_iter}")
                            logger.info(f"\t--> Training finished by early stop")
                            logger.info(f"\t--> Final test, load best ckpt")
                            accelerator.load_state(self.ckpt_last)
                            self.run_test()
                            return

                # test set
                if self.tm.nb_iter % test_iter == 0:
                    self.run_test()

                self.tm.nb_iter += 1
                if self.tm.nb_iter > total_iter: 
                    logger.info(f"\t--> Training finished by reaching max iterations")
                    logger.info(f"\t--> Final test")
                    self.run_test()
                    return


    def train_one_step(self, batch, grad_clip):
        
        # Prepare
        x_noisy = batch['x_noisy'].unsqueeze(1)
        x_clean = batch['x_clean'].unsqueeze(1)
        x_noisy = self.convert_audio(x_noisy)
        x_clean = self.convert_audio(x_clean)
        
        # Forward
        with torch.no_grad():
            feat_noisy = self.backbone.encoder(x_noisy)
            feat_clean = self.backbone.encoder(x_clean)

            codes, scale = self.backbone.encode(x_noisy)[0]
            codes = codes.transpose(0, 1)
            embs = []
            for i, indices in enumerate(codes):
                layer = self.backbone.quantizer.vq.layers[i]
                quantized = layer.decode(indices)
                embs.append(quantized)
            embs = torch.stack(embs, dim=0) # (n_q, B, x_dim, T)

        feat_pred = self.model(feat_noisy, embs)

        # Compute loss
        if self.amp:
            with accelerator.autocast():
                loss = self.loss_func['train'](feat_pred, feat_clean)
        else:
            loss = self.loss_func['train'](feat_pred, feat_clean)

        # Gradient descent
        self.optimizer.zero_grad()
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            grad_norm = accelerator.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)
        self.optimizer.step()

        # Mean reduce
        loss = accelerator.reduce(loss, reduction="mean").item()
        grad_norm = accelerator.reduce(grad_norm, reduction="mean").item()

        return loss, grad_norm

    
    @torch.no_grad()
    def run_eval(self):
        """Distributed evaluation
        for inputs, targets in validation_dataloader:
            predictions = model(inputs)
            # Gather all predictions and targets
            all_predictions, all_targets = accelerator.gather_for_metrics((predictions, targets))
            # Example of use with a *Datasets.Metric*
            metric.add_batch(all_predictions, all_targets)
        """
        self.model.eval()
        am_loss = utils.AverageMeter()

        for batch in self.val_loader:
            
            x_noisy = batch['x_noisy'].unsqueeze(1)
            x_clean = batch['x_clean'].unsqueeze(1)
            x_noisy = self.convert_audio(x_noisy)
            x_clean = self.convert_audio(x_clean)

            feat_noisy = self.backbone.encoder(x_noisy)
            feat_clean = self.backbone.encoder(x_clean)

            # pred code
            codes, scale = self.backbone.encode(x_noisy)[0]
            codes = codes.transpose(0, 1)
            embs = []
            for i, indices in enumerate(codes):
                layer = self.backbone.quantizer.vq.layers[i]
                quantized = layer.decode(indices)
                embs.append(quantized)
            embs = torch.stack(embs, dim=0) # (n_q, B, x_dim, T)
            
            # Prediction
            feat_pred = self.model(feat_noisy, embs)
            all_predictions, all_targets = accelerator.gather_for_metrics((feat_pred, feat_clean))
            loss = self.loss_func['val'](all_predictions, all_targets)

            am_loss.update(loss.item())

        metrics = {}
        metrics['loss'] = am_loss.avg
        
        # print logs
        msg_dict = {}
        msg = f"Eval iter {self.tm.nb_iter:d}"
        for k, v in metrics.items():
            msg_dict[f'Val/{k}'] = v
            msg += '\t {} {:.4f}'.format(k, v)
        logger.info(msg)
        self.tracker.log(msg_dict, step=self.tm.nb_iter)

        self.model.train()
        return metrics
    

    @torch.no_grad()
    def run_test(self):
        self.model.eval()
        am_loss = utils.AverageMeter()
        am_sisdr = utils.AverageMeter()
        am_sisdr_i = utils.AverageMeter()
        am_pesq = utils.AverageMeter()

        for batch in self.test_loader:
            x_noisy = batch['x_noisy'].unsqueeze(1)
            x_clean = batch['x_clean'].unsqueeze(1)
            x_noisy = self.convert_audio(x_noisy)
            x_clean = self.convert_audio(x_clean)

            feat_noisy = self.backbone.encoder(x_noisy)
            feat_clean = self.backbone.encoder(x_clean)

            # pred code
            codes, scale = self.backbone.encode(x_noisy)[0]
            codes = codes.transpose(0, 1)
            embs = []
            for i, indices in enumerate(codes):
                layer = self.backbone.quantizer.vq.layers[i]
                quantized = layer.decode(indices)
                embs.append(quantized)
            embs = torch.stack(embs, dim=0) # (n_q, B, x_dim, T)

            # Prediction
            feat_pred = self.model(feat_noisy, embs)
            all_predictions, all_targets = accelerator.gather_for_metrics((feat_pred, feat_clean))
            loss = self.loss_func['val'](all_predictions, all_targets)

            # Recon audio quality eval
            x_recon = self.backbone.decoder(feat_pred)
            x_recon = x_recon[:,:,:x_clean.shape[-1]]
            x_noisy, x_clean, x_recon = accelerator.gather_for_metrics((x_noisy, x_clean, x_recon))

            x_noisy = self.reconvert_audio(x_noisy)
            x_clean = self.reconvert_audio(x_clean)
            x_recon = self.reconvert_audio(x_recon)

            sisdr = - losses.pairwise_neg_sisdr(x_recon, x_clean)
            sisdri = losses.pairwise_neg_sisdr(x_recon, x_noisy) - losses.pairwise_neg_sisdr(x_recon, x_clean)
            pesq_wb = pesq(16000, x_recon.squeeze().cpu().numpy(), x_clean.squeeze().cpu().numpy(), mode='wb') # suppose 16kHz

            am_loss.update(loss.item())
            am_sisdr.update(sisdr.mean().item())
            am_sisdr_i.update(sisdri.mean().item())
            am_pesq.update(pesq_wb)

        metrics = {}
        metrics['loss'] = am_loss.avg
        metrics['SI-SDR'] = am_sisdr.avg
        metrics['SI-SDR_i'] = am_sisdr_i.avg
        metrics['PESQ_WB'] = am_pesq.avg
        
        # print logs
        msg_dict = {}
        msg = f"Test iter {self.tm.nb_iter:d}"
        for k, v in metrics.items():
            msg_dict[f'Test/{k}'] = v
            msg += '\t {} {:.4f}'.format(k, v)
        logger.info(msg)
        self.tracker.log(msg_dict, step=self.tm.nb_iter)

        self.model.train()
        return


