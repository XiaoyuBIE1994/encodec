#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2023 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inria.fr)
License agreement in LICENSE.txt
"""

import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
from torch.utils.data import Dataset
from accelerate.logging import get_logger

from ..utils import normalize_wav_np

logger = get_logger(__name__)

class DatasetVoiceBankDemand(Dataset):
    def __init__(self,
        tsv_filepath: str = "",
        fs: int = 16000,
        chunk_size: float = 2,
        use_noise: bool = False,
        normalize_audio: bool = False,
        **kwargs
    ) -> None:
        super().__init__()

        self.EPS = 1e-8
        self.tsv_filepath = Path(tsv_filepath)
        self.fs = fs
        self.chunk_size = chunk_size
        self.use_noise = use_noise
        self.normalize_audio = normalize_audio

        if self.chunk_size < 0:
            self.chunk_len = None
            self.run_test = True
        else:
            self.chunk_len = int(fs * chunk_size)
            self.run_test = False

        if self.tsv_filepath.is_file():
            metadata = pd.read_csv(self.tsv_filepath)
        else:
            logger.error('No tsv file found in: {}'.format(self.tsv_filepath))

        orig_utt = len(metadata)
        wav_lens = metadata['length'].map(lambda x: x / self.fs)
        orig_len, drop_utt, drop_len = 0, 0, 0
        drop_rows = []

        if not self.run_test:
            for row_idx in range(len(wav_lens)):
                orig_len += wav_lens[row_idx]
                if wav_lens[row_idx] < self.chunk_size:
                    drop_rows.append(row_idx)
                    drop_utt += 1
                    drop_len += wav_lens[row_idx]

            logger.info("Drop {}/{} utts ({:.2f}/{:.2f}h), shorter than {:.2f}s".format(
                drop_utt, orig_utt, drop_len / 3600, orig_len / 3600, self.chunk_size
            ))
            logger.info('Actual data size: {} utterance, ({:.2f} h)'.format(
                orig_utt-drop_utt, (orig_len-drop_len) / 3600
            ))

        
        self.metadata = metadata.drop(drop_rows)
        self.length = len(self.metadata)
        self.utt_len = orig_len - drop_len

        # check noise
        if self.use_noise and 'noise_path' not in metadata.columns:
            self.use_noise = False
            logger.error('noise_path not found, please load the noisy version of csv file')


    
    def __len__(self):
        return self.length
    

    def __getitem__(self, idx:int):

        mix_info = self.metadata.iloc[idx]

        # Slice wav files
        if self.chunk_size <= 0:
            start = 0
            stop = None
        else:
            start = np.random.randint(0, int(mix_info['length']) - self.chunk_len + 1)
            stop = start + self.chunk_len
            
        # Load wav files
        batch = {}
        batch['x_noisy'] = sf.read(
            mix_info['noisy_filepath'], start=start, stop=stop, dtype="float32"
        )[0]
        batch['x_clean'] = sf.read(
            mix_info['clean_filepath'], start=start, stop=stop, dtype="float32"
        )[0]

        if self.normalize_audio:
            m_std = np.std(batch['noisy_filepath'], axis=-1, keepdims=True)
            for k in batch.keys():
                batch[k] = normalize_wav_np(batch[k], self.EPS, m_std)

        return batch


if __name__ == '__main__':

    from accelerate import Accelerator
    from omegaconf import OmegaConf
    cfg = OmegaConf.create()
    accelerator = Accelerator()

    cfg.tsv_filepath = '/mnt/beegfs/robotlearn/xbie/VoiceBankDemand/test.tsv'
    cfg.fs = 16000
    cfg.chunk_size = 2.0
    cfg.normalize_audio = False

    dataset = DatasetVoiceBankDemand(**cfg)

    print('Total data: {}'.format(len(dataset)))

    idx = np.random.randint(dataset.__len__())
    data_ = dataset.__getitem__(idx)

    for k, v in data_.items():
        print('audio idx: {} audio: {}, length: {}'.format(idx, k, len(v)))


    
