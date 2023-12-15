import torchaudio
import pandas as pd
from glob import glob
from tqdm import tqdm
from pathlib import Path
import soundfile as sf


data_dir = Path('/mnt/beegfs/robotlearn/xbie/VoiceBankDemand')

train_dir = data_dir / 'clean_trainset_26spk_wav_16k'
val_dir = data_dir / 'clean_valset_2spk_wav_16k'
test_dir = data_dir / 'clean_testset_wav_16k'

# read meta txt
meta_dict = {}
meta_file = data_dir / 'log_trainset_28spk.txt'
with open(meta_file, 'r') as f:
    for line in f:
        data = line.strip().split()
        meta_dict[data[0]] = {'noise_type': data[1], 'snr': float(data[2])}

meta_file = data_dir / 'log_testset.txt'
with open(meta_file, 'r') as f:
    for line in f:
        data = line.strip().split()
        meta_dict[data[0]] = {'noise_type': data[1], 'snr': float(data[2])}


# train
with open('train.tsv', 'w') as f:
    f.write('filename,clean_filepath,noisy_filepath,speaker,noise_type,snr,length\n')
    for filepath in tqdm(sorted(train_dir.glob('*.wav')), desc='train'):
        x, fs = sf.read(filepath)
        lens = len(x)
        noisy_filepath = str(filepath).replace('clean', 'noisy')
        filename = filepath.stem[:-4]
        speaker = filename[:4]
        noise_type = meta_dict[filename]['noise_type']
        snr = meta_dict[filename]['snr']
        line = '{},{},{},{},{},{:.1f},{}\n'.format(filename, filepath, noisy_filepath, speaker, noise_type, snr, lens)
        f.write(line)

# val
with open('val.tsv', 'w') as f:
    f.write('filename,clean_filepath,noisy_filepath,speaker,noise_type,snr,length\n')
    for filepath in tqdm(sorted(val_dir.glob('*.wav')), desc='val'):
        x, fs = sf.read(filepath)
        lens = len(x)
        noisy_filepath = str(filepath).replace('clean', 'noisy')
        filename = filepath.stem[:-4]
        speaker = filename[:4]
        noise_type = meta_dict[filename]['noise_type']
        snr = meta_dict[filename]['snr']
        line = '{},{},{},{},{},{:.1f},{}\n'.format(filename, filepath, noisy_filepath, speaker, noise_type, snr, lens)
        f.write(line)

# test
with open('test.tsv', 'w') as f:
    f.write('filename,clean_filepath,noisy_filepath,speaker,noise_type,snr,length\n')
    for filepath in tqdm(sorted(test_dir.glob('*.wav')), desc='test'):
        x, fs = sf.read(filepath)
        lens = len(x)
        noisy_filepath = str(filepath).replace('clean', 'noisy')
        filename = filepath.stem[:-4]
        speaker = filename[:4]
        noise_type = meta_dict[filename]['noise_type']
        snr = meta_dict[filename]['snr']
        line = '{},{},{},{},{},{:.1f},{}\n'.format(filename, filepath, noisy_filepath, speaker, noise_type, snr, lens)
        f.write(line)

import time


while True:
    print('\rWaiting---', end='')
    time.sleep(1)