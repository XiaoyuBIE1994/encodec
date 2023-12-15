from encodec import EncodecModel
from encodec.utils import convert_audio

import torchaudio
import torch

# Instantiate a pretrained EnCodec model
# model = EncodecModel.encodec_model_24khz()
# The number of codebooks used will be determined bythe bandwidth selected.
# E.g. for a bandwidth of 6kbps, `n_q = 8` codebooks are used.
# Supported bandwidths are 1.5kbps (n_q = 2), 3 kbps (n_q = 4), 6 kbps (n_q = 8) and 12 kbps (n_q =16) and 24kbps (n_q=32).
# For the 48 kHz model, only 3, 6, 12, and 24 kbps are supported. The number
# of codebooks for each is half that of the 24 kHz model as the frame rate is twice as much.

target_bandwidths = [1.5, 3., 6, 12., 24.]
sample_rate = 24_000
channels = 1
model = EncodecModel._get_model(target_bandwidths, sample_rate, channels,
            causal=True, model_norm='weight_norm', audio_normalize=False, name='encodec_24khz')

state_dict = torch.load('pretrained/encodec_24khz-d7cc33bc.th', map_location='cpu')
model.load_state_dict(state_dict)
model.eval()
model.set_target_bandwidth(6.0)

# Load and pre-process the audio waveform
wav, sr = torchaudio.load("/mnt/beegfs/robotlearn/xbie/VoiceBankDemand/noisy_testset_wav_16k/p232_001_16k.wav")
wav = convert_audio(wav, sr, model.sample_rate, model.channels)
wav = wav.unsqueeze(0)

# Extract discrete codes from EnCodec
with torch.no_grad():
    # encoded_frames = model.encode(wav)
    emb = model.encoder(wav) # [B, x_dim, T]
    print(emb.shape)
breakpoint()
codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]