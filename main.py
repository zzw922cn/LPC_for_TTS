

from audio import *
import numpy as np
from hparams import Hparams as hparams

input_wav_file = 'test.wav'
sample_rate = 24000
lpc_order = 8

orig_audio, pred_audio, residual, lpcs = lpc_audio(input_wav_file, lpc_order, hparams)

save_wav(pred_audio, 'wavs/pred.wav', hparams)
save_wav(orig_audio, 'wavs/orig.wav', hparams)
save_wav(residual, 'wavs/error.wav', hparams)
