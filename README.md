# LPC_for_TTS
Linear Prediction Coefficients estimation from mel-spectrogram implemented in Python based on Levinson-Durbin algorithm.

基于Levinson-Durbin归纳法来做线性预测系数的估计。此代码可用于LPC系数的估计，也可用于LPCNet等合成器的特征提取。流程是从音频得到梅尔谱，梅尔谱得到LPC。

```Python
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
```

Raw audio:
![image](https://user-images.githubusercontent.com/11649939/111761869-562df580-88db-11eb-933f-be4c07712d25.png)

Predicted audio:
![image](https://user-images.githubusercontent.com/11649939/111761943-67770200-88db-11eb-957d-73197d9e4e46.png)

Prediction error:
![image](https://user-images.githubusercontent.com/11649939/111762018-7b226880-88db-11eb-9efb-43c32ff942ae.png)
