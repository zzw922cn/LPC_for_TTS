import numpy as np
from numpy import linalg as LA
import librosa
import librosa.filters
from scipy.io import wavfile
from scipy.fftpack import ifft


def load_wav(filefilepath, hparams):
    """ 这里推荐用wavfile并手动scale的方式,而librosa.core.load会产生较大误差 """

    _, raw_data = wavfile.read(filefilepath)
    raw_data = raw_data.astype(np.float32)
    float_data = (raw_data + 32768) / 65535. * 2 - 1

    return float_data


def save_wav(float_data, filepath, hparams):

    data = (float_data + 1) / 2 * 65535. - 32768
    wavfile.write(filepath, hparams.sample_rate, data.astype(np.int16))

def melspectrogram(wav, hparams):

    D = _stft(wav, hparams)

    S = _linear_to_mel(np.abs(D), hparams)

    S = normalize_spec(S)

    return S.astype(np.float32)

def normalize_spec(spectrogram):
  return np.log(1. + 10000 * spectrogram)

def denormalize_spec(spectrogram):
  return (np.exp(spectrogram) - 1.) / 10000

def denormed_melsp_to_linearsp(denormed_mel, hparams):
    """ 梅尔谱转化成线性谱 """
    denormed_mel = denormalize_spec(denormed_mel)

    return _mel_to_linear(denormed_mel, hparams)

def linearsp_to_autocorr(linearsp):
    power = linearsp**2
    fft_power = np.concatenate([power, power[::-1, :][1:-1, :]], axis=0)
    return ifft(fft_power, n=fft_power.shape[0], axis=0).real

def autocorr_to_lpc(ac, hparams, lpc_order):
    sample_rate = hparams.sample_rate
    ac = ac[0:lpc_order + 1, :]
    theta = (2 * np.pi * 40 / sample_rate)**2

    #  对自相关系数做平滑化处理
    lag_window = np.exp([[-0.5 * theta * i**2] for i in range(lpc_order + 1)])
    ac = ac * lag_window

    return levinson_durbin(lpc_order, ac)

def levinson_durbin(lpc_order, auto_corr):
    """ lpc_order: 阶数, auto_corr: 自相关系数 """

    num_frames = auto_corr.shape[-1]

    #  假设a_0=1,因此这里会多出一个元素
    Ak = np.zeros((lpc_order+1, num_frames), dtype=np.float32)
    Ak[0, :] = 1.0

    #  根据递推公式,可以反推出必有E0=R0
    E0 = np.copy(auto_corr[0, :])
    Ek = E0

    for k in range(lpc_order):
        lamb = 0.
        for j in range(k+1):
            lamb -= Ak[j, :] * auto_corr[k+1-j, :]

        lamb /= np.maximum(1e-6, Ek)

        #  根据a[n] = a[n] +lambda*a[k+1-n]
        #  每次赋值两个元素
        for n in range((k+1)//2+1):
            temp = Ak[k+1-n, :] + lamb * Ak[n, :]
            Ak[n, :] = Ak[n, :]+lamb*Ak[k+1-n, :]
            Ak[k+1-n, :] = temp

        Ek = Ek * (1-lamb**2)

    #  返回值不包含第一行的0
    return Ak[1:, :]


def lpc_predict(lpcs, signal_slice, clip_lpc=True):

    #  自回归线性组合
    pred = np.sum(lpcs * signal_slice, axis=0)

    if clip_lpc:
         pred = np.clip(pred, -1., 1.)

    return pred


def lpc_reconstruction(lpcs, lpc_order, audio):
    """ 从LPC中去恢复音频,并计算误差 """

    num_points = lpcs.shape[-1]

    if audio.shape[0] == num_points:
        #  起始点以0作为填充
        audio = np.pad(audio, ((lpc_order, 0)), 'constant')

    elif audio.shape[0] != num_points + lpc_order:
      raise RuntimeError('dimensions of lpcs and audio must match')

    indices = np.reshape(np.arange(lpc_order), [-1, 1]) + np.arange(
        lpcs.shape[-1])

    signal_slices = audio[indices]
    pred = lpc_predict(lpcs, signal_slices)
    origin_audio = audio[lpc_order:]

    error = origin_audio - pred

    return origin_audio, pred, error

def lpc_audio(input_wav_file, lpc_order, hparams):
    wav = load_wav(input_wav_file, hparams)

    #  根据自相关系数与功率谱之间的关系
    mel = melspectrogram(wav, hparams)
    linear = denormed_melsp_to_linearsp(mel, hparams)
    autocorr = linearsp_to_autocorr(linear)

    #  根据LD归纳算法计算LPC
    lpcs = autocorr_to_lpc(autocorr, hparams, lpc_order)

    #  根据自回归性质以及线性组合公式
    lpcs = -1 * lpcs[::-1, :]

    #  对每一帧都进行同样的自回归计算
    lpcs = np.repeat(lpcs, 240, axis=-1)
    lpcs = lpcs[:, :wav.shape[-1]]

    orig_audio, pred, error = lpc_reconstruction(lpcs, lpc_order, wav)

    return orig_audio, pred, error, lpcs

def _stft(y, hparams):

    return librosa.stft(y=y, n_fft=hparams.n_fft, hop_length=hparams.hop_size, win_length=hparams.win_size)

def _linear_to_mel(spectogram, hparams):

    mel_basis = _build_mel_basis(hparams)

    return np.dot(mel_basis, spectogram)

def _mel_to_linear(mel_spectrogram, hparams):

    _inv_mel_basis = np.linalg.pinv(_build_mel_basis(hparams))

    return np.maximum(1e-10, np.dot(_inv_mel_basis, mel_spectrogram))

def _build_mel_basis(hparams):

    return librosa.filters.mel(hparams.sample_rate, hparams.n_fft, n_mels=hparams.num_mels,
                               fmin=20, fmax=hparams.sample_rate/2)
