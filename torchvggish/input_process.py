import numpy as np
import torch
import torchaudio.transforms as ta_trans
from . import vggish_params
from .tf_input_process.vggish_input import waveform_to_examples


class VGGishLogMelSpectrogram(ta_trans.MelSpectrogram):
    '''
    This is a _log_ mel-spectrogram transform that adheres to the transform
    used by Google's vggish model input processing pipeline
    '''
    def forward(self, waveform):
        r"""
        Args:
            waveform (torch.Tensor): Tensor of audio of dimension (..., time)

        Returns:
            torch.Tensor: Mel frequency spectrogram of size (..., ``n_mels``, time)
        """
        specgram = self.spectrogram(waveform)
        specgram = specgram ** 0.5  # TODO: document this hack later!!
        mel_specgram = self.mel_scale(specgram)
        mel_specgram = torch.log(mel_specgram + vggish_params.LOG_OFFSET)
        return mel_specgram


def torch_log_melspec_ope():
    audio_sample_rate = vggish_params.SAMPLE_RATE
    window_length_samples = int(round(
        audio_sample_rate * vggish_params.STFT_WINDOW_LENGTH_SECONDS
    ))
    hop_length_samples = int(round(
        audio_sample_rate * vggish_params.STFT_HOP_LENGTH_SECONDS
    ))
    fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))
    assert window_length_samples == 400
    assert hop_length_samples == 160
    assert fft_length == 512
    mel_trans = VGGishLogMelSpectrogram(
        vggish_params.SAMPLE_RATE, n_fft=fft_length,
        win_length=window_length_samples, hop_length=hop_length_samples,
        f_min=vggish_params.MEL_MIN_HZ, f_max=vggish_params.MEL_MAX_HZ,
        n_mels=vggish_params.NUM_BANDS
    )
    return mel_trans


def waveform_to_input(waveform, sample_rate, method='torch'):
    '''
    Args:
        waveform: [num_channels, num_steps]
        sample_rate: per second sample rate
    '''
    assert method in ('torch', 'tf')
    func = _torch_process_input if method == 'torch' else _tf_process_input
    return func(waveform, sample_rate, reshape_into_batches=True)


def _torch_process_input(waveform, sample_rate, reshape_into_batches=False):
    # init the ops
    resampler = ta_trans.Resample(sample_rate, vggish_params.SAMPLE_RATE)
    mel_trans = torch_log_melspec_ope()

    x = waveform.mean(axis=0, keepdims=True)  # average over channels
    x = resampler(x)
    x = mel_trans(x)  # [1, num_freq, time_steps]
    x = x.squeeze(dim=0).T  # [time_steps, num_freq]
    if reshape_into_batches:
        window_size = 96
        length = len(x)
        batch_size = length // window_size
        num_frames_to_use = batch_size * window_size
        x = x[:num_frames_to_use, :]
        x = x.reshape(batch_size, 1, window_size, x.shape[-1])
    return x


def _tf_process_input(waveform, sample_rate, reshape_into_batches=False):
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.T.numpy()
    x = waveform_to_examples(waveform, sample_rate)
    x = torch.from_numpy(x).unsqueeze(dim=1).float()
    if reshape_into_batches:
        pass  # tf does batching by default
    else:
        x = x.reshape(-1, x.shape[-1])
    return x
