"""
article{Forsgren_Martiros_2022,
  author = {Forsgren, Seth* and Martiros, Hayk*},
  title = {{Riffusion - Stable diffusion for real-time music generation}},
  url = {https://riffusion.com/about},
  year = {2022}
}"""
import warnings

import numpy as np
import pydub
import torch
import torchaudio

import argparse

from spectrogram_params import SpectrogramParams
from  

#code to convert an audio file to a spectrogram

class SpectrogramConverter:
    """
    Convert between audio segments and spectrogram tensors using torchaudio.

    In this class a "spectrogram" is defined as a (batch, time, frequency) tensor with float values
    that represent the amplitude of the frequency at that time bucket (in the frequency domain).
    Frequencies are given in the perceptul Mel scale defined by the params. A more specific term
    used in some functions is "mel amplitudes".

    The spectrogram computed from `spectrogram_from_audio` is complex valued, but it only
    returns the amplitude, because the phase is chaotic and hard to learn. The function
    `audio_from_spectrogram` is an approximate inverse of `spectrogram_from_audio`, which
    approximates the phase information using the Griffin-Lim algorithm.

    Each channel in the audio is treated independently, and the spectrogram has a batch dimension
    equal to the number of channels in the input audio segment.

    Both the Griffin Lim algorithm and the Mel scaling process are lossy.

    For more information, see https://pytorch.org/audio/stable/transforms.html
    """

    def __init__(self, params: SpectrogramParams, device: str = "cuda"):
        self.p = params

        self.device = torch_util.check_device(device)

        if device.lower().startswith("mps"):
            warnings.warn(
                "WARNING: MPS does not support audio operations, falling back to CPU for them",
                stacklevel=2,
            )
            self.device = "cpu"

        # https://pytorch.org/audio/stable/generated/torchaudio.transforms.Spectrogram.html
        self.spectrogram_func = torchaudio.transforms.Spectrogram(
            n_fft=params.n_fft,
            hop_length=params.hop_length,
            win_length=params.win_length,
            pad=0,
            window_fn=torch.hann_window,
            power=None,
            normalized=False,
            wkwargs=None,
            center=True,
            pad_mode="reflect",
            onesided=True,
        ).to(self.device)

        # https://pytorch.org/audio/stable/generated/torchaudio.transforms.GriffinLim.html
        self.inverse_spectrogram_func = torchaudio.transforms.GriffinLim(
            n_fft=params.n_fft,
            n_iter=params.num_griffin_lim_iters,
            win_length=params.win_length,
            hop_length=params.hop_length,
            window_fn=torch.hann_window,
            power=1.0,
            wkwargs=None,
            momentum=0.99,
            length=None,
            rand_init=True,
        ).to(self.device)

        # https://pytorch.org/audio/stable/generated/torchaudio.transforms.MelScale.html
        self.mel_scaler = torchaudio.transforms.MelScale(
            n_mels=params.num_frequencies,
            sample_rate=params.sample_rate,
            f_min=params.min_frequency,
            f_max=params.max_frequency,
            n_stft=params.n_fft // 2 + 1,
            norm=params.mel_scale_norm,
            mel_scale=params.mel_scale_type,
        ).to(self.device)

        # https://pytorch.org/audio/stable/generated/torchaudio.transforms.InverseMelScale.html
        self.inverse_mel_scaler = torchaudio.transforms.InverseMelScale(
            n_stft=params.n_fft // 2 + 1,
            n_mels=params.num_frequencies,
            sample_rate=params.sample_rate,
            f_min=params.min_frequency,
            f_max=params.max_frequency,
            max_iter=params.max_mel_iters,
            tolerance_loss=1e-5,
            tolerance_change=1e-8,
            sgdargs=None,
            norm=params.mel_scale_norm,
            mel_scale=params.mel_scale_type,
        ).to(self.device)

    def audio_to_spectrogram(audio: pydub.AudioSegment) -> np.ndarray:
        """
        Compute a spectrogram from an audio segment.

        Args:
            audio: Audio segment which must match the sample rate of the params

        Returns:
            spectrogram: (channel, frequency, time)
        """

        #hardcoding sample rate for now
        assert int(audio.frame_rate) == 44100, "Audio sample rate must match params"

        # Get the samples as a numpy array in (batch, samples) shape
        waveform = np.array([c.get_array_of_samples() for c in audio.split_to_mono()])

        # Convert to floats if necessary
        if waveform.dtype != np.float32:
            waveform = waveform.astype(np.float32)

        waveform_tensor = torch.from_numpy(waveform).to("cpu")
        amplitudes_mel = mel_amplitudes_from_waveform(waveform_tensor)
        return amplitudes_mel.cpu().numpy()

    def mel_amplitudes_from_waveform(
            self,
            waveform: torch.Tensor,
        ) -> torch.Tensor:
            """
            Torch-only function to compute Mel-scale amplitudes from a waveform.

            Args:
                waveform: (batch, samples)

            Returns:
                amplitudes_mel: (batch, frequency, time)
            """
            # Compute the complex-valued spectrogram
            spectrogram_complex = self.spectrogram_func(waveform)

            # Take the magnitude
            amplitudes = torch.abs(spectrogram_complex)

            # Convert to mel scale
            return self.mel_scaler(amplitudes)


    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('filename')

    args = parser.parse_args()

    print(args.filename)

    
