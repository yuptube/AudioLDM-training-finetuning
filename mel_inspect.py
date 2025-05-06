import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

# === CONFIGURATION ===
FMAX = 16000
MEL_BIN = 64
OUTPUT_DIR = 'waveform_comparison'
AI_WAVS =[ "log/latent_diffusion/audioldm_original_with_imagebind/audioldm_with_imagebind/infer_04-30-16:00_cfg_scale_3.5_ddim_200_n_cand_3/trumpet.wav"]


os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_waveform_and_mel(wav_paths, output_path, mel_bins=64, fmax=8000):
    n_files = len(wav_paths)
    fig, axs = plt.subplots(2, n_files, figsize=(5 * n_files, 8))

    if n_files == 1:
        axs = np.expand_dims(axs, axis=1)  # <-- FIX: axs always 2D!

    for idx, wav_path in enumerate(wav_paths):
        y, sr = librosa.load(wav_path, sr=None)
        time_axis = np.linspace(0, len(y) / sr, num=len(y))

        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=mel_bins, fmax=fmax)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        axs[0, idx].plot(time_axis, y)
        axs[0, idx].set_title(f'Waveform {idx}')
        axs[0, idx].set_xlabel('Time (s)')
        axs[0, idx].set_ylabel('Amplitude')

        img = librosa.display.specshow(mel_spec_db, sr=sr, ax=axs[1, idx], x_axis='time', y_axis='mel', fmax=fmax)
        axs[1, idx].set_title(f'Mel Spectrogram {idx}')
        fig.colorbar(img, ax=axs[1, idx], format="%+2.0f dB")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

# === RUN ===
output_file = os.path.join(OUTPUT_DIR, f'waveform_and_mel_comparison_{MEL_BIN}_{FMAX}_600k.png')
plot_waveform_and_mel(AI_WAVS, output_file, mel_bins=MEL_BIN, fmax=FMAX)

print(f"Saved comparison plot at: {output_file}")