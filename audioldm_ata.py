from imagebind.imagebind.models.imagebind_model import ModalityType
from imagebind.imagebind.models import imagebind_model 
from imagebind.imagebind import data
from audioldm_train.utilities.model_util import instantiate_from_config
import torch
import yaml
import os
from pytorch_lightning import seed_everything
import soundfile as sf
from audioldm_train.conditional_models import CLAPAudioEmbeddingClassifierFreev2
import contextlib
import wave
import torchaudio
import numpy as np
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model(
    ckpt_path=None,
    config=None,
    model_name="audioldm-s-full.ckpt"
):
    print("Load AudioLDM: %s", model_name)
    # No normalization here
    config_yaml_path = os.path.join(config)

    if config is not None:
        assert type(config) is str
        config = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
    
    exp_name = os.path.basename(config_yaml_path.split(".")[0])
    exp_group_name = os.path.basename(os.path.dirname(config_yaml_path))
    log_path = config["log_directory"]

    latent_diffusion = instantiate_from_config(config["model"])
    latent_diffusion.set_log_dir(log_path, exp_group_name, exp_name)
    resume_from_checkpoint = ckpt_path

    checkpoint = torch.load(resume_from_checkpoint, map_location=device)
    '''Original. Here is a bug that, an unexpected key "cond_stage_model.model.text_branch.embeddings.position_ids" exists in the checkpoint file. '''
    # latent_diffusion.load_state_dict(checkpoint["state_dict"])
    '''2023.10.17 Fix the bug by setting the paramer "strict" as "False" to ignore the unexpected key. '''
    latent_diffusion.load_state_dict(checkpoint["state_dict"], strict=False)
    latent_diffusion.eval()
    latent_diffusion = latent_diffusion.to(device)
    return latent_diffusion

def duration_to_latent_t_size(duration):
    return int(duration * 25.6)

def set_cond_audio(latent_diffusion):
    # print(dir(latent_diffusion))
    # fixed , get the right key , don't know if it is right
    latent_diffusion.cond_stage_model_metadata["film_clap_cond1"]["cond_stage_key"] = "waveform"
    latent_diffusion.cond_stage_models[0].embed_mode="audio"

    return latent_diffusion

def set_cond_text(latent_diffusion):
    latent_diffusion.cond_stage_model_metadata["film_clap_cond1"]["cond_stage_key"] = "text"
    latent_diffusion.cond_stage_models[0].embed_mode="text"
    return latent_diffusion

# video_file="quack.mp4"

# # Instantiate model
# model = imagebind_model.imagebind_huge(pretrained=True)
# model.eval()
# model.to(device)
# # Load data
# inputs = {
#     ModalityType.VISION: data.load_and_transform_video_data(
#         [video_file], 
#         device,
#         clip_duration=1,  # Increased from 2 to 5 seconds for better temporal context
#         clips_per_video=1  # Increased from 5 to 10 clips for better coverage
#     )
# }

# with torch.no_grad():
#     embeddings = model(inputs)[ModalityType.VISION]

# print("emb dimenstion is " , embeddings.shape)
# print("emb is " , embeddings)

# mock_vision_embedding = torch.randn(1, 1024).to(device)  

# ==== Construct Batch with Vision Input ====



def make_batch_form_text_to_audio(text, waveform=None, fbank=None, batchsize=1, fname="reproduce"):
    text = [text] * batchsize

    if batchsize < 1:
        print("Warning: Batchsize must be at least 1. Setting to 1.")
        batchsize = 1
    
    if fbank is None:
        fbank = torch.zeros((batchsize, 1024, 64))
    else:
        fbank = torch.FloatTensor(fbank)
        fbank = fbank.expand(batchsize, 1024, 64)
        assert fbank.size(0) == batchsize
        
    stft = torch.zeros((batchsize, 1024, 512))

    if waveform is None:
        waveform = torch.zeros((batchsize, 160000))
    else:
        waveform = torch.FloatTensor(waveform)
        waveform = waveform.expand(batchsize, -1)
        assert waveform.size(0) == batchsize

    fname = [fname] * batchsize

    batch = {
        "log_mel_spec": fbank,
        "stft": stft,
        "label_vector": None,
        "fname": fname,
        "waveform": waveform,
        "text": text
    }

    return batch

def normalize_wav(waveform):
    waveform = waveform - np.mean(waveform)
    waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
    return waveform * 0.5

def pad_wav(waveform, segment_length):
    waveform_length = waveform.shape[-1]
    assert waveform_length > 100, "Waveform is too short, %s" % waveform_length
    if segment_length is None or waveform_length == segment_length:
        return waveform
    elif waveform_length > segment_length:
        return waveform[:segment_length]
    elif waveform_length < segment_length:
        temp_wav = np.zeros((1, segment_length))
        temp_wav[:, :waveform_length] = waveform
    return temp_wav

def read_wav_file(filename, segment_length):
    # waveform, sr = librosa.load(filename, sr=None, mono=True) # 4 times slower
    waveform, sr = torchaudio.load(filename)  # Faster!!!
    waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
    waveform = waveform.numpy()[0, ...]
    waveform = normalize_wav(waveform)
    waveform = waveform[None, ...]
    waveform = pad_wav(waveform, segment_length)
    
    waveform = waveform / np.max(np.abs(waveform))
    waveform = 0.5 * waveform
    
    return waveform

def seed_everything(seed):


    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

unconditional_guidance_scale=2.5
ddim_sampling_steps= 200
n_candidates_per_samples= 3
duration = 5
waveform=None
checkpoint = "data/checkpoints/audioldm-s-full.ckpt"
audio_ldm_yaml = "audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original.yaml"
# config_yaml = "audioldm_train/config/audioldm_original_with_imagebind/audioldm_with_imagebind.yaml"
check_pt_path = "log/latent_diffusion/audioldm_original_with_imagebind/audioldm_with_imagebind/checkpoints/checkpoint-fad-133.00-global_step=244999.ckpt"
video_file = "quack.mp4"
audio_path = "trumpet.wav"
text = "" # must be string

# ==== Load Latent Diffusion Model from Checkpoint ====
seed_everything(42)

audioldm = build_model(checkpoint ,audio_ldm_yaml, "audioldm-s")
audioldm.latent_t_size = duration_to_latent_t_size(duration)


waveform = None

if(audio_path is not None):
    waveform = read_wav_file(audio_path, int(duration * 102.4) * 160)
# print("Waveform shape:", waveform.shape)
# print("Waveform checksum:", np.sum(waveform))


if waveform is not None:
    print("Generate audio that has similar content as %s" % audio_path)
    audioldm = set_cond_audio(audioldm)
else:
    print("Generate audio using text %s" % text)
    audioldm = set_cond_text(audioldm)

batch = make_batch_form_text_to_audio(text,waveform=waveform , fname="trumpet")

print(batch)

with torch.no_grad():
    waveform = audioldm.generate_sample(
        [batch],
        unconditional_guidance_scale=unconditional_guidance_scale,
        ddim_steps=ddim_sampling_steps,
        n_candidate_gen_per_text=n_candidates_per_samples,
        duration=duration,
        unconditional_prob= 0.0,
        )


