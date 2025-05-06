from audioldm_train.utilities.model_util import instantiate_from_config
import torch
import yaml
import os
from pytorch_lightning import seed_everything
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

def set_cond_video(latent_diffusion):
    print(dir(latent_diffusion))
    latent_diffusion.cond_stage_model_metadata["film_imagebind_cond1"]["cond_stage_key"] = "fname"
    latent_diffusion.cond_stage_models[0].emb_mode="video"
    return latent_diffusion

def set_cond_audio(latent_diffusion):
    # print(dir(latent_diffusion))

    latent_diffusion.cond_stage_model_metadata["film_imagebind_cond1"]["cond_stage_key"] = "fname"
    latent_diffusion.cond_stage_models[0].emb_mode="audio"
    return latent_diffusion
def set_latent_t_size(latent_diffusion, duration):
    latent_diffusion.latent_t_size = int(duration * 25.6)
    return latent_diffusion
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
    
def seed_everything(seed):


    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

unconditional_guidance_scale=2.5
ddim_sampling_steps= 201
n_candidates_per_samples= 3
duration = 5
waveform=None
audio_ldm_yaml = "audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original_medium.yaml"
config_yaml = "audioldm_train/config/audioldm_original_with_imagebind/audioldm_with_imagebind.yaml"
check_pt_path = "log/latent_diffusion/audioldm_original_with_imagebind/audioldm_with_imagebind/checkpoints/checkpoint-fad-133.00-global_step=209999.ckpt"
video_file = "quack.mp4"
audio_path = "trumpet.wav"
text = "" # must be string

# ==== Load Latent Diffusion Model from Checkpoint ====
seed_everything(42)

audioldm = build_model(check_pt_path ,config_yaml, "imagebind")

audioldm=set_cond_audio(audioldm)
audioldm = set_latent_t_size(audioldm , duration)
waveform = None
batch = make_batch_form_text_to_audio(text,waveform=waveform , fname=audio_path)

with torch.no_grad():
    waveform = audioldm.generate_sample(
        [batch],
        unconditional_guidance_scale=unconditional_guidance_scale,
        ddim_steps=ddim_sampling_steps,
        n_candidate_gen_per_text=n_candidates_per_samples,
        duration=duration,
        )


