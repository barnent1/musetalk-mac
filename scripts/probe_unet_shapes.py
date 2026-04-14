"""Probe the actual input shapes MuseTalk's UNet sees at runtime."""
import os, sys
ROOT = "/Users/barnent1/Projects/musetalk-mac"
os.chdir(os.path.join(ROOT, "upstream"))
sys.path.insert(0, os.path.join(ROOT, "upstream"))

import torch
from transformers import WhisperModel
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.utils import load_all_model, datagen

device = torch.device("mps")
vae, unet, pe = load_all_model(
    unet_model_path="./models/musetalkV15/unet.pth",
    vae_type="sd-vae",
    unet_config="./models/musetalkV15/musetalk.json",
    device=device,
)
pe = pe.to(device); vae.vae = vae.vae.to(device); unet.model = unet.model.to(device)
audio_processor = AudioProcessor(feature_extractor_path="./models/whisper")
weight_dtype = unet.model.dtype
whisper = WhisperModel.from_pretrained("./models/whisper").to(device=device, dtype=weight_dtype).eval()

features, length = audio_processor.get_audio_feature("data/audio/demo_tts.wav")
chunks = audio_processor.get_whisper_chunk(features, device, weight_dtype, whisper, length,
                                           fps=24, audio_padding_length_left=2, audio_padding_length_right=2)
print(f"whisper_chunks type: {type(chunks)}, len: {len(chunks)}, first shape: {chunks[0].shape}")
print(f"  dtype: {chunks[0].dtype}")

# Fake a latent of the right shape
import cv2
import numpy as np
img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
latents = vae.get_latents_for_unet(img)
print(f"single latent shape: {latents.shape}, dtype: {latents.dtype}")

# Stack batch of 16 (or however many chunks available) to mimic datagen output
B = min(16, len(chunks))
whisper_batch = chunks[:B].to(device)
latent_batch = torch.cat([latents] * B, dim=0)
print(f"whisper_batch shape: {whisper_batch.shape}, dtype: {whisper_batch.dtype}")
print(f"latent_batch shape:  {latent_batch.shape}, dtype: {latent_batch.dtype}")

# Pass through PE
audio_feat = pe(whisper_batch)
print(f"after PE (audio_feat): shape {audio_feat.shape}, dtype {audio_feat.dtype}")

# Call UNet
timesteps = torch.tensor([0], device=device)
with torch.no_grad():
    pred = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feat).sample
print(f"UNet output shape: {pred.shape}, dtype: {pred.dtype}")
