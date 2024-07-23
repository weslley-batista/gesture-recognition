!pip install -q git+https://github.com/huggingface/transformers.git decord
!pip install -q pytube
!pip install -q yt-dlp

from yt_dlp import YoutubeDL

youtube_url = 'https://youtu.be/3nrGQszyAnQ'

ydl_opts = {
    'format': 'mp4',
    'outtmpl': 'downloaded_video.mp4'
}

with YoutubeDL(ydl_opts) as ydl:
    ydl.download([youtube_url])

file_path = 'downloaded_video.mp4'

from decord import VideoReader, cpu
import torch
import numpy as np

from huggingface_hub import hf_hub_download

np.random.seed(0)

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

videoreader = VideoReader(file_path, num_threads=1, ctx=cpu(0))

# sample 32 frames
videoreader.seek(0)
indices = sample_frame_indices(clip_len=32, frame_sample_rate=4, seg_len=len(videoreader))
video = videoreader.get_batch(indices).asnumpy()

video.shape


from PIL import Image

Image.fromarray(video[0])

from transformers import XCLIPProcessor, XCLIPModel

model_name = "microsoft/xclip-base-patch16-zero-shot"
processor = XCLIPProcessor.from_pretrained(model_name)
model = XCLIPModel.from_pretrained(model_name)

import torch

inputs = processor(text=["programming course", "eating spaghetti", "dancing"], videos=list(video), return_tensors="pt", padding=True)

# forward pass
with torch.no_grad():
    outputs = model(**inputs)

probs = outputs.logits_per_video.softmax(dim=1)

# Imprimir as probabilidades em porcentagem
for text, prob in zip(["programming course", "eating spaghetti", "dancing"], probs[0]):
    print(f"{text}: {prob.item() * 100:.2f}%")
