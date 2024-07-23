!pip install -q git+https://github.com/huggingface/transformers.git decord

from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode
import numpy as np
import cv2
import torch
from PIL import Image
from transformers import XCLIPProcessor, XCLIPModel

def capture_frames(num_frames=32, quality=0.8):
    js = Javascript('''
    async function captureFrames(num_frames, quality) {
        const video = document.createElement('video');
        const stream = await navigator.mediaDevices.getUserMedia({video: true});
        document.body.appendChild(video);
        video.srcObject = stream;
        await video.play();

        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        
        const frames = [];
        for (let i = 0; i < num_frames; i++) {
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            const dataUrl = canvas.toDataURL('image/jpeg', quality);
            frames.push(dataUrl);
            await new Promise(resolve => setTimeout(resolve, 100));  // wait 100ms between frames
        }

        stream.getVideoTracks()[0].stop();
        video.remove();
        return frames;
    }
    ''')
    display(js)
    data = eval_js('captureFrames({}, {})'.format(num_frames, quality))
    frames = [b64decode(frame.split(',')[1]) for frame in data]
    return frames

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

# Capture frames from webcam
frames_data = capture_frames()

# Convert frames to images and stack them
frames = [cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR) for frame in frames_data]
video = np.stack(frames)

# Model inference
model_name = "microsoft/xclip-base-patch16-zero-shot"
processor = XCLIPProcessor.from_pretrained(model_name)
model = XCLIPModel.from_pretrained(model_name)

inputs = processor(text=["programming course", "eating", "playing"], videos=list(video), return_tensors="pt", padding=True)

# forward pass
with torch.no_grad():
    outputs = model(**inputs)

probs = outputs.logits_per_video.softmax(dim=1)

# Print probabilities in percentage
for text, prob in zip(["programming course", "eating spaghetti", "dancing"], probs[0]):
    print(f"{text}: {prob.item() * 100:.2f}%")
