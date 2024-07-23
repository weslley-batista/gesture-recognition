import threading
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
    print(f"Captured {len(frames)} frames.")
    return frames

def process_video(video):
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

# Thread for capturing frames
capture_thread = threading.Thread(target=lambda: globals().update({'frames_data': capture_frames()}))
capture_thread.start()

# Wait for the capture thread to complete
capture_thread.join()

# Check if frames_data is populated
if 'frames_data' not in globals():
    print("No frames captured.")
else:
    # Convert frames to images and stack them
    frames = [cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR) for frame in frames_data]
    video = np.stack(frames)

    # Thread for processing video
    process_thread = threading.Thread(target=lambda: process_video(video))
    process_thread.start()

    # Wait for the process thread to complete
    process_thread.join()
