from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode
import numpy as np
from transformers import XCLIPProcessor, XCLIPModel
import torch
from PIL import Image
import io
import time

# Função para capturar uma série de fotos automaticamente
def take_photo_series(num_frames=8, quality=0.8):
    js = Javascript('''
        async function takePhotoSeries(num_frames, quality) {
            const div = document.createElement('div');
            const video = document.createElement('video');
            video.style.display = 'block';
            const stream = await navigator.mediaDevices.getUserMedia({video: true});
            document.body.appendChild(div);
            div.appendChild(video);
            video.srcObject = stream;
            await video.play();
            google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

            const frames = [];
            for (let i = 0; i < num_frames; i++) {
                const canvas = document.createElement('canvas');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                canvas.getContext('2d').drawImage(video, 0, 0);
                frames.push(canvas.toDataURL('image/jpeg', quality));
                await new Promise(resolve => setTimeout(resolve, 1000)); // Captura uma frame por segundo
            }
            stream.getVideoTracks()[0].stop();
            div.remove();
            return frames;
        }
    ''')
    display(js)
    frames_data = eval_js(f'takePhotoSeries({num_frames}, {quality})')
    frames = [b64decode(frame.split(',')[1]) for frame in frames_data]
    return frames

# Inicializa o processador e o modelo
model_name = "microsoft/xclip-base-patch32"
processor = XCLIPProcessor.from_pretrained(model_name)
model = XCLIPModel.from_pretrained(model_name)

def process_video(frames):
#    inputs = processor(text=["three fingers", "open hand", "OK sign"], videos=frames, return_tensors="pt")
    inputs = processor(text=["playing sports", "eating spaghetti", "go shopping"], videos=list(frames), return_tensors="pt", padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)

    probs = outputs.logits_per_video.softmax(dim=1)
    return probs

try:
    # Captura exatamente 8 frames
    frames_data = take_photo_series(num_frames=8)
    
    # Converte frames capturados em arrays numpy
    frames = [np.array(Image.open(io.BytesIO(frame)).convert("RGB")) for frame in frames_data]

    if len(frames) > 0:
        probs = process_video(frames)
        print('Probabilidade: ', probs)

        # Exibe os frames capturados
        for frame in frames:
            display(Image.fromarray(frame))

except Exception as err:
    print(str(err))
