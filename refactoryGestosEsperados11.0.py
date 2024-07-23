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
def take_photo_series_continuous(interval=5, quality=0.9):
    js = Javascript('''
        async function takePhotoSeriesContinuous(interval, quality) {
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
            const startTime = new Date().getTime();

            while (true) {
                const currentTime = new Date().getTime();
                if (currentTime - startTime >= interval * 1000) {
                    const canvas = document.createElement('canvas');
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    canvas.getContext('2d').drawImage(video, 0, 0);
                    frames.push(canvas.toDataURL('image/jpeg', quality));
                    if (frames.length >= 16) {
                        break;
                    }
                }
                await new Promise(resolve => setTimeout(resolve, 100)); // Aguarde 100ms antes de capturar o próximo frame
            }
            stream.getVideoTracks()[0].stop();
            div.remove();
            return frames;
        }
    ''')
    display(js)
    frames_data = eval_js(f'takePhotoSeriesContinuous({interval}, {quality})')
    frames = [b64decode(frame.split(',')[1]) for frame in frames_data]
    return frames

# Inicializa o processador e o modelo
model_name = "microsoft/xclip-base-patch32"
processor = XCLIPProcessor.from_pretrained(model_name)
model = XCLIPModel.from_pretrained(model_name)

def pad_frames(frames, size=(224, 224), target_length=8):
    padded_frames = [Image.fromarray(frame).resize(size) for frame in frames]
    while len(padded_frames) < target_length:
        padded_frames.append(Image.new('RGB', size))  # Adiciona um frame vazio se necessário
    return padded_frames[:target_length]

def process_video(frames):
    padded_frames = pad_frames(frames)
    padded_frames = [np.array(frame) for frame in padded_frames]
    inputs = processor(text=["raising arms", "crossed arms", "raising leg"], videos=[padded_frames], return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    probs = outputs.logits_per_video.softmax(dim=1)
    return probs

try:
    while True:
        # Captura frames continuamente a cada 5 segundos
        frames_data = take_photo_series_continuous(interval=5)

        # Converte frames capturados em arrays numpy
        frames = [np.array(Image.open(io.BytesIO(frame)).convert("RGB")) for frame in frames_data]

        if len(frames) > 0:
            probs = process_video(frames)
            percentages = probs * 100
            print('Probabilidades em porcentagens: ', percentages)

            # Exibe os frames capturados
            #for frame in frames:
            #    display(Image.fromarray(frame))

except Exception as err:
    print(str(err))
