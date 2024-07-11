from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode
import numpy as np
from transformers import XCLIPProcessor, XCLIPModel
import torch
from PIL import Image
import io

# Função para capturar uma foto
def take_photo(quality=0.8):
    js = Javascript('''
        async function takePhoto(quality) {
            const div = document.createElement('div');
            const capture = document.createElement('button');
            capture.textContent = 'Capture';
            div.appendChild(capture);

            const video = document.createElement('video');
            video.style.display = 'block';
            const stream = await navigator.mediaDevices.getUserMedia({video: true});

            document.body.appendChild(div);
            div.appendChild(video);
            video.srcObject = stream;
            await video.play();

            // Resize the output to fit the video element.
            google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

            // Wait for Capture to be clicked.
            await new Promise((resolve) => capture.onclick = resolve);

            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);
            stream.getVideoTracks()[0].stop();
            const dataUrl = canvas.toDataURL('image/jpeg', quality);
            div.remove();
            return dataUrl;
        }
    ''')
    display(js)
    data = eval_js('takePhoto({})'.format(quality))
    binary = b64decode(data.split(',')[1])
    return binary

# Inicializa o processador e o modelo
model_name = "microsoft/xclip-base-patch32"
processor = XCLIPProcessor.from_pretrained(model_name)
model = XCLIPModel.from_pretrained(model_name)

def process_video(frames):
    inputs = processor(text=["playing sports", "open hand", "positive sign"], videos=list(frames), return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    probs = outputs.logits_per_video.softmax(dim=1)
    return probs

try:
    frames = []
    num_frames = 8  # Número de frames para formar um vídeo curto

    while len(frames) < num_frames:
        binary = take_photo()
        image = Image.open(io.BytesIO(binary)).convert("RGB")
        frames.append(np.array(image))

        # Exibe a imagem capturada
        display(image)

        if len(frames) == num_frames:
            probs = process_video(frames)
            print('Probabilidade: ', probs)
            frames = []  # Reiniciar a captura de frames

        input("Pressione Enter para capturar a próxima imagem...")

except Exception as err:
    print(str(err))
