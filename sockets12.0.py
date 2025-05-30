#server
import socket
import threading
import time
import numpy as np
import cv2
from transformers import XCLIPProcessor, XCLIPModel
import torch
from PIL import Image

# Inicializa o processador e o modelo
model_name = "microsoft/xclip-base-patch32"
processor = XCLIPProcessor.from_pretrained(model_name)
model = XCLIPModel.from_pretrained(model_name)

def pad_frames(frames, size=(224, 224), target_length=8):
    padded_frames = []
    for frame in frames:
        frame = (frame * 255).astype(np.uint8)  # Converte para uint8
        frame = Image.fromarray(frame).resize(size)
        padded_frames.append(frame)
    while len(padded_frames) < target_length:
        padded_frames.append(Image.new('RGB', size))  # Adiciona um frame vazio se necessário
    return padded_frames[:target_length]

def process_video(frames):
    print(f"Frames capturados: {len(frames)}\n")
    padded_frames = pad_frames(frames)
    padded_frames = [np.array(frame) for frame in padded_frames]

    inputs = processor(text=["raising arms", "crossed arms", "raising leg"], videos=[padded_frames], return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    probs = outputs.logits_per_video.softmax(dim=1)
    return probs

def capture_frames(interval=5, target_frames=16):
    cap = cv2.VideoCapture(0)
    frames = []
    start_time = time.time()
    
    while len(frames) < target_frames:
        ret, frame = cap.read()
        if not ret:
            break
        current_time = time.time()
        if current_time - start_time >= interval:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0)  # Converte para float e normaliza
            start_time = current_time
    
    cap.release()
    return frames

def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', 8080))
    server_socket.listen(1)
    print("Servidor iniciado e aguardando conexões...")

    client_socket, addr = server_socket.accept()
    print(f"Conexão estabelecida com {addr}")

    try:
        while True:
            # Captura frames continuamente a cada 5 segundos
            frames = capture_frames(interval=5)

            if len(frames) > 0:
                probs = process_video(frames)
                print(f"Probabilidades:\n{probs}\n")
                percentages = (probs * 100).numpy().tolist()[0]  # Converte para porcentagens e transforma em lista
                print(f"Porcentagens:\n{percentages}\n")
                message = ','.join(map(str, percentages))
                
                # Adiciona print para verificar o envio de dados
                print(f"Enviando dados: {message}")

                # Envia as porcentagens para o cliente
                client_socket.sendall(message.encode())
                time.sleep(0.5)  # Aguarda 0.5 segundos antes de enviar a próxima mensagem
    except Exception as e:
        print(f"Erro: {e}")
    finally:
        client_socket.close()
        server_socket.close()

# Inicia o servidor em uma thread separada
server_thread = threading.Thread(target=start_server)
server_thread.start()

#client
import socket

def start_client():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('localhost', 8080))

    try:
        while True:
            data = client_socket.recv(1024).decode()
            if data:
                print('Data pura recebida:', data)
                percentages = list(map(float, data.split(',')))
                boolean_array = [p > 70.0 for p in percentages]
                print(f"Array de booleans recebida: {boolean_array}")
            else:
                print("Nenhum dado recebido.")
    except Exception as e:
        print(f"Erro: {e}")
    finally:
        client_socket.close()

start_client()
