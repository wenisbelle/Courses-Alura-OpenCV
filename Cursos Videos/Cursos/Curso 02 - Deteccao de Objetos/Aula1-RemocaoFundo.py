import numpy as np
import cv2

VIDEO = "Curso 02 - Deteccao de Objetos/Dados/Rua.mp4"
#VIDEO = "Curso 02 - Deteccao de Objetos/Dados/Arco.mp4"
#VIDEO = "Curso 02 - Deteccao de Objetos/Dados/Estradas.mp4"
#VIDEO = "Curso 02 - Deteccao de Objetos/Dados/Peixes.mp4"

# Dentro do opencv já existe uma função que faz a captura dos vídeos e a conversão pros frames
cap = cv2.VideoCapture(VIDEO)
hasFrame, frame = cap.read()

if not hasFrame:
    print("Não foi possível ler o vídeo")
    exit()

#vamos pegar frames aleatórios do vídeo
#estamos pegando 72 frames aleatórios a partir de uma distribuição uniforme, todos tem a mesma chance de ser escolhidos
framesIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=72)

frames = []
for fid in framesIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    hasFrame, frame = cap.read()
    frames.append(frame)

# Calculate the median along the time axis
medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)

cv2.imshow("Median frame", medianFrame)
cv2.waitKey(0) # 

cv2.imwrite("Curso 02 - Deteccao de Objetos/Dados/median_frame.jpg", medianFrame)
