import numpy as np
import cv2
from time import sleep

VIDEO = "Curso 02 - Deteccao de Objetos/Dados/Rua.mp4"
delay = 10
# Vamos transformar em escala de cinza pra diminuir o custo computacional

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

#cv2.imshow("Median frame", medianFrame)
#cv2.waitKey(0) # 

cv2.imwrite("Curso 02 - Deteccao de Objetos/Dados/median_frame.jpg", medianFrame)

#----------Aula 2
# Dentro do opencv já existe uma função que faz a captura dos vídeos e a conversão pros frames
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)
#cv2.imshow("Gray Median frame", grayMedianFrame)
#cv2.waitKey(0)
cv2.imwrite("Curso 02 - Deteccao de Objetos/Dados/gray_median_frame.jpg", grayMedianFrame)

while True:
    tempo = float(1/delay)
    sleep(tempo)

    hasFrame, frame = cap.read()
    if not hasFrame:
        print("Fim do vídeo")
        break

    # Convert current frame to grayscale
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate absolute difference of current frame and the median frame, just to fint the objects that are moving
    dframe = cv2.absdiff(frameGray, grayMedianFrame)    
    # Let's apply a threshold just to make things more visible around the cars and the moving objects
    th, dframe = cv2.threshold(dframe, 30, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    #cv2.imshow("Frames em cinza ", frameGray)

    cv2.imshow("Frames em cinza ", dframe)


    # to see frame by frame
    #cv2.waitKey(0)

    if cv2.waitKey(1) & 0xFF == ord('c'):
        break

cap.release()



