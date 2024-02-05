import cv2
import mediapipe as mp
import numpy as np
import time
import threading

import pygame
print(cv2.cuda.getCudaEnabledDeviceCount())


"""
# Initialize the mixer module
pygame.mixer.init()

# Load the sound file
pygame.mixer.music.load('beep-warning-6387.mp3')

mp_draw = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

p_olho_esq = [385, 380, 387, 373, 362, 263]
p_olho_dir = [160, 144, 158, 153, 33, 133]
p_boca = [82, 87, 13, 14, 312, 317, 78, 308]

p_olhos = p_olho_esq + p_olho_dir

ear_limiar = 0.4
boca_limiar = 0.3
dormindo = 0
t_inicial = time.time()
stop_sound = False
contagem_piscadas = 0
c_tempo = 0
contagem_temporaria = 0
contagem_lista = []

t_piscadas = time.time()


def calculo_ear(face, p_olho_dir, p_olho_esq):
    
    try:    
        face = np.array([[coord.x, coord.y] for coord in face])

        face_esq = face[p_olho_esq]
        face_dir = face[p_olho_dir]

        a = np.linalg.norm(face_esq[0] - face_esq[1])
        b = np.linalg.norm(face_esq[2] - face_esq[3])
        c = np.linalg.norm(face_esq[4] - face_esq[5])

        ear_esq = (a + b) / (2.0 * c)

        a = np.linalg.norm(face_dir[0] - face_dir[1])
        b = np.linalg.norm(face_dir[2] - face_dir[3])
        c = np.linalg.norm(face_dir[4] - face_dir[5])

        ear_dir = (a + b) / (2.0 * c)

    except:
        ear_esq = 0
        ear_dir = 0
    #retornar a média dos valores    
    return (ear_dir + ear_esq) / 2.0

def calculo_mar(face, p_boca):
    
    try:    
        face = np.array([[coord.x, coord.y] for coord in face])

        face_boca = face[p_boca]


        a = np.linalg.norm(face_boca[0] - face_boca[1])
        b = np.linalg.norm(face_boca[2] - face_boca[3])
        c = np.linalg.norm(face_boca[4] - face_boca[5])
        d = np.linalg.norm(face_boca[6] - face_boca[7])

        mar_boca = (a + b + c) / (2.0 * d)


    except:
        mar_boca = 0   
    
    return mar_boca


with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_landmarks=False) as facemesh:
    
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        sucesso, frame = cap.read()
        if not sucesso:
            print("Não foi possível ler o frame")
            continue    
        
        comprimento, largura, _ = frame.shape

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        saida_facemesh = facemesh.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        try:
            for face_lendmark in saida_facemesh.multi_face_landmarks:
                mp_draw.draw_landmarks(frame, face_lendmark, mp_face_mesh.FACEMESH_CONTOURS, 
                                       landmark_drawing_spec=mp_draw.DrawingSpec(color=(255,102,102), thickness=1, circle_radius=1),
                                       connection_drawing_spec=mp_draw.DrawingSpec(color=(102,204,0), thickness=1))
                
                
                face = face_lendmark.landmark
                for id_coord, coord_xyz in enumerate(face):
                    if id_coord in p_olhos:
                       coord_cv = mp_draw._normalized_to_pixel_coordinates(coord_xyz.x,coord_xyz.y, largura, comprimento)
                       cv2.circle(frame, coord_cv, 2, (255,0,0), -1)
                    
                    if id_coord in p_boca:
                       coord_cv = mp_draw._normalized_to_pixel_coordinates(coord_xyz.x,coord_xyz.y, largura, comprimento)
                       cv2.circle(frame, coord_cv, 2, (255,0,0), -1)

                ear = calculo_ear(face, p_olho_dir, p_olho_esq)
                mar_boca = calculo_mar(face, p_boca)
                cv2.rectangle(frame, (0,1), (290, 140), (58,58,55), -1)
                cv2.putText(frame, f"EAR: {round(ear, 2)} e Mar:{round(mar_boca,2)}", (1,24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
          

                if ear < ear_limiar and mar_boca < boca_limiar:
                    if dormindo == 0:
                        t_inicial = time.time()
                    else:
                        t_inicial = t_inicial
                    
                    if dormindo == 0:
                        contagem_piscadas += 1
                    else:
                        contagem_piscadas = contagem_piscadas
                    
                    dormindo = 1
                
                if (dormindo == 1 and ear >= ear_limiar) or (ear<= ear_limiar and mar_boca >= boca_limiar):
                    dormindo = 0

                t_final = time.time()
                tempo_decorrido = t_final - t_piscadas


                if tempo_decorrido >= (c_tempo+1):
                    c_tempo = tempo_decorrido
                    piscadas_ps = contagem_piscadas - contagem_temporaria
                    contagem_temporaria = contagem_piscadas
                    contagem_lista.append(piscadas_ps)

                    if len(contagem_lista) < 60:
                        contagem_lista = contagem_lista
                    else:
                        contagem_lista[-60:]
                    
                    if tempo_decorrido < 60:
                        piscadas_pm = 15
                    else:
                        piscadas_pm = sum(contagem_lista)


                tempo = (t_final - t_inicial) if dormindo == 1 else 0
                cv2.putText(frame, f"Piscadas: {contagem_piscadas}", (1,120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (109,233,219), 2)
                cv2.putText(frame, f"Piscadas pm: {piscadas_pm}", (200,120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (109,233,219), 2)
                cv2.putText(frame, f"Tempo dormindo: {round(tempo, 3)}", (1, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)  
                
                if tempo > 1.5 or piscadas_pm < 10: # esse tempo de 1.5 veio de um artigo que buscava encontrar a relação entre o tempo de olhos fechados e a sonolência

                    cv2.rectangle(frame, (30,400), (610, 452), (109,233,219), -1)
                    cv2.putText(frame, "ACORDA", (80, 435), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                    # Start a new thread to play the sound
                    pygame.mixer.music.play()
                else:
                    pygame.mixer.music.stop()  

        except:
             pass
        
        cv2.imshow('frame', frame)
            
        if cv2.waitKey(1) & 0xFF == 27:
                break

cap.release()
cv2.destroyAllWindows()"""









