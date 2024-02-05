import cv2
import mediapipe as mp # Importando a biblioteca mediapipe
import os
import subprocess
import threading


mp_maos = mp.solutions.hands
mp_desenho = mp.solutions.drawing_utils

maos = mp_maos.Hands() # Inicializando o objeto mãos

camera = cv2.VideoCapture(0) # Inicializando a captura de vídeo
resolucao_x = 1280
resolucao_y= 720

camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolucao_x)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolucao_y)

DEDO_INDICADOR = [True, False, False, False]
DEDO_INDICADOR_MEIO = [True, True, False, False]
DEDO_INDICADOR_MEIO_ANELAR = [True, True, True, False]
DEDO_ANELAR_MINDINHO = [False, False, True, True]
DEDOS_ABAIXADOS = [False, False, False, False]
DEDO_INDICADOR_MINIMO = [True, False, False, True]


text_editor_event = threading.Event()
browser_event = threading.Event()
calculator_event = threading.Event()

text_editor_process = None


def encontra_coordenadas_maos(img, lado_invertido = False):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convertendo a imagem para o padrão RGB

    resultado = maos.process(img_rgb) # Processando a imagem
    
    todas_maos = [] # Lista para armazenar as coordenadas de todas as mãos
    if resultado.multi_hand_landmarks: # Se houverem mãos na imagem    
        for lado_mao, marcacao_maos in zip(resultado.multi_handedness, resultado.multi_hand_landmarks):
            info_mao = {} 
            coordenadas = []
            for marcacao  in marcacao_maos.landmark:
                coord_x, coord_y, coord_z = int(marcacao.x * resolucao_x), int(marcacao.y * resolucao_y), int(marcacao.z * resolucao_x) # Obtendo as coordenadas da marcação real, ou seja, em pixels
                coordenadas.append([coord_x, coord_y, coord_z])

            info_mao["coordenadas"] = coordenadas
            
            if lado_invertido:
                if lado_mao.classification[0].label == "Left":
                    info_mao["lado"] = "Right"
                else:
                    info_mao["lado"] = "Left"
            else:
                info_mao["lado"] = lado_mao.classification[0].label

            todas_maos.append(info_mao) 
            mp_desenho.draw_landmarks(img, marcacao_maos, mp_maos.HAND_CONNECTIONS)

    return img, todas_maos


def dedos_levantados(mao):

    dedos = []
 
    for ponta_dedo in [8,12, 16, 20]:
        if mao["coordenadas"][ponta_dedo][1] <  mao["coordenadas"][ponta_dedo - 2][1]: # no opencv o eixo y é invertido
            dedos.append(True)
        else:
            dedos.append(False)

    return dedos


def polegar_levantado(mao):

    polegar = []
    ponta_polegar = 4

    if mao["coordenadas"][ponta_polegar][1] <  mao["coordenadas"][ponta_polegar - 1][1]: # no opencv o eixo y é invertido
        polegar.append(True)
    else:
        polegar.append(False)

    return polegar



def open_text_editor():
    global text_editor_process

    try:
        text_editor_process = subprocess.Popen(["gedit"])
        text_editor_process.wait()  # Wait for gedit to finish

    except Exception as e:
        print(f"Error opening text editor: {e}")
    finally:
        text_editor_event.clear()


def close_text_editor():
    global text_editor_process
    try:
        if text_editor_process and text_editor_process.poll() is None:
            # If gedit process is still running, terminate it
            text_editor_process.terminate()
            text_editor_process.wait()  # Wait for the termination to complete

    except Exception as e:
        print(f"Error closing gedit: {e}")



def open_browser():
    try:
        subprocess.run(["google-chrome"])
    except Exception as e:
        print(f"Error opening browser: {e}")
    finally:
        browser_event.clear()

def open_calculator():
    try:
        subprocess.run(["gnome-calculator"])
    except Exception as e:
        print(f"Error opening calculator: {e}")
    finally:
        calculator_event.clear()



while True:
    sucesso, img = camera.read() # Lendo a imagem da câmera
    img = cv2.flip(img, 1) # Invertendo a imagem horizontalmente

    img, todas_maos = encontra_coordenadas_maos(img)

    if len(todas_maos) == 1:
        info_dedos_mao1 = dedos_levantados(todas_maos[0])
        
        if todas_maos[0]["lado"] == "Right":

            if info_dedos_mao1 == DEDO_INDICADOR and not text_editor_event.is_set():
                threading.Thread(target=open_text_editor).start()
                text_editor_event.set()
    
               # se eu quiser com o local
               # file_path = "/path/to/your/text/file.txt"
               # subprocess.run(["gedit", file_path])
                
            if info_dedos_mao1 == DEDO_INDICADOR_MEIO and not browser_event.is_set():
                threading.Thread(target=open_browser).start()
                browser_event.set()
            
            if info_dedos_mao1 == DEDO_INDICADOR_MEIO_ANELAR and not calculator_event.is_set():
                threading.Thread(target=open_calculator).start()
                calculator_event.set()
    
            if info_dedos_mao1 == DEDOS_ABAIXADOS and text_editor_process.poll() is None:
                threading.Thread(target=close_text_editor).start()
    
            if info_dedos_mao1 == DEDO_INDICADOR_MINIMO:
                break


    cv2.imshow("imagem", img) # Exibindo a imagem
    tecla = cv2.waitKey(1) # Capturando a tecla pressionada

    if tecla == 27: # Se a tecla pressionada for ESC
        break 