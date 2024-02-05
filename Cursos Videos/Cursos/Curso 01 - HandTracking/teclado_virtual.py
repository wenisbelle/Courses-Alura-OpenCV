import cv2
import mediapipe as mp # Importando a biblioteca mediapipe
import subprocess
import threading
from time import sleep
import concurrent.futures
from pynput.keyboard import Controller


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
DEDO_MINIMO = [False, False, False, True]

BRANCO = (255,255,255)
PRETO = (0,0,0)
AZUL = (255,0,0)
VERDE = (0,255,0)
VERMELHO = (0,0,255)
AZUL_CLARO = (255,255,0)

teclas = ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"], ["A", "S", "D", "F", "G", "H", "J", "K", "L", "Ç"], ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "-"]

OFFSET = 50
contador = 0
texto = ">"
teclado = Controller()


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

def imprime_botoes(img, posicao, letra, tamanho=50, cor_retangulo = BRANCO, cor_texto = PRETO):
    cv2.rectangle(img, posicao, (posicao[0] + tamanho, posicao[1] + tamanho), cor_retangulo, cv2.FILLED)
    cv2.rectangle(img, posicao, (posicao[0] + tamanho, posicao[1] + tamanho), AZUL, 1)
    cv2.putText(img, letra, (posicao[0] + 15 , posicao[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, cor_texto, 2)
    return img

def apaga_texto(texto):
    novo_texto = texto[:-1]
    sleep(0.15)
    return novo_texto



while True:
    sucesso, img = camera.read() # Lendo a imagem da câmera
    img = cv2.flip(img, 1) # Invertendo a imagem horizontalmente

    img, todas_maos = encontra_coordenadas_maos(img)

    
    if len(todas_maos) == 1:
        info_dedos_mao1 = dedos_levantados(todas_maos[0])

        
        if todas_maos[0]["lado"] == "Left":
            indicador_x, indicador_y, indicador_z = todas_maos[0]["coordenadas"][8]
            cv2.putText(img, f"Distancia indicador: {int(indicador_z)}", (800,50), cv2.FONT_HERSHEY_COMPLEX, 1, AZUL_CLARO, 2)
            
            for indice_linha, linha_teclado in enumerate(teclas):
                for indice, tecla in enumerate(linha_teclado):
                    if sum(info_dedos_mao1) <= 1:
                        tecla = tecla.lower() 
                    img = imprime_botoes(img, (OFFSET+indice*80, OFFSET+indice_linha*80), tecla)
                    if OFFSET+indice*80 < indicador_x < OFFSET+50+indice*80 and OFFSET+indice_linha*80<indicador_y<OFFSET+50+indice_linha*80:
                        img = imprime_botoes(img, (OFFSET+indice*80, OFFSET+indice_linha*80), tecla, cor_retangulo=VERDE)
                        if indicador_z < -85:
                            contador = 1
                            escreve = tecla
                            img = imprime_botoes(img, (OFFSET+indice*80, OFFSET+indice_linha*80), tecla, cor_retangulo=AZUL_CLARO)

            if contador:
                contador += 1
                if contador ==3:
                    texto+=escreve
                    contador = 0
                    teclado.press(escreve)

            if info_dedos_mao1 == DEDO_MINIMO and len(texto)>1:
                # Create a thread for the 'process_text' function
                print("apagando")
                # Create a ThreadPoolExecutor
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    # Submit the function to the executor
                    future = executor.submit(apaga_texto, texto)
                    # Wait for the thread to finish and retrieve the result
                    texto = future.result()


            cv2.rectangle(img, (OFFSET, 450), (830,500), BRANCO, cv2.FILLED)
            cv2.rectangle(img, (OFFSET, 450), (830,500), AZUL, 1)
            cv2.putText(img, texto[-40:], (OFFSET, 475), cv2.FONT_HERSHEY_COMPLEX, 1, PRETO, 2)
            cv2.circle(img, (indicador_x,indicador_y), 7, AZUL, cv2.FILLED)


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
    
            if info_dedos_mao1 == DEDOS_ABAIXADOS:
                threading.Thread(target=close_text_editor).start()
    
            if info_dedos_mao1 == DEDO_INDICADOR_MINIMO:
                break


    cv2.imshow("imagem", img) # Exibindo a imagem
    tecla = cv2.waitKey(1) # Capturando a tecla pressionada

    if tecla == 27: # Se a tecla pressionada for ESC
        break 

with open("texto.txt", "w") as f:
    f.write(texto)