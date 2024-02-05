#Nesta etapa, utilizaremos as máscaras preestabelecidas pelo OpenCV. São elas:
#
#KNN: muito comum nos processamentos de machine learning, trata-se de um método de cluster onde os grupos são formados com base na sua semelhança. Essa máscara utiliza as distâncias dos pixels para definir o plano de fundo e o objeto de primeiro plano;
#GMG: utiliza o Teorema de Bayes e o aplica nos 5 primeiros segundos do vídeo. Através dessa teoria, atualiza os números de pixels e atribui pesos maiores aos novos pixels, o que permite melhor identificar os possíveis objetos de primeiro plano;
#CNT: é um algoritmo count que verifica os valores dos pixels nos frames anteriores a fim de tentar identificar se esses pixels correspondem ao plano de fundo ou à objetos em movimento;
#MOG: mixture-of-gaussians ou mistura de fundo adaptativa, utiliza a curva gaussiana comum na qual cada pixel é caracterizado por sua intensidade no espaço de cores RGB;
#MOG2: versão melhorada do MOG.

import cv2
import sys

VIDEO = "Curso 02 - Deteccao de Objetos/Dados/Ponte.mp4"

algorithm_types = ["KNN", "GMG", "CNT", "MOG", "MOG2"] 
algorithm_type = algorithm_types[4]

# KNN = 4.91 segundos
# GMG = 6.66 segundos
# CNT = 4.02 segundos
# MOG = 11.46 segundos
# MOG2 = 9.05 segundos

def Subtractor(algorithm_type):
    if algorithm_type == "KNN":
        return cv2.createBackgroundSubtractorKNN()
    elif algorithm_type == "GMG":
        return cv2.bgsegm.createBackgroundSubtractorGMG()
    elif algorithm_type == "CNT":
        return cv2.bgsegm.createBackgroundSubtractorCNT()
    elif algorithm_type == "MOG":
        return cv2.bgsegm.createBackgroundSubtractorMOG()
    elif algorithm_type == "MOG2":
        return cv2.createBackgroundSubtractorMOG2()
    else:
        print("Alforithm not found")
        sys.exit()

cap = cv2.VideoCapture(VIDEO)
e1 = cv2.getTickCount()

background_subtractor = []

for i, alg in enumerate(algorithm_types):
    print(i,alg)
    background_subtractor.append(Subtractor(alg))


def main():
    frame_number = -1
    while (cap.isOpened()):
        has_frame, frame = cap.read()
        if not has_frame:
            break
        
        frame = cv2.resize(frame, (0, 0), fx=0.35, fy=0.35)
        frame_number += 1

        knn = background_subtractor[0].apply(frame)
        gmg = background_subtractor[1].apply(frame)
        cnt = background_subtractor[2].apply(frame)
        mog = background_subtractor[3].apply(frame)
        mog2 = background_subtractor[4].apply(frame)

        cv2.imshow("Original", frame)
        cv2.imshow("knn", knn)
        cv2.imshow("gmg", gmg)
        cv2.imshow("cnt", cnt)
        cv2.imshow("mog", mog)
        cv2.imshow("mog2", mog2)

        if cv2.waitKey(1) & 0xFF == ord('c'):
            break

    e2 = cv2.getTickCount()

    print("Tempo de processamento: {} segundos".format((e2 - e1)/cv2.getTickFrequency()))  

main()


