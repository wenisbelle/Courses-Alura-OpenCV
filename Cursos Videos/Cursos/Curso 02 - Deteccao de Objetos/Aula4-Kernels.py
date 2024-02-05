import numpy as np 
import cv2
import sys


VIDEO = "Curso 02 - Deteccao de Objetos/Dados/Ponte.mp4"

algorithm_types = ["KNN", "GMG", "CNT", "MOG", "MOG2"] 
algorithm_type = algorithm_types[1]

def Kernel(KERNEL_TYPE):
    if KERNEL_TYPE == "RECT":
        return cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    elif KERNEL_TYPE == "dilation":
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    elif KERNEL_TYPE == "CROSS":
        return cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    if KERNEL_TYPE == "opening":
        return np.ones((3,3), np.uint8)
    if KERNEL_TYPE == "closing":
        return np.ones((3,3), np.uint8)
    else:
        print("Kernel not found")
        sys.exit()

def Filter(img, filter):
    if filter == "opening":
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, Kernel('closing'), iterations=2)
    elif filter == "closing":
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, Kernel(filter), iterations=2)
    elif filter == "dilation":
        return cv2.dilate(img, Kernel(filter), iterations=2)
    elif filter == "combine":
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, Kernel("closing"), iterations=2)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, Kernel("opening"), iterations=2)
        dilation = cv2.dilate(opening, Kernel("dilation"), iterations=2)
        return dilation
    else:
        print("Filter not found")
        sys.exit()  



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
subtractor = Subtractor(algorithm_type)

def main():
    while (cap.isOpened()):
        ok, frame = cap.read()
        if not ok:
            print("Não foi possível ler o vídeo")
            break
        
        frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)

        mask = subtractor.apply(frame)
        mask_Filtered = Filter(mask, "dilation")
        cars_after_mask = cv2.bitwise_and(frame, frame, mask=mask_Filtered)

        #cv2.imshow("Mask", mask)
        cv2.imshow("Frame", frame)
        cv2.imshow("Mask Filtered", mask_Filtered)

        if cv2.waitKey(1) & 0xFF == ord('c'):
            break

main()


