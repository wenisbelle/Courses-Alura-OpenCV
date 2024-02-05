import torch
import cv2
import numpy as np

print(torch.cuda.is_available())

print("OpenCV Version:", cv2.__version__)
print("CUDA Support:", cv2.cuda.getCudaEnabledDeviceCount() > 0)

modelo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

cap = cv2.VideoCapture(0)

while cap.isOpened:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    frame = cv2.resize(frame, (640, 480))
    results = modelo(frame)
    cv2.imshow('YOLO', np.squeeze(results.render()))

    if cv2.waitKey(1) == ord('q'):
        break

cap.release
cv2.destroyAllWindows()
