import numpy as np 
import cv2

VIDEO = "video/PisaTimelapse.mp4"

cap = cv2.VideoCapture(VIDEO)

# Check if the video was opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file '{VIDEO}'")
    # Optionally, release the VideoCapture object
    cap.release()
else:
    print(f"Video file '{VIDEO}' opened successfully")

framesIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=72) # 72 frames, igual a 24 fps

frames = []
for fid in framesIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    hasFrame, frame = cap.read()
    frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    frames.append(frame)

# Calculate the median along the time axis
medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)

cv2.imshow("Frame aleatório", frames[0])
cv2.imshow("Median frame", medianFrame)
cv2.waitKey(0) # 27 = ESC
cv2.imwrite("video/median_frame.jpg", medianFrame)