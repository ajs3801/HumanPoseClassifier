import cv2
import os

VIDEO = "Video/stand3.avi"
VIDEO_NAME = "stand3"
DATA_PATH = "videoFlip"
cap = cv2.VideoCapture(VIDEO)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')

width = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
videopath = os.path.join(DATA_PATH, '{}_flip.avi'.format(VIDEO_NAME))
out = cv2.VideoWriter(videopath, fourcc, fps, (width, height))

while cap.isOpened():
  ret, frame = cap.read()

  if not ret:
    break

  frame = cv2.flip(frame,1)

  out.write(frame)
  print("SAVE..")

out.release()
cap.release()
cv2.destroyAllWindows()