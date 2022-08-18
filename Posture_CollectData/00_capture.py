import numpy as np
import math
from datetime import datetime
import cv2
import os
import time
import mediapipe as mp # Import mediapipe

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

time.sleep(2)
DATA_PATH = os.path.join('Video')

name = "lying"
def main(VIDEO_PATH):
  frame_count = 0

  # start detection
  cap = cv2.VideoCapture(VIDEO_PATH)
  fourcc = cv2.VideoWriter_fourcc(*'DIVX')

  width = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps = int(cap.get(cv2.CAP_PROP_FPS))

  videopath = os.path.join(DATA_PATH, '{}.avi'.format(datetime.today().strftime('%Y%m%d%H%M')+'_'+name))
  out = cv2.VideoWriter(videopath, fourcc, fps, (width, height))
  with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
      success, image = cap.read()
      if not success:
        continue

      frame_count += 1
      print(frame_count)
      out.write(image)

      cv2.imshow('Original', image)

      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

      if (frame_count == 500):
        break

    out.release()
    cap.release()

if __name__ == "__main__":
  VIDEO_PATH = 0
  main(VIDEO_PATH)