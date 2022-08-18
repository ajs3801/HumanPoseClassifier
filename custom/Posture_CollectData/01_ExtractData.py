import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
import numpy as np
import csv

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

class_name = "stand"

cap = cv2.VideoCapture("videoFlip/stand3_flip.avi")

# Initiate holistic model
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if (ret == False):
          break

        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False        
        
        # Make Detections
        results = pose.process(image)
        # print(results.face_landmarks)
        
        # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
        
        # Recolor image back to BGR for rendering
        image.flags.writeable = True   
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(
          image,
          results.pose_landmarks,
          mp_pose.POSE_CONNECTIONS,
          landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        # Export coordinates
        try:
          if results.pose_world_landmarks:
            row = list(np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_world_landmarks.landmark]).flatten())
            # Append class name 
            row.insert(0, class_name)

            # Export to CSV
            with open('Dataset.csv', mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(row) 

            print("SAVE [lenght is {}]".format(len(row)))
        except:
            pass
      
        cv2.imshow('Raw Webcam Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
