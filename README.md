# HumanPoseEstimation
This project is about a HPC(HumanPoseClassifier) implemented by Machine Learning and Deep Learning.

## PREVIEW
![DEMO_AdobeExpress](https://user-images.githubusercontent.com/43237393/185286388-212dc244-152f-4639-927f-b0dfa7b64010.gif)

## FUNCTIONALITY
1. count squat,lunge,pushup
2. angle Feedback

  > Squat: Leg angle
  
  > Pushup: Elbow anple
  
  > Lunge : Leg angle and waist angle

3. MSE Feedback : compare with the expert pose (User must stand with 45 degree)

## TREE
> ```src``` : Main project source code

```src/Posture_DeepLearning``` : the HPE implemented by DeepLearning(main_tf.py)
 
```src/Posture_MachineLearning``` : the HPE implemented by MachineLearning(main.py)

<br/>

> ```custom``` : You can collect data, extract features, and also train by the custom data set

```custom/Posture_CollectData``` : Collect custom data by cv2, extract features, and save the extracted features to .CSV

```custom/Posture_Data``` : The extracted feature I used to train

```custom/Posture_Train``` : You can train the data by ML and DL
