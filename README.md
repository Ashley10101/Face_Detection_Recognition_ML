# Face Detection and Recognition
Contains self-made datasets, as well as face recognition systems implemented using traditional machine learning algorithms. __Highlight:__ all algorithms are implemented manually without any integrated package calls.  
Machine Learning Algorithms: Adaboost  
Language: Python
## Overall Process
(1) Dataset Creation  
(2) Face Detection  
(3) Feature Extraction  
(4) Face Recognition

## Implementation
run main.py
sample result  
![Image text](https://raw.github.com/Ashley10101/repositpry/master/Face_Detection_Recognition_ML/sample_result.png)

## Notification
The code is not concerned with correctness, but mainly focus on manual implementation of traditional machine learning algorithms

## 1. Dataset Creation
`data path`: `../image/`  
`label path`: `../face_detection/image_mark.txt`  
(1) Collect images containing human faces.  
(2) Label the face region of each image.  
The face region of each image is stored as a `.txt` file in the format of `image/bihan_01.png x y h w`, where _x y_ represents the coordinates of the upper left corner of the face region and _h w_ represents the width of the face region.  
_note: The provided dataset was collected from the TV series ‘Serenade of Peaceful Joy’_ 

## 2. Face Detection 
`code path`: `../face_detection`  
(1) Creat dataset  
`code path`: `../face_detection/gen_image_mark.py`  
`output`: `adaboost_image`  
This dataset contains face regions and non-face regions and is used to train the face detector. The size of all images is kept consistent.  
face region: 1_xxx.png  
non-face region: 00_xxx.png; 01_xxx.png  
  
(2) Generate Haar Feature for each image  
`code path`: `../face_detection/gen_haar_feature.py`  
`output`: `HAAR Feature.xlsx` 

(3) Get a classifier for each feature  
`code path`: `../face_detection/Adaboost.py`  
`output`: `Strong Classifier.xlsx`

## 3. Face Recognition 
`code path`: `face_recognition`  
(1) Creat dataset  
`code path`: `../face_recognition/split_face.py`   
`output`: `face`   
(2) Feature Extraction  
`code path`: `../face_recognition/PCA.py`
