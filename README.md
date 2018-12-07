# Breast-cancer-classification-project-
ML project that classifies tumors as malignat or benign

Predicting if the cancer diagnosis is benign or malignant based on several observations/features
30 features are used, examples:

  - radius (mean of distances from center to points on the perimeter)
  - texture (standard deviation of gray-scale values)
  - perimeter
  - area
  - smoothness (local variation in radius lengths)
  - compactness (perimeter^2 / area - 1.0)
  - concavity (severity of concave portions of the contour)
  - concave points (number of concave portions of the contour)
  - symmetry 
  - fractal dimension ("coastline approximation" - 1)
  
  Datasets are linearly separable using all 30 input features

Number of Instances: 569
Class Distribution: 212 Malignant, 357 Benign
Target class:
   - Malignant
   - Benign
   
Data: 
Dataset from Scikit learn modul (569 instances, 30 features)

Goal: 
Predict whether the cancer diagnosis is benign or malignant based on for ex. area, smoothness, perimeter, symmetry.

Challenge: 
Imbalanced classes (solved by oversampling)

Algorithms: 
SVM classifier, GB classifier, RF classifier

Measures: Confusion metrics, Area under the ROC

Project delivery: Python script executing locally hosted flask api, that takes in raw data, preprocess them, do the predictions and provide downloadable zipped .xlsx file that alongside input dataset provides predictions.

Instructions: Download raw_data.csv, all zip files (after unzipping make sure to have final_model.pkl in separate "model" folder created among the other downloaded files) and flask_predict_api.py.

Through your command line navigate to the folder you are storing these files. Make sure you have python path in your enviroment variables and run command python flask_predict_api.py

From your browser navigate to http://localhost:8000/apidocs. Click on predict_api and then try it out!. Insert raw_unseen_data and press execute. After some time scroll down and click on Download the zip.file, which contains the predictions.
