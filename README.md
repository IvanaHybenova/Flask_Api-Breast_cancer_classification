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

Project delivery: Pipeline that first normalize the data and then apply classifier (% cross validated accuracy)
