# -*- coding: utf-8 -*-
"""
Spyder Editor

Breast cancer
"""
"""
STEP #1: PROBLEM STATEMENT
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
 """
 
 # 2. Importing data
 # import libraries 
import pandas as pd # Import Pandas for data manipulation using dataframes
import numpy as np # Import Numpy for data statistical analysis 
import matplotlib.pyplot as plt # Import matplotlib for data visualisation
import seaborn as sns # Statistical data visualization
# %matplotlib inline
 
# Import Cancer data drom the Sklearn library
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
cancer

# to see what we have here
cancer.keys()
print(cancer['DESCR'])

print(cancer ['target'])
print(cancer ['target_names'])
print(cancer ['feature_names'])

cancer['data'].shape   # 569 rows and 30 columns

# Extracting the dataset only
df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))  # to put those columns together use append

df_cancer.head()
df_cancer.tail()

# Summarize numerical features
df_cancer.describe()

# 3. Visualizing the data
sns.pairplot(df_cancer, hue = 'target', vars = ['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'] )

sns.countplot(df_cancer['target'], label = "Count") 
# how many is 0 and how many is 1

sns.scatterplot(x = 'mean area', y = 'mean smoothness', hue = 'target', data = df_cancer)

# Let's check the correlation between the variables 
# Strong correlation between the mean radius and mean perimeter, mean area and mean primeter
plt.figure(figsize=(20,10)) 
sns.heatmap(df_cancer.corr(), annot=True) 



'''
 M O D E L   T R A I N I N G
'''
# NumPy for numerical computing
import numpy as np

# Pandas for DataFrames
import pandas as pd
pd.set_option('display.max_columns', 100)

# Matplotlib for visualization
from matplotlib import pyplot as plt
# display plots in the notebook
%matplotlib inline 

# Seaborn for easier visualization
import seaborn as sns

# Scikit-Learn for Modeling
import sklearn

# Pickle for saving model files
import pickle

# Import support vector classifier
from sklearn.svm import SVC 

# Import RandomForestClassifier and GradientBoostingClassifer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Function for splitting training and test set
from sklearn.model_selection import train_test_split # Scikit-Learn 0.18+

# Function for balancing the classes
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Function for creating model pipelines
from sklearn.pipeline import make_pipeline

# For data preprocessing
from sklearn import preprocessing

# Helper for cross-validation
from sklearn.model_selection import GridSearchCV

# Classification metrics (added later)
from sklearn.metrics import roc_curve, auc

# Import confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix

# 1. Fitting and tuning
# Create separate object for target variable
y = df_cancer['target']

# Create separate object for input features
X = df_cancer.drop(['target'], axis = 1)

# Split X and y into train and test sets
# Important: Also pass in the argument stratify=df.status in order to make sure the target variable's classes 
# are balanced in each subset of data! This is stratified random sampling.
sns.countplot(df_cancer['target'], label = "Count") 
# how many is 0 and how many is 1
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=1234,
                                                    stratify=df_cancer.target)  # in order to have balanced classes in each subset

# Data oversampling
ros = RandomOverSampler(random_state=0)
ros.fit(X_train, y_train)
X_resampled, y_resampled = ros.fit_sample(X_train, y_train)

# how many is 0 and how many is 1 in resampled data
sns.countplot(y_resampled, label = "Count") 

# Renaming the sets
X_train_before_sampling = X_train
y_train_before_sampling = y_train

X_train = X_resampled
y_train = y_resampled

# how many is 0 and how many is 1 in y_train now
sns.countplot(y_train, label = "Count") 

# Print number of observations in X_train, X_test, y_train, and y_test
print( len(X_train), len(X_test), len(y_train), len(y_test) )

# Decide whether to normalize or standardize the data

# without preprocessing
sns.distplot(df_cancer['mean area'])
plt.show()

sns.distplot(df_cancer['mean smoothness'])
plt.show()

sns.distplot(df_cancer['area error'])
plt.show()

# looks like they just need to be put on the same scale by normalization

# Build model pipelines =======================================================
# Pipeline dictionary
pipelines = {
    'svc' : make_pipeline(preprocessing.Normalizer(),                           
                         SVC(random_state=123, probability=True)),
    'rf' : make_pipeline(preprocessing.Normalizer(), RandomForestClassifier(random_state=123)),
    'gb' : make_pipeline(preprocessing.Normalizer(), GradientBoostingClassifier(random_state=123))
}

# List tuneable hyperparameters of SVC
pipelines['svc'].get_params()

# Support vector classifier hyperparameters
svc_hyperparameters = {
    'svc__C' : [0.1, 1, 10, 100],  # higher means higher penalty, prone to overfitting
    'svc__gamma': [1, 0.1, 0.01, 0.001], # large gamma means closer data points have higher weights, prone to overfitting
    'svc__kernel': ['rbf', 'linear']
}

# List tuneable hyperparameters of RF
pipelines['rf'].get_params()

# Random Forest hyperparameters
rf_hyperparameters = {
    'randomforestclassifier__n_estimators': [100, 200, 500, 1000],
    'randomforestclassifier__max_features': ['auto',  0.33]
}

# List tuneable hyperparameters of GB
pipelines['gb'].get_params()

# Boosted Tree hyperparameters
gb_hyperparameters = {
    'gradientboostingclassifier__n_estimators': [100, 200],
    'gradientboostingclassifier__learning_rate': [0.05, 0.1, 0.2],
    'gradientboostingclassifier__max_depth': [1, 3, 5]
}

# Create hyperparameters dictionary
hyperparameters = {
    'svc' : svc_hyperparameters,
    'rf' : rf_hyperparameters,
    'gb' : gb_hyperparameters
}

# Fit & tune models with cross-validation
# Create empty dictionary called fitted_models
fitted_models = {}

# Loop through model pipelines, tuning each one and saving it to fitted_models
for name, pipeline in pipelines.items():
    # Create cross-validation object from pipeline and hyperparameters
    model = GridSearchCV(pipeline, hyperparameters[name], cv=10, n_jobs=-1)
    
    # Fit model on X_train, y_train
    model.fit(X_train, y_train)
    
    # Store model in fitted_models[name] 
    fitted_models[name] = model
    
    # Print '{name} has been fitted'
    print(name, 'has been fitted.')

'''
W I N N E R   S E L E C T I O N
'''
# Evaluate metrics ===========================================================
# Display best_score_ for each fitted model
for name, model in fitted_models.items():
    print( name, model.best_score_ )
    
# Display best parameters for each fitted model
for name, model in fitted_models.items():
    print( name, model.best_params_ )
    
# building Confusion matrix
for name, model in fitted_models.items():
    # Coumpute the predictions
    pred = fitted_models[name].predict(X_test)
    
    # Display confusion matrix for y_test and pred
    cm = confusion_matrix(y_test, pred)

    sns.heatmap(cm, annot = True)
    print(classification_report(y_test,pred))
    
    # Area under ROC curve
    # Area under ROC curve is the most reliable metric for classification tasks.
    # Area under ROC curve is equivalent to the probability that a randomly chosen '0' 
    # observation ranks higher (has a higher predicted probability) than a randomly chosen '1' observation.
    # Basically, it's saying... if you grabbed two observations and exactly one of them was 
    # the positive class and one of them was the negative class, what's the likelihood that 
    # your model can distinguish the two?
        
    # building ROC curve
    # Calculate ROC curve from y_test and pred
    pred = model.predict_proba(X_test)
    pred = [p[1] for p in pred]

    fpr, tpr, thresholds = roc_curve(y_test, pred)
    
    # Store fpr, tpr, thresholds in DataFrame and display last 10
    pd.DataFrame({'FPR': fpr, 'TPR' : tpr, 'Thresholds' : thresholds}).tail(10)
     
    # Initialize figure
    fig = plt.figure(figsize=(8,8))
    plt.title('Receiver Operating Characteristic')
    
    # Plot ROC curve
    plt.plot(fpr, tpr, label='l1')
    plt.legend(loc='lower right')
    
    # Diagonal 45 degree line
    plt.plot([0,1],[0,1],'k--')
    
    # Axes limits and labels
    plt.xlim([-0.1,1.1])
    plt.ylim([-0.1,1.1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    # AUROC - area under the ROC curve
    # Remember, that AUROC is equivalent to the probability that a randomly chosen '0' observation 
    # ranks higher (has a higher predicted probability) than a randomly chosen '1' observation.
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, pred)
    
    # Calculate AUROC
    print( auc(fpr, tpr) )

# Picking the winner
for name, model in fitted_models.items():
    pred = model.predict_proba(X_test)
    pred = [p[1] for p in pred]
    
    fpr, tpr, thresholds = roc_curve(y_test, pred)
    print( name, auc(fpr, tpr) )
    
# Save winning model as final_model.pkl
with open('final_model.pkl', 'wb') as f:
    pickle.dump(fitted_models['rf'].best_estimator_, f)
    
# If we output that object directly, we can also see the winning values for our hyperparameters.
fitted_models['rf'].best_estimator_


'''
5. P R O J E C T   D E L I V E R Y
'''
# NumPy for numerical computing
import numpy as np

# Pandas for DataFrames
import pandas as pd
pd.set_option('display.max_columns', 100)
# pd.options.mode.chained_assignment = None  # default='warn'

# Pickle for reading model files
import pickle

# Scikit-Learn for Modeling
import sklearn
from sklearn.model_selection import train_test_split # Scikit-Learn 0.18+

# Area under ROC curve - if we don't need to plot the ROC curve
from sklearn.metrics import roc_auc_score

# Load final_model.pkl as model
with open('final_model.pkl', 'rb') as f:
    model = pickle.load(f)

# 1. Confirm your model was saved correctly ===================================
# Display model object
model

# Replicate model scores on the test set
# Load data
# Import Cancer data drom the Sklearn library
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
cancer

cancer['data'].shape   # 569 rows and 30 columns

# Extracting the dataset only
df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))  # to put those columns together use append

# Create separate object for target variable
y = df_cancer.target

# Create separate object for input features
X = df_cancer.drop('target', axis=1)

# Split X and y into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=1234,
                                                    stratify=df_cancer.target)
# Predict X_test
pred = model.predict_proba(X_test)

# Get just the prediction for the postive class (1)
pred = [p[1] for p in pred]

# Print AUROC
print( 'AUROC:', roc_auc_score(y_test, pred) )



























