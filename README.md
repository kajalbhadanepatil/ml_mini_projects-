# ml_mini_projects-
This Repository contains mini Machine Learning projects(Supervised) completed by me for academic, self learning, and hobby purposes, presented in the form of iPython Notebooks.

# Introduction 
## Supervised Machine Learning 
Supervised learning is when the model is getting trained on a labelled dataset. 
A labelled dataset is one that has both input and output parameters.
Types of Supervised Learning:  

Classification: It is a Supervised Learning task where output is having defined labels(discrete value). The goal here is to predict discrete values belonging to a particular class and evaluate them on the basis of accuracy. 
It can be either binary or multi-class classification. In binary classification, the model predicts either 0 or 1; yes or no but in the case of multi-class classification, the model predicts more than one class. 

Regression: It is a Supervised Learning task where output is having continuous value. The goal here is to predict a value as much closer to the actual output value as our model can and then evaluation is done by calculating the error value. The smaller the error the greater the accuracy of our regression model.

Example of Supervised Learning Algorithms:  

* Linear Regression
* Logistic Regression
* Decision Trees
* Random Forest
* K Nearest Neighbor
* Support Vector Machine (SVM)
* Gaussian Naive Bayes

# Contents 
* [Logistic Regression](https://github.com/kajalbhadanepatil/ml_mini_projects-/blob/main/project_logistic%20regression_h1n1%20vaccine.ipynb) - To predict whether people got H1N1 vaccines using information they shared about their backgrounds, opinions, and health behaviors.
It is a binary classification problem
* [Decission Tree](https://github.com/kajalbhadanepatil/ml_mini_projects-/blob/main/project_decision_tree_heart%20attack.ipynb) - Analyze the heart disease dataset to explore the machine learning algorithms and build decision tree model to predict the disease.
It is a binary classification problem
* [Random forest and Boosting](https://github.com/kajalbhadanepatil/ml_mini_projects-/blob/main/project_ensemble_regresson_taxifare%20price.ipynb) - Given pickup and dropoff locations, the pickup timestamp, and the passenger count, the objective is to predict the fare of the taxi ride using ensemble techniques. It is a regression problem 
* [Random forest and Boosting](https://github.com/kajalbhadanepatil/ml_mini_projects-/blob/main/project_ensemble_classification_breast_cancer.ipynb) -  Given the details of cell nuclei taken from breast mass, predict whether or not a patient has breast cancer using the Ensembling Techniques. It is binary classification problem
* [KNN](https://github.com/kajalbhadanepatil/ml_mini_projects-/blob/main/project_knn_mobilePricePrediction.ipynb) - Predict a price range, indicating how high the price is, using K-Nearest Neighbors algorithm. It is multiclass classification problem 
* [SVM](https://github.com/kajalbhadanepatil/ml_mini_projects-/blob/main/project_svm_termdeposit.ipynb) - Predict if a customer subscribes to a term deposits or not, when contacted by a marketing agent using SVM. It is binary classification problem

# Technical Aspect

* The Models in Python as scripting langauge are :
  * Logistic Regression (Statsmodels library,sklearn Library)
  * DecisionTree (sklearn library) 
  * Random Forest (sklearn library) 
  * AdaBoost (sklearn Library)
  * GradientBoosting (sklearn Library)
  * XGB (xgboost)
  * KNN (sklearn Library)
  * SVM (sklearn Library)
* For Feature Selection the following technique used :
  * Random Forest Classifier (Sklearn library)
* Data Distribution balancing done by :
  * SMOTE
* For Data standerdizing the following methods used (from sklearn library):
  * Standerd Scaler
  * Minmax Scaler
* For evaluating results ( from sklearn library)
  * confusion_matrix (for classification problem) 
  * classification_report (for classification problem)
  * mean_squared_error (for regression problem)
  * r2_score (for regression problem)







