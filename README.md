# Salary Prediction: Project Overview

  Problem Definition: Examining a set of job postings with salaries and then predicting salaries for a new set of job postings.  
  
* Used 3 data sets with 1,000,000 rows: train_features.csv, train_salaries.csv (target) and test_features.csv
* Reviewed and cleaned the dataset by removing not important rows and features.
* Performed EDA to see the correlation between the input features and their relationship with the target feature(salary). 
* Engeneered features to convert categorial variables into numerical and made them ready for machine learning process. 
* Used Linear Regression, Random Forest and GradientBoosting algorithms for model building. Pipeline and transformation was used too. 
* Used MSE as accruacy metric to see which algorithm performs better on the model. 
* Fit/trained the model with the lowest MSE on the trained data and made prediction on the test dataset.
* Found the most important feature impacting the salary (target feature).

## Code and Resources Used 
**Python Version:** Python 3 (Jupyter Notebook)
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn
* Google
* Website: Data Science Dream Job portfolio building 
* Book: Data Preparation for Machine Learning. Jason Brownlee
* other Data Science projects

## Datasets and features

1 train_features has metadata for an individual job posting. The "jobId" column represents a unique identifier for the job posting.
2 train_salaries has salaries for job postings. Each row associates a "jobId" with a "salary".
3 test_features contains metadata for an individual job posting. We have to predict the salaries for the job postings provided in the csv file test_features.

*	JobID
*	CompanyId
*	JobType 
*	Industry
*	Degree
*	Major
*	Years_of_Experience
*	Miles_from_Metropolis
*	Salary

 ## Data Loading, Cleaning and Manipulation
 
* Imported the required libraries. 
*	Loaded the data in the Data Frames. 
*	Merged the training and target datasets in one file.
*	Did not find missing values on the datasets.
*	Some outliers were found and removed from the datasets. 
*	Checked for duplicates. There were no duplicates present in the data 


## EDA

I looked at the corralation between the variables.


![image](https://user-images.githubusercontent.com/75549127/110561082-4f95c600-8104-11eb-8b19-4c3de86b7161.png)

Based on the above heatmap we see that the jobType is most strongly correlated with salary, followed by degree, major, and yearsExperience. Among the features, we see that degree and major have a strong degree of correlation and jobType has a moderate degree of correlation with both degree and major.

## Model Building 

Removed the JobId feature as it is has unique Id for individuals and then transformed the categorical variables into dummy variables. Performed one_hot_encoding on both train and test sets.    

Tried three different models and evaluated them using Mean Squared Error. 

I tried three different models:
*	**Linear Regression**
*	**Gradient Boosting**
*	**Random Forest**

## Model performance
The Random Forest model far outperformed the other approaches on the test and validation sets. 
*	**Random Forest** : MSE = 366.65
*	**Linear Regression**: MSE = 384.44
*	**Gradient Boosting**: MSE = 398.37

## Production 
  Selected the model Gradient Boosting for salary prediction. Obtained predictions using the model and stored the file in csv format.


![image](https://user-images.githubusercontent.com/75549127/110562356-908eda00-8106-11eb-95a9-7252d9322715.png)

We found that "years_of_experieince" has the highest impact on salary.

