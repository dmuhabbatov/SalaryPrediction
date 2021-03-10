# Salary Prediction: Project Overview
* Examining a set of job postings with salaries and then predicting salaries for a new set of job postings.  
* Used 3 data sets with 1,000,000 rows: train_features.csv, train_salaries.csv (target) and test_features.csv
* Reviewed and cleaned the dataset by using Python 3 (Jupyter Notebook)
* Performed EDA to see the correlation between the input features and their effect on the target feature(salary).
* Engeneered features to convert categorial variables into numerical and made them ready for machine learning process. 
* Used Linear Regression, Random Forest and GradientBoosting algorithms for model building.
* Used MSE as accruacy metric to see which algorithm permorms better on the model. 
* Fit the model with the lowest MSE on the trained data and made prediction on the test data set
* Found out the most important feature effecting the salary (target) prediction for future job postings.

## Code and Resources Used 
**Python Version:** 3.7 
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn
* Google
* Website: Data Science Dream Job portfolio building 
* Book: Data Preparation for Machine Learning. Jason Brownlee

## Datasets and features

train_features.csv contains metadata for an individual job posting. The "jobId" column represents a unique identifier for the job posting.

train_salaries.csv contains salaries for job postings. Each row associates a "jobId" with a "salary".

test_features.csv contains metadata for an individual job posting. We have to predict the salaries for the job postings provided in the csv file test_features.

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
*	Merged the training data set having the target features (train_features.csv) and salaries (train_salaries.csv) in one file.
*	Training dataset does not have null values 
*	Checked for duplicates. There were no duplicates present in the data 
*	Checked for invalid data in the dataset. There were 5 rows in the dataset containing salaries less than zero

## EDA

I looked at the corralation between the variables. We dropped JobId as it is unique ID


![image](https://user-images.githubusercontent.com/75549127/110561082-4f95c600-8104-11eb-8b19-4c3de86b7161.png)

Based on the above heatmap we see that the jobType is most strongly correlated with salary, followed by degree, major, and yearsExperience. Among the features, we see that degree and major have a strong degree of correlation and jobType has a moderate degree of correlation with both degree and major.

## Model Building 

First, I transformed the categorical variables into dummy variables. I also split the data into train and tests sets with a test size of 20%.   

I tried three different models and evaluated them using Mean Absolute Error. I chose MAE because it is relatively easy to interpret and outliers aren’t particularly bad in for this type of model.   

I tried three different models:
*	**Multiple Linear Regression** – Baseline for the model
*	**Lasso Regression** – Because of the sparse data from the many categorical variables, I thought a normalized regression like lasso would be effective.
*	**Random Forest** – Again, with the sparsity associated with the data, I thought that this would be a good fit. 

## Model performance
The Random Forest model far outperformed the other approaches on the test and validation sets. 
*	**Random Forest** : MAE = 11.22
*	**Linear Regression**: MAE = 18.86
*	**Ridge Regression**: MAE = 19.67

## Production 
  Selected the model Gradient Boosting for salary prediction. Obtained predictions using the model and stored the file in csv format.


![image](https://user-images.githubusercontent.com/75549127/110562356-908eda00-8106-11eb-95a9-7252d9322715.png)



