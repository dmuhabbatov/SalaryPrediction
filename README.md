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
Google
Website: Data Science Dream Job portfolio building 
Book: Data Preparation for Machine Learning. Jason Brownlee

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

