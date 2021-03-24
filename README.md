# Salary Prediction: Project Contents 
  
* The Problem Definition
* Data Quality Check
* Exploratory Data Analysis
* Model Building 
* Model Performance 
* Prediction and Production
* Conclusion


## Problem Definition:

Examining a set of job postings with salaries and then predicting salaries for a new set of job postings. Salary prediction is really important for companies to facilitate the budget projection and for future staff planning. Thus, analysing and finding the main factors impacting the salary predictions, will help companies to solve this problem.   


## Datasets and features. 

1. The train_features has metadata for an individual job posting. The "jobId" column represents a unique identifier for the job posting.

2. The train_salaries has salaries for job postings. Each row associates a "jobId" with a "salary".

3. The test_features contains metadata for an individual job posting. We have to predict the salaries for the job postings provided in the csv file test_features.

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

For checking the correlation between each input and the output variable, we used regplot for examining numerical variables and boxplot for categorical variables. The notebook has details on that but here I included the heatmap for the purpose of simplicity to quickly glance the relation between all variables. 


![image](https://user-images.githubusercontent.com/75549127/110561082-4f95c600-8104-11eb-8b19-4c3de86b7161.png)

Based on the above heatmap we see that the jobType is most strongly correlated with salary, followed by degree, major, and yearsExperience. Among the features, we see that degree and major have a strong degree of correlation and jobType has a moderate degree of correlation with both degree and major.

## Model Building 

Removed the JobId feature as it is has unique Id for individuals and then transformed the categorical variables into dummy variables. Performed one_hot_encoding on both train and test sets.    

Tried three different models, applied 5-fold cross validation and measured MSE as evaluation metric.

*	**Linear Regression**
*	**Gradient Boosting**
*	**Random Forest**

## Model performance
The Random Forest model far outperformed the other approaches. 
*	**Random Forest** : MSE = 366.65
*	**Linear Regression**: MSE = 384.44
*	**Gradient Boosting**: MSE = 398.37

## Prediction and Production 

  Selected the Random Forest model for salary prediction. Obtained predictions using the model and stored the file in csv format. The prediction file is also saved here on my repository: https://github.com/dmuhabbatov/SalaryPrediction/blob/main/predictions.csv 
 
 Finally analysed the feature importance method using bar plot:  
    

![image](https://user-images.githubusercontent.com/75549127/110562356-908eda00-8106-11eb-95a9-7252d9322715.png)



## Conclusion
  
  After performing the EDA and predicting the model with the random forest algorithm and looking at the feature importance analysis, we can conclude that "year_of_experience" will have the highest impact on the target (salary) followed by "jobtype". In short, this salary prediction result along with the feature importance chart can help the companies to make better decision.     

 ## Code and Resources Used 
**Python Version:** Python 3 (Jupyter Notebook)
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn
* Google
* Website: Data Science Dream Job portfolio building 
* Book: Data Preparation for Machine Learning. Jason Brownlee
* other Data Science projects
