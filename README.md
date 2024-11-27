# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project provides a workflow of building models on predicting customers who are likely to run. 

## Files and data description
* churn_library.py: This file contains a library of all necessary function to build a model for predicting the churning rate.
* churn_script_logging_and_tests.py: This file contains testing functions to test each input function in **churn_library.py**, and would log testing result messages to a log file **churn_library.log** in logs folder.
* churn_notebook.ipynb: The jupyternotebook file to showcase the workflow of churn rate predictive model building and performance checking.
* conftest.py: The config of testing
* requirements_py3.10: The dependency of this project
* data/bank_data.csv: The dataset to use for building churn rate predictive model
* images/eda: The folder that contains the EDA visualization results: univariate plots of all variables (including numerical and categorical), bivariate plot of Total_Trams_Amt by Attrition_Flag, heatmap of all numerical variables
* image/results: The folder that contains the predictive model results of logistic regression model and random forest model
* logs/churn_library.log: The log file that contains the loggging information while doing testing on all input function
* models: The folder that contains the well-trained predicitve models, including logistic regression model and random forest model



## Running Files
How do you run your files? What should happen when you run your files?

* Install the necessary packages for this project under python 3.10:
    * (Run on terminal) python -m pip install -r requirements_py3.10.txt

* Run the blocks on **churn_notebook.ipynb** to show the workflow of building predictive model of churn rate and get the model performances

* Do unit test on all input function **in churn_library.py**, the testing result will be logged in logs/churn_library.log:
    * (Run on terminal) python -m churn_script_logging_and_tests
    * (or use pytest) pytest --log-file=./logs/churn_library.log --log-file-level=INFO churn_script_logging_and_tests.py



