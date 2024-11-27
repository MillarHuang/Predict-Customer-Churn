"""
A library of functions to find customers who are likely to churn, which contains 
all necessary function to build a model for predicting the churning rate

Author: Zhicong Huang

Datetime: 2024.11.26
"""


# import libraries
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, RocCurveDisplay
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    try:
        df = pd.read_csv(pth, index_col=0)
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        return df
    except BaseException:
        print("The file wasn't found or the file doesn't appear to have rows and columns")
        return None


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    images_dir = './images/eda/'
    df_num = df.select_dtypes(include=['number'])
    df_cat = df.select_dtypes(include=['object'])
    # Visualization on quantitative variables
    for column in df_num:
        # Numerical column: Plot histogram
        plt.figure(figsize=(8, 6))
        sns.histplot(df_num[column], stat='density', kde=True)
        plt.title(f'Distribution of {column}')
        # Save the plot
        plt.savefig(f"{images_dir}{column}_plot.png", dpi=300)
        plt.close()  # Close the figure to save memory

    # Produce heatmap on quantitative variables
    plt.figure(figsize=(20, 10))
    sns.heatmap(df_num.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.title('Heatmap of quantitative variables')
    # Save the plot
    plt.savefig(
        f"{images_dir}heatmap_quantitative_variables_plot.png",
        dpi=300)
    plt.close()  # Close the figure to save memory

    # Visualization on categorical variables
    for column in df_cat:
        # Categorical column: Plot countplot
        plt.figure(figsize=(8, 6))
        sns.countplot(x=column, data=df_cat)
        plt.title(f'Count of {column}')
        # Save the plot
        plt.savefig(f"{images_dir}{column}_plot.png", dpi=300)
        plt.close()  # Close the figure to save memory

    # Bivariate plot: Box plot of Total_Trams_Amt by Attrition_Flag
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df['Attrition_Flag'], y=df['Total_Trans_Amt'])
    plt.title("Box Plot: Total_Trams_Amt by Attrition_Flag")
    plt.xlabel("Attrition_Flag")
    plt.ylabel("Total_Trams_Amt")
    # Save the plot
    plt.savefig(f"{images_dir}bivariate_plot.png", dpi=300)
    plt.close()  # Close the figure to save memory


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for 
            naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    # Add the new column with propotion of churn for each category for this
    # categorical variable
    for cat_variable in category_lst:
        # New column name
        name = f"{cat_variable}_{response}"
        df[name] = df.groupby(cat_variable)[
            response].transform(lambda x: x.mean())

    return df


def perform_feature_engineering(df, category_lst, response):
    '''
    input:
              df: pandas dataframe
              category_lst: list of columns that contain categorical features, which 
              will be turned to a new column with propotion of churn for each category
              response: string of response name [optional argument that could be used 
              for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    # Use one-hot encoding to transform the response variable "Attrition_Flag"
    # as numerical
    df[response] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    # Feature engineering
    y = df[response]
    X = pd.DataFrame()
    # Add the new column with propotion of churn for each category for this
    # categorical variable
    df = encoder_helper(df, category_lst, response)
    # Pick the independent variables
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']
    X[keep_cols] = df[keep_cols]
    # Split dataset into training set and testing set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    result_path = './images/results/'
    # Generate classification report for logistic regression
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    # Save the figure
    plt.savefig(
        f'{result_path}classification_report_lr.png',
        bbox_inches='tight',
        dpi=300)
    plt.show()
    plt.close()

    # Generate classification report for random forest
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    # Save the figure
    plt.savefig(
        f'{result_path}classification_report_rf.png',
        bbox_inches='tight',
        dpi=300)
    plt.show()
    plt.close()


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))
    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])
    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    # Save the figure
    plt.savefig(
        f'{output_pth}feature_importance.png',
        bbox_inches='tight',
        dpi=300)
    plt.show()
    plt.close()


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''

    # Model training
    # Random Forest model training
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)
    # Logistic regression model training
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    lrc.fit(X_train, y_train)
    # Random Forest model prediction
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)
    # Logistic regression model prediction
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # Save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # Produces AUC score and plot and save it
    fig, ax = plt.subplots()
    lrc_plot = RocCurveDisplay.from_estimator(lrc, X_test, y_test)
    plt.close(fig)

    auc_path = './images/results/'
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = RocCurveDisplay.from_estimator(
        cv_rfc.best_estimator_, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    # Save the figure
    plt.savefig(
        f'{auc_path}Models_result_AUC.png',
        bbox_inches='tight',
        dpi=300)
    plt.show()
    plt.close()

    # Produces classification report and store model results as image
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)
    # Creates and stores the feature importances plot
    output_pth = './images/results/'
    feature_importance_plot(cv_rfc.best_estimator_, X_test, output_pth)
