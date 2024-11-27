"""
A file of testing functions to test and log message on all necessary function
to build a model for predicting the churning rate

Author: Zhicong Huang

Datetime: 2024.11.26
"""

import os
import logging
import pytest
import churn_library as cls


logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = cls.import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
        pytest.df = df
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found ERROR")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns ERROR")
        raise err


def test_eda():
    '''
    test perform eda function
    '''
    df = pytest.df
    try:
        cls.perform_eda(df)
        logging.info("Testing perform_eda: Running function SUCCESS")
    except BaseException:
        logging.info("Testing perform_eda: Running function ERROR")
    try:
        # Check if every variable plot is generated
        for variable in df.columns:
            path = f'./images/eda/{variable}_plot.png'
            assert os.path.isfile(path)
        # Check if heatmap is generated
        assert os.path.isfile(
            './images/eda/heatmap_quantitative_variables_plot.png')
        logging.info(
            "Testing perform_eda: All EDA plots are completely produced SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_eda: Some visualization plots were not found ERROR")
        raise err


def test_encoder_helper():
    '''
    test encoder helper
    '''
    df = pytest.df
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    category_lst = ['Gender',
                    'Education_Level',
                    'Marital_Status',
                    'Income_Category',
                    'Card_Category']
    try:
        new_df = cls.encoder_helper(df, category_lst, 'Churn')
        logging.info("Testing encoder_helper: Running function SUCCESS")
    except BaseException:
        logging.error("Testing encoder_helper: Running function ERROR")
    try:
        new_columns = [col + "_Churn" for col in category_lst]
        for new_column in new_columns:
            assert new_column in new_df.columns
        logging.info(
            "Testing encoder_helper: Produced expected columns SUCCESS")
    except AssertionError:
        logging.error("""Testing encoder_helper: new columns according to the
                      propotion of churn for each category for every categorical
                      variable were not fully created in the dataframe ERROR""")


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''
    category_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']
    response = 'Churn'
    df = pytest.df
    try:
        pass
    except BaseException:
        pass
    try:
        X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
            df, category_lst, response)
        logging.info(
            "Testing perform_feature_engineering: Running function SUCCESS")
        pytest.X_train = X_train
        pytest.X_test = X_test
        pytest.y_train = y_train
        pytest.y_test = y_test
    except Exception as err:
        logging.error(
            "Testing perform_feature_engineering: Running function ERROR")
        raise err
    try:
        assert 'Churn' not in X_train
        assert 'Churn' not in X_test
    except AssertionError:
        logging.error(
            "Testing perform_feature_engineering: Response variable is in independent variable set")
    try:
        assert X_train.ndim == 2
        assert X_test.ndim == 2
        assert y_train.ndim == 1
        assert y_test.ndim == 1
        logging.info("""Testing perform_feature_engineering: The shape of training/test
                     set is as expected SUCCESS""")
    except AssertionError:
        logging.error("""Testing perform_feature_engineering: The shape of training/test
                      set is not as expected ERROR""")


def test_train_models():
    '''
    test train_models
    '''
    X_train = pytest.X_train
    X_test = pytest.X_test
    y_train = pytest.y_train
    y_test = pytest.y_test
    try:
        cls.train_models(X_train, X_test, y_train, y_test)
        logging.info("Testing train_models: Running function SUCCESS")
    except BaseException:
        logging.info("Testing train_models: Running function ERROR")
    try:
        assert os.path.isfile('./models/logistic_model.pkl')
        assert os.path.isfile('./models/rfc_model.pkl')
        logging.info(
            "Testing train_models: Model is completely stored SUCCESS")
    except AssertionError:
        logging.error(
            "Testing train_models: Model is not completely stored ERROR")
    try:
        assert os.path.isfile('./images/results/classification_report_lr.png')
        assert os.path.isfile('./images/results/classification_report_rf.png')
        logging.info(
            "Testing train_models: Classification report is completed SUCCESS")
    except AssertionError:
        logging.error(
            "Testing train_models: Classification report is not completed ERROR")
    try:
        assert os.path.isfile('./images/results/Models_result_AUC.png')
    except AssertionError:
        logging.error("Testing train_models: AUC plot is not completed ERROR")
    try:
        assert os.path.isfile('./images/results/feature_importance.png')
        logging.info(
            "Testing train_models: Feature importance plot is completed SUCCESS")
    except AssertionError:
        logging.error(
            "Testing train_models: Feature importance plot is not completed ERROR")


if __name__ == "__main__":
    test_import()
    test_eda()
    test_encoder_helper()
    test_perform_feature_engineering()
    test_train_models()
