"""
The config of testing

Author: Zhicong Huang

Datetime: 2024.11.26
"""
import pytest


def pytest_configure():
    """
    The pytest config: here to define globally shared variables for all testing function
    """
    pytest.df = None
    pytest.X_train = None
    pytest.X_test = None
    pytest.y_train = None
    pytest.y_test = None
