import subprocess
import sys
import re
import os
import importlib

# Include Dash, Streamlit, and Lazy Predict in the required libraries list
required_libraries = [
    "numpy", "pandas", "matplotlib.pyplot", "matplotlib.ticker",
    "seaborn", "plotly.express", "plotly.offline", "scikit-learn",
    "xgboost", "imblearn", "dash", "streamlit", "lazypredict"  # Add Dash, Streamlit, and Lazy Predict
]

# Check for installed libraries and install missing ones
missing_libraries = [lib for lib in required_libraries if not importlib.util.find_spec(lib)]

if missing_libraries:
    print("The following libraries are not installed and will be installed:")
    for lib in missing_libraries:
        print(lib)

    #subprocess.run(["pip", "install", *missing_libraries], capture_output=True)

# Import the required libraries, including Dash, Streamlit, and Lazy Predict
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import plotly.express as px
import plotly.offline as pyo
import dash  # Import Dash
#import streamlit as st  # Import Streamlit
#import streamlit_pandas as sp
from pydantic_settings import BaseSettings
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
from pydantic_settings import BaseSettings
#import LazyPredict  # Import Lazy Predict

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve, average_precision_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)