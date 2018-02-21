import os 
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
import pandas as pd 
import numpy as np

class ColumnsRemoving(BaseEstimator, TransformerMixin):
    def __init__(self, remove_cols=[]):
        self.remove_cols = remove_cols

    def transform(self, X, **transform_params):
        trans = X.drop(self.remove_cols, axis=1) 
        return trans

    def fit(self, X, y=None, **fit_params):
        return self

class DataFrameImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def transform(self, X, **transform_params):
        return X.fillna(self.fill)
        
    def fit(self, X, y=None, **fit_params):
        fill = []
        for col in X.columns:
            fill.append(X[col].value_counts().index[0])
        self.fill = pd.Series(fill, index = X.columns)    
        return self        
    
class TypeSelector(BaseEstimator, TransformerMixin):
    def __init__(self, dtype):
        self.dtype = dtype
    
    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X, **transform_params):
        assert isinstance(X, pd.DataFrame)
        return X.select_dtypes(include = [self.dtype])
        
class DataFrameLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        labelencoder = LabelEncoder()
        trans = labelencoder.fit_transform(X)
        return trans.reshape(-1,1)
        
class DataFrameOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, **transform_params):
        #assert isinstance(X, pd.DataFrame)
        onehotencoder = OneHotEncoder()
        return onehotencoder.fit_transform(X).toarray()
