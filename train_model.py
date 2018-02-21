from settings import *
from pipeline import *
import pandas as pd 
import numpy as np

def Xy_split(df):
    y = df.Survived.values
    X = df.drop('Survived', 1)
    return X, y

def build_pipeline():
    steps_boolean_types = [('selector', TypeSelector('bool'))]
    steps_number_types = [('selector', TypeSelector(np.number))
                            , ('scaler', StandardScaler())]
    steps_category_types = [('selector', TypeSelector('O'))
                            , ('labeler', DataFrameLabelEncoder())
                            , ('encoder', DataFrameOneHotEncoder())
                            ]
    remove_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked']
    steps=[ ("cols_removing", ColumnsRemoving(remove_cols))
            , ("imputer", DataFrameImputer())
            , ('features', FeatureUnion(n_jobs=1, transformer_list=
                                      [('steps_boolean_types', Pipeline(steps_boolean_types))
                                      ,('steps_number_types', Pipeline(steps_number_types))
                                      ,('steps_category_types', Pipeline(steps_category_types))
                                       ]))
            ]
    return Pipeline(steps)

if __name__ == "__main__":
    train_data_path = RAW_DATA_DIR + "/" + train_file
    train = pd.read_csv(train_data_path)
    X, y = Xy_split(train)
   
    # Construct a pipeline
    pipeline = build_pipeline()
    
    # Fit and transform data using pipeline dump it in models directory
    X_transformed = pipeline.fit_transform(X)
    pipeline_path = MODELS_DIR + "/" + pipeline_file
    joblib.dump(pipeline, pipeline_path, compress = 1)
    
    # Train a Logistic regression and dump it in models directory
    lr = LogisticRegression()
    lr.fit(X_transformed, y)
    model_path = MODELS_DIR + "/" + model_file
    joblib.dump(lr, model_path, compress = 1)
