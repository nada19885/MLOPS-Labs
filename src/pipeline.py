from numpy.core import numeric
import pandas as pd
from pandas.core.arrays import categorical
from sklearn import pipeline
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

def load_and_preprocess_data(data_path, test_size, random_state):
    data = pd.read_csv(data_path)
    data = data[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]].dropna()
    X = data.drop("Survived", axis=1)
    y = data["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test
    

def saved_processed_data(X_train, X_test , y_train , y_test , processed_data_path):
    os.makedirs(processed_data_path , exist_ok=True)
    X_train.to_parquet(f"{processed_data_path}/train_X.parquet")
    X_test.to_parquet(f"{processed_data_path}/test_X.parquet")
    y_train.to_parquet(f"{processed_data_path}/train_y.parquet")
    y_test.to_parquet(f"{processed_data_path}/test_y.parquet")

def create_pipeline(model_type , model_params):
    numeric_features = ["Age" , "SibSp" , "Parch" , "Fare"]
    categorical_features = ["Pclass" , "Sex" , "Embarked"]
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler" , StandardScaler()),
        ]
    )   

    categorical_transformer = Pipeline(
        steps =[
            ("imputer" , SimpleImputer(strategy="constant" , fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    ) 


    preprocessor = ColumnTransformer(
        transformers=[
            ("num" , numeric_transformer , numeric_features),
            ("cat" , categorical_transformer , categorical_features),
        ]
    )

    if model_type == "logistic":
        model = LogisticRegression(**model_params)
    elif model_type == "random_forest":
        model = RandomForestClassifier(**model_params)
    else:
        raise ValueError("Model Must be of type 'logestic' or 'random forest' ") 

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model ),
        ]
    )

    return pipeline 

def train_and_evaluate(X_train , X_test , y_train , y_test , model_type , model_params , model_path):
    pipeline = create_pipeline(model_type , model_params)
    pipeline.fit(X_train,y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test , y_pred)
    os.makedirs(model_path , exist_ok=True)
    with open(f"{model_path}/{model_type}.pkl" , "wb") as f:
        pickle.dump(pipeline,f)
    return accuracy      