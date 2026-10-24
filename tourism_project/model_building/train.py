import pandas as pd
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def train_model():
    # Load train and test splits created by prep.py
    X_train = pd.read_csv("Xtrain.csv")
    X_test = pd.read_csv("Xtest.csv")
    y_train = pd.read_csv("ytrain.csv").squeeze("columns")
    y_test = pd.read_csv("ytest.csv").squeeze("columns")
    
    # Handle categorical variables via one-hot encoding
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test = pd.get_dummies(X_test, drop_first=True)
    
    # Ensure test columns align with train columns
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    
    # Start MLflow experiment tracking
    mlflow.set_experiment("Wellness_Tourism_Experiment")
    
    with mlflow.start_run():
        n_estimators = 100
        max_depth = 5
        learning_rate = 0.1
        
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("learning_rate", learning_rate)
        
        # Define and train model
        model = XGBClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            learning_rate=learning_rate, 
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Evaluate performance
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        # Log metrics
        mlflow.log_metric("accuracy", acc)
        print(f"--- Model Training Complete ---")
        print(f"Validation Accuracy: {acc:.4f}")
        
        # Save the best model to the deployment folder
        os.makedirs("tourism_project/deployment", exist_ok=True)
        model_path = "tourism_project/deployment/model.pkl"
        joblib.dump(model, model_path)
        print(f"Model saved successfully to {model_path}")

if __name__ == "__main__":
    train_model()
