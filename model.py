# Importing Necessary Libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

X,y = load_iris(return_X_y=True)

X = pd.DataFrame(X, columns=["Feature 1", "Feature 2", "Feature 3", "Feature 4"])
y = pd.Series(y,name="target")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model_pipeline = Pipeline([("Model", LogisticRegression())])
model_pipeline.fit(X_train, y_train)
y_preds = model_pipeline.predict(X_test)
score = accuracy_score(y_test, y_preds)
print(f"Model Accuracy {score}")

pickle.dump(model_pipeline, open("model_pipeline.pkl", "wb"))


