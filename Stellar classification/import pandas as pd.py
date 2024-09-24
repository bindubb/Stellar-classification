import pandas as pd
import numpy as np
from joblib import dump

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import precision_recall_fscore_support

# Load the dataset
stellar = pd.read_csv("star_classification.csv")

# Drop unnecessary columns if they exist
columns_to_drop = ["obj_ID", "run_ID", "rerun_ID", "cam_col", "field_ID", "spec_obj_ID","MJD","fiber_ID","plate"]
stellar.rename(columns = {"u" : "Ultravoilet", "g" : "Green", "r":"Red", "i" : "Near Infrared", "z": "Infrared" }, inplace = True)
stellar.drop(columns=[col for col in columns_to_drop if col in stellar.columns], axis=1, inplace=True)

# Encode the target variable
encoder = LabelEncoder()
stellar['class'] = encoder.fit_transform(stellar['class'])

# Define features (X) and target (y)
X = stellar.drop('class', axis=1)
y = stellar['class']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70, random_state=150)

# Initialize the model
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.01)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate precision, recall, and F1 score using the 'macro' average method
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')

# Print the evaluation metrics
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1_score}')

dump(model,"gb_model.joblib",compress=1)
