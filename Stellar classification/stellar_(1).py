# -*- coding: utf-8 -*-
"""stellar (1).ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1WOVmKeUp4bivbqvBNgOsK5rQ43eXmndm
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
from IPython.display import display
from tabulate import tabulate
import pickle

stellar_data = pd.read_csv("//content//star_classification.csv")

stellar_data.head()

stellar_data.tail()

stellar_data.rename(columns = {"u" : "Ultravoilet", "g" : "Green", "r":"Red", "i" : "Near Infrared", "z": "Infrared" }, inplace = True)

stellar_data.head()

stellar_data.info()

stellar_data.describe()

print('Number of Null values present:\n' ,stellar_data.isnull().sum())

print("The shape of the dataset: ", stellar_data.shape)

print("Number of duplicate values present :\n", stellar_data.duplicated().sum())

print("The count of total values of the target variable :\n", stellar_data['class'].value_counts())

features = ['alpha', 'delta', 'Ultravoilet', 'Green', 'Red', 'Near Infrared', 'Infrared', 'redshift']

stellar_data[features].describe().T

target = stellar_data["class"]

class_breakdown = stellar_data["class"].value_counts()
fig, (bar) = plt.subplots( figsize=(12, 5))
classes = ["Galaxies", "Stars", "Quasars"]


bar.set_title("Breakdown of Stellar Objects")
bar.set_xlabel("Class")
bar.set_ylabel("Number of Objects")
sns.barplot(x=classes, y=class_breakdown.values,
            palette= "ocean", ax=bar)
plt.show()

print(stellar_data["class"].value_counts())

class_breakdown = stellar_data["class"].value_counts()
fig, ( pie) = plt.subplots( figsize=(12, 5))
color_map_name = {'GALAXY': "#1f3b4d", 'STAR': "#ac7e04", 'QSO': "#117779"}
classes = ["Galaxies", "Stars", "Quasars"]


pie.set_title("Proportions of Stellar Objects")
pie.pie(class_breakdown.values, labels=classes, autopct='%1.1f%%',
        colors=color_map_name.values(), startangle=90)

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
# Define features and target
features = ['alpha', 'delta', 'Ultravoilet', 'Green', 'Red', 'Near Infrared', 'Infrared', 'redshift']
target = stellar_data["class"]

# Create box plots for each feature with target
plt.figure(figsize=(15, 10))
for i, feature in enumerate(features):
  plt.subplot(3, 3, i+1)
  sns.boxplot(x='class', y=feature, data=stellar_data, palette={'GALAXY': "#1f3b4d", 'STAR': "#ac7e04", 'QSO': "#117779"})
  plt.title(feature)
plt.tight_layout()
plt.show()

display(stellar_data[stellar_data['Green'] == min(stellar_data['Green'])])

display(stellar_data[stellar_data['Ultravoilet'] == min(stellar_data['Ultravoilet'])])

stellar_data = stellar_data.drop(index=[79543])
print("The shape of the dataset: ", (stellar_data.shape))

import matplotlib.pyplot as plt
# Define features and target
features = ['alpha', 'delta', 'Ultravoilet', 'Green', 'Red', 'Near Infrared', 'Infrared', 'redshift']
target = stellar_data["class"]

# Create box plots for each feature with target
plt.figure(figsize=(20, 13))
for i, feature in enumerate(features):
  plt.subplot(3, 3, i+1)
  sns.boxplot(x='class', y=feature, data=stellar_data, palette={'GALAXY': "#1f3b4d", 'STAR': "#ac7e04", 'QSO': "#117779"})
  plt.title(feature)
plt.tight_layout()
plt.show()

sns.boxplot(stellar_data['redshift'])

sns.distplot(stellar_data["redshift"])

percentile25 = stellar_data['redshift'].quantile(0.25)
print("Percentile25:",percentile25)
percentile75 = stellar_data['redshift'].quantile(0.75)
print("Percentile75:", percentile75)
iqr = percentile75 - percentile25
print("IQR:", iqr)

upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr
print("Upper limit",upper_limit)
print("Lower limit",lower_limit)

stellar_data[stellar_data['redshift'] > upper_limit]

new_df_cap = stellar_data.copy()

new_df_cap['redshift'] = np.where(
    new_df_cap['redshift'] > upper_limit,
    upper_limit,
    np.where(
        new_df_cap['redshift'] < lower_limit,
        lower_limit,
        new_df_cap['redshift']
    )
)

new_df_cap.shape



features = ['alpha', 'delta', 'Ultravoilet', 'Green', 'Red', 'Near Infrared', 'Infrared', 'redshift']
target = stellar_data["class"]

# Create box plots for each feature with target
plt.figure(figsize=(20, 13))
for i, feature in enumerate(features):
  plt.subplot(3, 3, i+1)
  sns.kdeplot(data=new_df_cap, x=feature, hue=target, shade=True, palette = "ocean")
  plt.title(feature)
plt.tight_layout()
plt.show()

P = sns.pairplot(data=new_df_cap[['alpha', 'delta', 'redshift', 'class']],
                 hue='class', palette = "ocean")
P.fig.suptitle(t='Pair Plot - Stellar Data', y=1.02, fontsize=10)
plt.show()


plt.figure(figsize=(5, 3))
sns.heatmap(data=new_df_cap[['alpha', 'delta', 'redshift']].corr(),
            annot=True, fmt='.2f', linewidths=0.1)
plt.show()

from astropy import units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt

# Extract alpha, delta, and class
alpha = stellar_data['alpha']
delta = stellar_data['delta']
classes = stellar_data['class']

# Convert to SkyCoord objects
coords = SkyCoord(alpha, delta, unit='deg')

# Define new colors for each class
color_map = {'GALAXY': 'purple', 'STAR': 'orange', 'QSO': 'cyan'}

# Create a Mollweide projection
plt.figure(figsize=(10, 6))
plt.subplot(projection="mollweide")

# Plot the coordinates with class-based colors
for c in color_map:
  plt.scatter(coords[classes == c].ra.wrap_at(180 * u.deg).radian,
              coords[classes == c].dec.radian,
              marker='.', color=color_map[c], label=c)

# Set labels and title
plt.xlabel('Right Ascension')
plt.ylabel('Declination')
plt.title('Distribution of Sky Coordinates (Mollweide Projection)')

# Add a legend
plt.legend()

# Show the plot
plt.grid(True)
plt.show()

from astropy import units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt

# Extract alpha, delta, and class
alpha = stellar_data['alpha']
delta = stellar_data['delta']
classes = stellar_data['class']

# Convert to SkyCoord objects
coords = SkyCoord(alpha, delta, unit='deg')

# Creating a Mollweide projection for stars only
plt.figure(figsize=(10, 6))
plt.subplot(projection="mollweide")
plt.scatter(coords[classes == 'STAR'].ra.wrap_at(180 * u.deg).radian,
            coords[classes == 'STAR'].dec.radian,
            marker='.', color='orange', label='STAR')
plt.xlabel('Right Ascension')
plt.ylabel('Declination')
plt.title(f'Distribution of STAR (Mollweide Projection)')
plt.grid(True)
plt.show()

# Creating a Mollweide projection for galaxies only
plt.figure(figsize=(10, 6))
plt.subplot(projection="mollweide")
plt.scatter(coords[classes == 'GALAXY'].ra.wrap_at(180 * u.deg).radian,
            coords[classes == 'GALAXY'].dec.radian,
            marker='.', color='purple', label='GALAXY')
plt.xlabel('Right Ascension')
plt.ylabel('Declination')
plt.title(f'Distribution of GALAXY (Mollweide Projection)')
plt.grid(True)
plt.show()

# Creating a Mollweide projection for quasars only
plt.figure(figsize=(10, 6))
plt.subplot(projection="mollweide")
plt.scatter(coords[classes == 'QSO'].ra.wrap_at(180 * u.deg).radian,
            coords[classes == 'QSO'].dec.radian,
            marker='.', color='cyan', label='QSO')
plt.xlabel('Right Ascension')
plt.ylabel('Declination')
plt.title(f'Distribution of QSO (Mollweide Projection)')
plt.grid(True)
plt.show()

P = sns.pairplot(data=new_df_cap[['Ultravoilet', 'Green', 'Red', 'Near Infrared', 'Infrared','redshift','class']], hue='class')
P.fig.suptitle(t='Pair Plot - Stellar Data', y=1.02)
plt.show()

plt.figure(figsize=(5, 3))
sns.heatmap(data=new_df_cap[['Ultravoilet', 'Green', 'Red', 'Near Infrared', 'Infrared','redshift']].corr(),
            annot=True, fmt='.2f', linewidths=0.1)
plt.show()

import matplotlib.pyplot as plt

# Filter data for galaxies only
galaxy_data = stellar_data[stellar_data['class'] == 'STAR']

# Select features and corresponding colors
features = {
    'Ultravoilet': 'purple',
    'Green': 'green',
    'Red': 'red',
    'Near Infrared': 'brown',
    'Infrared': 'black'
}

# Create histograms with specified colors
plt.figure(figsize=(12, 8))
for i, (feature, color) in enumerate(features.items()):
  plt.subplot(2, 3, i+1)
  plt.hist(galaxy_data[feature], bins=20, color=color)
  plt.title(feature)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

# Filter data for galaxies only
galaxy_data = stellar_data[stellar_data['class'] == 'QSO']

# Select features and corresponding colors
features = {
    'Ultravoilet': 'purple',
    'Green': 'green',
    'Red': 'red',
    'Near Infrared': 'brown',
    'Infrared': 'black'
}

# Create histograms with specified colors
plt.figure(figsize=(12, 8))
for i, (feature, color) in enumerate(features.items()):
  plt.subplot(2, 3, i+1)
  plt.hist(galaxy_data[feature], bins=20, color=color)
  plt.title(feature)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

# Filter data for galaxies only
galaxy_data = stellar_data[stellar_data['class'] == 'GALAXY']

# Select features and corresponding colors
features = {
    'Ultravoilet': 'purple',
    'Green': 'green',
    'Red': 'red',
    'Near Infrared': 'brown',
    'Infrared': 'black'
}

# Create histograms with specified colors
plt.figure(figsize=(12, 8))
for i, (feature, color) in enumerate(features.items()):
  plt.subplot(2, 3, i+1)
  plt.hist(galaxy_data[feature], bins=20, color=color)
  plt.title(feature)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Create a combined histogram for redshift distribution
plt.figure(figsize=(10, 6))
sns.histplot(stellar_data[stellar_data['class'] == 'GALAXY']['redshift'], bins=30, kde=True, color='blue', label='Galaxies')
sns.histplot(stellar_data[stellar_data['class'] == 'QSO']['redshift'], bins=30, kde=True, color='yellow', label='Quasars')
plt.title('Redshift Distribution of Galaxies and Quasars')
plt.xlabel('Redshift')
plt.ylabel('Frequency')
plt.legend()
plt.show()

features = ['alpha', 'delta', 'Ultravoilet', 'Green', 'Red', 'Near Infrared', 'Infrared','redshift']

X = stellar_data[features]
y = stellar_data["class"].values

print(X.columns)
print("The shape of X: {}".format(X.shape))
print("The shape of y: {}".format(y.shape))

from sklearn.model_selection import train_test_split
(X_train,X_test,y_train,y_test) = train_test_split(X, y,stratify=y, test_size=0.20, random_state=0)
(X_train,X_cv,y_train,y_cv) = train_test_split(X_train, y_train, stratify=y_train, test_size=0.20, random_state=0)

print("X_train dataset: {}".format(X_train.shape))
print(" X_cv dataset: {}".format(X_cv.shape))
print(" X_test dataset: {}".format(X_test.shape))

print(" y_train dataset: {}".format(y_train.shape))
print(" y_cv dataset: {}".format(y_cv.shape))
print(" y_test dataset: {}".format(y_test.shape))

from sklearn.preprocessing import MinMaxScaler
scaling = MinMaxScaler()
X_train = scaling.fit_transform(X=X_train)
X_cv = scaling.transform(X=X_cv)
X_test = scaling.transform(X=X_test)

def dataframe(ar, cols):
    df = pd.DataFrame(data=ar, columns=cols)
    return df

train_df_fea = dataframe(ar=X_train, cols=features)
cv_df_fea = dataframe(ar=X_cv, cols=features)
test_df_fea = dataframe(ar=X_test, cols=features)

def featurize(df):
    df['Green-Red'] = df['Green'] - df['Red']
    df['Near Infrared-Infrared'] = df['Near Infrared'] - df['Infrared']
    df['Ultravoilet-Red'] = df['Ultravoilet'] - df['Red']
    df['Near Infrared-Red'] = df['Near Infrared'] - df['Red']
    df['Infrared-Red'] = df['Infrared'] - df['Red']
    return df

train_df_fea = featurize(df=train_df_fea)
cv_df_fea = featurize(df=cv_df_fea)
test_df_fea = featurize(df=test_df_fea)

fea_cols = ['Ultravoilet', 'Green', 'Red', 'redshift', 'Green-Red', 'Near Infrared-Infrared', 'Ultravoilet-Red', 'Near Infrared-Red', 'Infrared-Red']

print(fea_cols)

X_train_fea = train_df_fea[fea_cols].values
X_cv_fea = cv_df_fea[fea_cols].values
X_test_fea = test_df_fea[fea_cols].values

import os
def export_data(data, target, f_name):
    """
    This function exports the data.

    Parameters
    ----------
    `data`: dataframe
    `filename`: the filename that data will be exported to
    """
    if os.path.isdir('./data'):
        pass
    else:
        os.mkdir(path='./data')

    data['class'] = target
    data.to_csv(path_or_buf=os.path.join('./data', f_name), index=None)
    print("The data is exported to '{}'.".format(f_name))

export_data(data=train_df_fea[fea_cols], target=y_train,
            f_name='train_fea.csv')

export_data(data=cv_df_fea[fea_cols], target=y_cv,
            f_name='cv_fea.csv')

export_data(data=test_df_fea[fea_cols], target=y_test,
            f_name='test_fea.csv')

export_data(data=test_df_fea[features], target=y_test,
            f_name='test_data.csv')

train_fea_df = pd.read_csv(filepath_or_buffer='./data/train_fea.csv')
cv_fea_df = pd.read_csv(filepath_or_buffer='./data/cv_fea.csv')
test_fea_df = pd.read_csv(filepath_or_buffer='./data/test_fea.csv')

fea_col = list(train_fea_df.columns)
target= fea_col.pop()
l = cv_fea_df['class'].unique()
print("The features class is:\n",fea_col)
print("The target class is:",target)
print("The target labels are:",l)

X_train = train_fea_df[fea_cols].values
y_train = train_fea_df[target].values

X_cv = cv_fea_df[fea_cols].values
y_cv = cv_fea_df[target].values

X_test = test_fea_df[fea_cols].values
y_test = test_fea_df[target].values

print(X_train.shape, y_train.shape)
print(X_cv.shape, y_cv.shape)
print(X_test.shape, y_test.shape)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import classification_report

# Create an SVM classifier object
clf = SVC(probability=True)

# Train the model using the training sets
clf.fit(X_train, y_train)

# Predict the response for cross-validation and test datasets
y_pred_cv_proba = clf.predict_proba(X_cv)
y_pred_test_proba = clf.predict_proba(X_test)
y_pred_cv = clf.predict(X_cv)
y_pred_test = clf.predict(X_test)

# Calculate accuracy scores
accuracy_cv = accuracy_score(y_cv, y_pred_cv)
accuracy_test = accuracy_score(y_test, y_pred_test)
train_loss = log_loss(y_train, clf.predict_proba(X_train))
cv_loss = log_loss(y_cv, y_pred_cv_proba)
test_loss = log_loss(y_test, y_pred_test_proba)


print(f"Cross-validation accuracy: {accuracy_cv}")
print(f"Test accuracy: {accuracy_test}")
print(f"Train Loss: {train_loss}")
print(f"CV Loss: {cv_loss}")
print(f"Test Loss: {test_loss}")

print(classification_report(y_test, y_pred_test))

print(classification_report(y_cv, y_pred_cv))

from sklearn.metrics import confusion_matrix

# Generate confusion matrix for cross-validation
cm_cv = confusion_matrix(y_cv, y_pred_cv)

# Generate confusion matrix for test set
cm_test = confusion_matrix(y_test, y_pred_test)

# Print confusion matrices
print("Confusion Matrix (Cross-validation):\n", cm_cv)
print("Confusion Matrix (Test):\n", cm_test)

from sklearn.metrics import confusion_matrix

# Create subplots for test and CV confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Plot test confusion matrix
sns.heatmap(cm_test, annot=True, fmt='d', cmap='viridis', xticklabels=l, yticklabels=l, ax=axes[0])
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('True')
axes[0].set_title('Confusion Matrix Heatmap (Test)')

# Plot CV confusion matrix
sns.heatmap(cm_cv, annot=True, fmt='d', cmap='viridis', xticklabels=l, yticklabels=l, ax=axes[1])
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('True')
axes[1].set_title('Confusion Matrix Heatmap (CV)')

plt.tight_layout()
plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt



# Create KNN classifier object
knn = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors

# Train the model
knn.fit(X_train, y_train)

# Predict for CV and test sets
y_pred_cv_proba = knn.predict_proba(X_cv)
y_pred_test_proba = knn.predict_proba(X_test)
y_pred_cv = knn.predict(X_cv)
y_pred_test = knn.predict(X_test)

# Calculate accuracy scores
accuracy_cv = accuracy_score(y_cv, y_pred_cv)
accuracy_test = accuracy_score(y_test, y_pred_test)

train_loss = log_loss(y_train, knn.predict_proba(X_train))
cv_loss = log_loss(y_cv, y_pred_cv_proba)
test_loss = log_loss(y_test, y_pred_test_proba)

print(f"Cross-validation accuracy: {accuracy_cv}")
print(f"Test accuracy: {accuracy_test}")
print(f"Train Loss: {train_loss}")
print(f"CV Loss: {cv_loss}")
print(f"Test Loss: {test_loss}")

# Generate classification reports
print("Classification Report (CV):\n", classification_report(y_cv, y_pred_cv))
print("Classification Report (Test):\n", classification_report(y_test, y_pred_test))

# Generate confusion matrices
cm_cv = confusion_matrix(y_cv, y_pred_cv)
cm_test = confusion_matrix(y_test, y_pred_test)

# Create subplots for confusion matrix heatmaps
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Plot CV confusion matrix
sns.heatmap(cm_cv, annot=True, fmt='d', cmap='viridis', xticklabels=l, yticklabels=l, ax=axes[0])
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('True')
axes[0].set_title('Confusion Matrix Heatmap (CV)')

# Plot test confusion matrix
sns.heatmap(cm_test, annot=True, fmt='d', cmap='viridis', xticklabels=l, yticklabels=l, ax=axes[1])
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('True')
axes[1].set_title('Confusion Matrix Heatmap (Test)')

plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Assuming you have defined X_train, y_train, and fea_cols
# X_train, y_train, fea_cols = ...

# Create and train a Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Get feature importances
importances = rf.feature_importances_

# Create a DataFrame for feature importances
feature_importances = pd.DataFrame({'feature': fea_cols, 'importance': importances})
feature_importances = feature_importances.sort_values(by='importance', ascending=False)

# Calculate cumulative importance
feature_importances['cumulative_importance'] = feature_importances['importance'].cumsum()

# Create subplots for feature importance and cumulative importance
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot feature importances
axes[0].bar(feature_importances['feature'], feature_importances['importance'])
axes[0].set_xlabel('Features')
axes[0].set_ylabel('Importance')
axes[0].set_title('Feature Importance')
axes[0].tick_params(axis='x', rotation=45)

# Plot cumulative feature importances
axes[1].plot(feature_importances['feature'], feature_importances['cumulative_importance'], marker='o')
axes[1].set_xlabel('Features')
axes[1].set_ylabel('Cumulative Importance')
axes[1].set_title('Cumulative Feature Importance')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Predict for CV and test sets
y_pred_cv = rf.predict(X_cv)
y_pred_test = rf.predict(X_test)
y_pred_cv_proba = rf.predict_proba(X_cv)
y_pred_test_proba = rf.predict_proba(X_test)

accuracy_cv = accuracy_score(y_cv, y_pred_cv)
accuracy_test = accuracy_score(y_test, y_pred_test)

train_loss = log_loss(y_train, rf.predict_proba(X_train))
cv_loss = log_loss(y_cv, y_pred_cv_proba)
test_loss = log_loss(y_test, y_pred_test_proba)

print(f"Cross-validation accuracy: {accuracy_cv}")
print(f"Test accuracy: {accuracy_test}")
print(f"Train Loss: {train_loss}")
print(f"CV Loss: {cv_loss}")
print(f"Test Loss: {test_loss}")

# Generate classification reports
print("Classification Report (CV):\n", classification_report(y_cv, y_pred_cv))
print("Classification Report (Test):\n", classification_report(y_test, y_pred_test))

# Generate confusion matrices
cm_cv = confusion_matrix(y_cv, y_pred_cv)
cm_test = confusion_matrix(y_test, y_pred_test)

# Create subplots for confusion matrix heatmaps
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Plot CV confusion matrix
sns.heatmap(cm_cv, annot=True, fmt='d', cmap='viridis', xticklabels=l, yticklabels=l, ax=axes[0])
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('True')
axes[0].set_title('Confusion Matrix Heatmap (CV)')

# Plot test confusion matrix
sns.heatmap(cm_test, annot=True, fmt='d', cmap='viridis', xticklabels=l, yticklabels=l, ax=axes[1])
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('True')
axes[1].set_title('Confusion Matrix Heatmap (Test)')

plt.tight_layout()
plt.show()

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you have defined X_train, X_cv, X_test, y_train, y_cv, y_test, and l
# X_train, X_cv, X_test, y_train, y_cv, y_test, l = ...

# Create Decision Tree classifier object
dt = DecisionTreeClassifier(random_state=42)

# Train the model
dt.fit(X_train, y_train)

# Predict for CV and test sets
y_pred_cv = dt.predict(X_cv)
y_pred_test = dt.predict(X_test)
y_pred_cv_proba = dt.predict_proba(X_cv)
y_pred_test_proba = dt.predict_proba(X_test)

accuracy_cv = accuracy_score(y_cv, y_pred_cv)
accuracy_test = accuracy_score(y_test, y_pred_test)

# Calculate loss
train_loss = log_loss(y_train, dt.predict_proba(X_train))
cv_loss = log_loss(y_cv, y_pred_cv_proba)
test_loss = log_loss(y_test, y_pred_test_proba)

print(f"Cross-validation accuracy: {accuracy_cv}")
print(f"Test accuracy: {accuracy_test}")
print(f"Train Loss: {train_loss}")
print(f"CV Loss: {cv_loss}")
print(f"Test Loss: {test_loss}")

# Generate classification reports
print("Classification Report (CV):\n", classification_report(y_cv, y_pred_cv))
print("Classification Report (Test):\n", classification_report(y_test, y_pred_test))

# Generate confusion matrices
cm_cv = confusion_matrix(y_cv, y_pred_cv)
cm_test = confusion_matrix(y_test, y_pred_test)

# Create subplots for confusion matrix heatmaps
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Plot CV confusion matrix
sns.heatmap(cm_cv, annot=True, fmt='d', cmap='viridis', xticklabels=l, yticklabels=l, ax=axes[0])
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('True')
axes[0].set_title('Confusion Matrix Heatmap (CV)')

# Plot test confusion matrix
sns.heatmap(cm_test, annot=True, fmt='d', cmap='viridis', xticklabels=l, yticklabels=l, ax=axes[1])
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('True')
axes[1].set_title('Confusion Matrix Heatmap (Test)')

plt.tight_layout()
plt.show()



import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you have defined X_train, X_cv, X_test, y_train, y_cv, y_test, and l
# X_train, X_cv, X_test, y_train, y_cv, y_test, l = ...

# Create AdaBoost classifier object
ab = AdaBoostClassifier(random_state=42)

# Train the model
ab.fit(X_train, y_train)

# Predict for CV and test sets
y_pred_cv = ab.predict(X_cv)
y_pred_test = ab.predict(X_test)

# Generate classification reports
print("Classification Report (CV):\n", classification_report(y_cv, y_pred_cv))
print("Classification Report (Test):\n", classification_report(y_test, y_pred_test))

# Generate confusion matrices
cm_cv = confusion_matrix(y_cv, y_pred_cv)
cm_test = confusion_matrix(y_test, y_pred_test)

# Create subplots for confusion matrix heatmaps
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Plot CV confusion matrix
sns.heatmap(cm_cv, annot=True, fmt='d', cmap='viridis', xticklabels=l, yticklabels=l, ax=axes[0])
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('True')
axes[0].set_title('Confusion Matrix Heatmap (CV)')

# Plot test confusion matrix
sns.heatmap(cm_test, annot=True, fmt='d', cmap='viridis', xticklabels=l, yticklabels=l, ax=axes[1])
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('True')
axes[1].set_title('Confusion Matrix Heatmap (Test)')

plt.tight_layout()
plt.show()

import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Create XGBoost classifier object
xgb_model = xgb.XGBClassifier(random_state=42)

# Initialize LabelEncoder
le = LabelEncoder()

# Fit and transform the target variable
y_train_encoded = le.fit_transform(y_train)

# Train the model with encoded target variable
xgb_model.fit(X_train, y_train_encoded)

# Predict for test set
y_pred = xgb_model.predict(X_test)

# Inverse transform the predicted labels to original classes
y_pred = le.inverse_transform(y_pred)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Generate classification report
print(classification_report(y_test, y_pred))

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Create Gradient Boosting classifier object
gb = GradientBoostingClassifier(random_state=42)

# Train the model
gb.fit(X_train, y_train)

# Predict for CV and test sets
y_pred_cv = gb.predict(X_cv)
y_pred_test = gb.predict(X_test)
y_pred_cv_proba = gb.predict_proba(X_cv)
y_pred_test_proba = gb.predict_proba(X_test)

# Generate confusion matrices
cm_cv = confusion_matrix(y_cv, y_pred_cv)
cm_test = confusion_matrix(y_test, y_pred_test)

accuracy_cv = accuracy_score(y_cv, y_pred_cv)
accuracy_test = accuracy_score(y_test, y_pred_test)

# Calculate loss
train_loss = log_loss(y_train, gb.predict_proba(X_train))
cv_loss = log_loss(y_cv, y_pred_cv_proba)
test_loss = log_loss(y_test, y_pred_test_proba)

print(f"Cross-validation accuracy: {accuracy_cv}")
print(f"Test accuracy: {accuracy_test}")
print(f"Train Loss: {train_loss}")
print(f"CV Loss: {cv_loss}")
print(f"Test Loss: {test_loss}")

# Generate classification reports
print("Classification Report (CV):\n", classification_report(y_cv, y_pred_cv))
print("Classification Report (Test):\n", classification_report(y_test, y_pred_test))


# Create subplots for confusion matrix heatmaps
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Plot CV confusion matrix
sns.heatmap(cm_cv, annot=True, fmt='d', cmap='viridis', xticklabels=l, yticklabels=l, ax=axes[0])
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('True')
axes[0].set_title('Confusion Matrix Heatmap (CV)')

# Plot test confusion matrix
sns.heatmap(cm_test, annot=True, fmt='d', cmap='viridis', xticklabels=l, yticklabels=l, ax=axes[1])
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('True')
axes[1].set_title('Confusion Matrix Heatmap (Test)')

plt.tight_layout()
plt.show()

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

# Define the parameter grid with limited range for faster tuning
param_dist = {
    'n_estimators': np.arange(50, 150, 25),  # reduced range for faster search
    'learning_rate': [0.01, 0.1],  # limited values
    'max_depth': [3, 4],  # smaller set of depths
    'min_samples_split': [2, 5]  # fewer split options
}

# Create a Gradient Boosting Classifier
gb = GradientBoostingClassifier()

# Instantiate RandomizedSearchCV with fewer iterations
random_search = RandomizedSearchCV(estimator=gb, param_distributions=param_dist,
                                   n_iter=5, scoring='accuracy', cv=3, verbose=1, n_jobs=-1, random_state=42)

# Fit the random search model
random_search.fit(X_train, y_train)

best_params = random_search.best_params_
best_model = random_search.best_estimator_

test_accuracy = best_model.score(X_test, y_test)

print("Best Hyperparameters:", best_params)
print("Test Accuracy:", test_accuracy)

# Calculate accuracy for train, cv, and test sets
train_accuracy = accuracy_score(y_train, best_model.predict(X_train))
cv_accuracy = accuracy_score(y_cv, best_model.predict(X_cv))
test_accuracy = accuracy_score(y_test, best_model.predict(X_test))


y_train_pred_proba = best_model.predict_proba(X_train)
y_cv_pred_proba = best_model.predict_proba(X_cv)
y_test_pred_proba = best_model.predict_proba(X_test)


train_loss = log_loss(y_train, y_train_pred_proba)
cv_loss = log_loss(y_cv, y_cv_pred_proba)
test_loss = log_loss(y_test, y_test_pred_proba)


print("Train Loss:", train_loss)
print("CV Loss:", cv_loss)
print("Test Loss:", test_loss)
print("Train Accuracy:", train_accuracy)
print("CV Accuracy:", cv_accuracy)
print("Test Accuracy:", test_accuracy)

import matplotlib.pyplot as plt
import numpy as np

# Model names
models = ['SVM', 'KNN', 'Random Forest', 'Decision Tree',  'Gradient Boosting', 'Tuned Gradient Boosting']

# Accuracy scores
accuracy = [0.95, 0.93, 0.99, 0.98, 0.97, 0.97]

# Colors for each bar
colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta']

# Create bar plot with colors
plt.figure(figsize=(10, 6))
bars = plt.bar(models, accuracy, color=colors)
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.ylim(0.9, 1.0)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Add percentages above bars
for bar in bars:
  yval = bar.get_height()
  plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.0%}', ha='center', va='bottom')

# Display the plot
plt.show()

import pickle

# Assuming 'best_model' is your trained Gradient Boosting model
with open('gradient_boosting_model.pkl', 'wb') as file:
  pickle.dump(best_model, file)

# Download the file (this might be slightly inaccurate, but try the following:)
from google.colab import files
files.download('gradient_boosting_model.pkl')

import joblib # import the joblib module

joblib.dump(best_model, 'my_model.joblib')
from google.colab import files
files.download('my_model.joblib')