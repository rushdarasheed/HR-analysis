# -*- coding: utf-8 -*-
"""final_assesment.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1FHqchtd72X-i3pqGJ3dBEHdLpKlYMp2-
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('train_LZdllcl.csv')
test = pd.read_csv('test_2umaH9m.csv')
sample_submission= pd.read_csv('sample_submission_M0L0uXE.csv')
train

train.head()

train.tail()

train.info()

train.describe()

train.dtypes

print(train.columns.tolist())

sns.countplot(x='is_promoted',data=train)
plt.title("Target Class Distribution")
plt.show()

print(train['is_promoted'].value_counts(normalize=True))

# Identify numeric features
numeric_features = train.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Identify categorical features
categorical_features = train.select_dtypes(include=['object']).columns.tolist()

# Print the results
print("Numeric Features:")
print(numeric_features)

print("\nCategorical Features:")
print(categorical_features)

# Identify categorical features
categorical_features = train.select_dtypes(include=['object']).columns.tolist()

for col in categorical_features:
    print(f"\nDistinct values in '{col}':")
    print(train[col].unique())

categorical_features = train.select_dtypes(include=['object']).columns.tolist()
selected_features = categorical_features[:3]

plt.figure(figsize=(18, 5))
for i, col in enumerate(selected_features):
    plt.subplot(1, 3, i + 1)
    sns.countplot(data=train, x=col)
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

train.isnull().sum()

#IMPUTATION
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy="most_frequent")
imputer.fit(train[['education','previous_year_rating']])
train[['education','previous_year_rating']]=imputer.transform(train[['education','previous_year_rating']])
train

train.isnull().sum()

#Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train['department'] = le.fit_transform(train['department'])
train['region'] = le.fit_transform(train['region'])
train['gender'] = le.fit_transform(train['gender'])
train['education'] = le.fit_transform(train['education'])
train['recruitment_channel'] = le.fit_transform(train['recruitment_channel'])
train

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train[['no_of_trainings','age','previos_year_rating','length_of_service','avg_training_score']] = scaler.fit_transform(train[['no_of_trainings','age','previous_year_rating','length_of_service','avg_training_score']])
train

#train_test_split
from sklearn.model_selection import train_test_split
x = train.drop('is_promoted', axis=1)
y = train['is_promoted']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print("Shape of x_train: ", x_train.shape)
print("Shape of x_test: ", x_test.shape)
print("Shape of y_train: ", y_train.shape)
print("Shape of y_test: ", y_test.shape)

# Model Selection Training - Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=10000)
lr.fit(x_train, y_train)

#Evaluation
y_pred_lr = lr.predict(x_test)
y_pred_lr

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred_lr))
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# MODEL Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train, y_train)

y_pred_rf = rf.predict(x_test)
y_pred_rf

print(accuracy_score(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# MODEL  k-Nearest Neighbors (kNN)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)

y_pred_knn = knn.predict(x_test)
y_pred_knn

print(accuracy_score(y_test, y_pred_knn))
print(confusion_matrix(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

# MODEL GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier()
gb.fit(x_train, y_train)

y_pred_gb = gb.predict(x_test)
y_pred_gb

print(accuracy_score(y_test, y_pred_gb))
print(confusion_matrix(y_test, y_pred_gb))
print(classification_report(y_test, y_pred_gb))

# MODEL MLP Classifier
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier()
mlp.fit(x_train, y_train)

y_pred_mlp = mlp.predict(x_test)
y_pred_mlp

print(accuracy_score(y_test, y_pred_mlp))
print(confusion_matrix(y_test, y_pred_mlp))
print(classification_report(y_test, y_pred_mlp))

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(random_state=42)

param_grid = {
    'n_estimators': [100, 150],         # number of boosting stages
    'learning_rate': [0.05, 0.1],       # step size shrinkage
    'max_depth': [3, 4, 5],             # depth of individual trees
    'subsample': [0.8, 1.0]             # fraction of samples used for fitting
}

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def get_metrics(model_name, y_true, y_pred):
    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred)
    }

results = []

# Logistic Regression
results.append(get_metrics("Logistic Regression", y_test, y_pred_lr))

# k-Nearest Neighbors
results.append(get_metrics("k-NN", y_test, y_pred_knn))

# Random Forest
results.append(get_metrics("Random Forest", y_test, y_pred_rf))

# MLP Classifier
results.append(get_metrics("MLP Classifier", y_test, y_pred_mlp))

# Gradient Boosting
results.append(get_metrics("Gradient Boosting", y_test, y_pred_gb))

comparison_df = pd.DataFrame(results)
comparison_df.set_index("Model", inplace=True)
comparison_df = comparison_df.sort_values(by="F1-Score", ascending=False)
comparison_df.style.background_gradient(cmap="YlGnBu", axis=1)

test

test.isnull().sum()

#IMPUTATION
from sklearn.impute import SimpleImputer
#np.nan = NaN
imputer=SimpleImputer(missing_values=np.nan,strategy="most_frequent")
imputer.fit(test[['education','previous_year_rating']])
test[['education','previous_year_rating']]=imputer.transform(test[['education','previous_year_rating']])
test

#Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
test['department'] = le.fit_transform(test['department'])
test['region'] = le.fit_transform(test['region'])
test['gender'] = le.fit_transform(test['gender'])
test['education'] = le.fit_transform(test['education'])
test['recruitment_channel'] = le.fit_transform(test['recruitment_channel'])
test

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
test[['no_of_trainings','age','previos_year_rating','length_of_service','avg_training_score']] = scaler.fit_transform(test[['no_of_trainings','age','previous_year_rating','length_of_service','avg_training_score']])
test

test_predictions = gb.predict(test)

sample_submission

sample_submission.to_csv("final_submission.csv",index=False)

