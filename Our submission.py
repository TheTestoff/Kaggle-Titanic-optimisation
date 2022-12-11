# Pandas for loading data and One-Hot Encoding Categorical Data
import pandas as pd
import numpy as np

# Sklearn for imputing missing values and model building
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Location of Titanic files
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Load training data
train = pd.read_csv("C:/Users/Scapji/Desktop/Competition/train.csv")

# Load test data
test = pd.read_csv("C:/Users/Scapji/Desktop/Competition/test.csv")
# ===============================================================

X = train.copy()
y = train["Survived"]
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=0)

# ===============================================================

mode_imputer = SimpleImputer(strategy='most_frequent')

X_train_full[['Age']] = mode_imputer.fit_transform(X_train_full[['Age']])

features = ['Pclass', 'Age']
X_train_fullt = X_train_full[features]

Sex_encoded = LabelEncoder().fit_transform(X_train_full.Sex)
X_train_fullt.insert(0, "Sex_encoded", Sex_encoded, True)
X_train_full = X_train_fullt

# ===============================================================

mode_imputer = SimpleImputer(strategy='most_frequent')

X_valid_full[['Age']] = mode_imputer.fit_transform(X_valid_full[['Age']])

features = ['Pclass', 'Age']
X_valid_fullt = X_valid_full[features]

Sex_encoded = LabelEncoder().fit_transform(X_valid_full.Sex)
X_valid_fullt.insert(0, "Sex_encoded", Sex_encoded, True)

X_valid_full = X_valid_fullt

# ===============================================================

# Distribution of ages is skewed to the right. Impute with mode (most_frequent)
mode_imputer = SimpleImputer(strategy='most_frequent')

# Apply imputer on Age column
test['Age'] = mode_imputer.fit_transform(test['Age'].values.reshape(-1, 1))

X_test = test.copy()

features = ['Pclass', 'Age']

X_test = X_test[features]

Sex_encoded = LabelEncoder().fit_transform(test.Sex)
X_test.insert(0, "Sex_encoded", Sex_encoded, True)

# ===============================================================

model = XGBRegressor(n_estimators=1000, random_state=0, learning_rate = 0.05)
model.fit(X_train_full, y_train)

# Get predictions on test data
preds = model.predict(X_valid_full)

mae_1 = mean_absolute_error(y_valid, preds)

print(mae_1)

preds_sub = model.predict(X_test)

preds_sub = [round(value) for value in preds_sub]

# Save to CSV file
output = pd.DataFrame({'PassengerId': test.PassengerId,
                      'Survived': preds_sub})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")
