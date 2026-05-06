
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
import joblib

data=pd.read_csv('data/job_data.csv')


cat_col=data.select_dtypes(include=["object"]).columns
num_col=data.select_dtypes(include=["number"]).columns

from sklearn.impute import SimpleImputer
imp_cat = SimpleImputer(strategy='most_frequent')
data[['company_type', 'job_role']] = imp_cat.fit_transform(data[['company_type', 'job_role']])






# Actually, since multiple, let's create separate
le_branch = LabelEncoder()
data['branch'] = le_branch.fit_transform(data['branch'])
joblib.dump(le_branch, 'le_branch.pkl')

le_company_type = LabelEncoder()
data['company_type'] = le_company_type.fit_transform(data['company_type'])
joblib.dump(le_company_type, 'le_company_type.pkl')

le_job_role = LabelEncoder()
data['job_role'] = le_job_role.fit_transform(data['job_role'])
joblib.dump(le_job_role, 'le_job_role.pkl')

filter_data=data.drop('student_id',axis=1)



"""Here 2 model is need to build for this purpose bcz it produce output for 1.Regression and Classification ,i have to use Ensemble leraning model Stacking and Random Forest

"""

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

# --- Classification Model (Predicting 'job_role') ---
print('Building Classification Model (Predicting job_role)...')

X_cls = filter_data.drop(columns=['placed', 'salary_lpa', 'job_role'])
y_cls = filter_data['job_role']

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

# Initialize and train XGBClassifier
xgb_classifier = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_classifier.fit(X_train_cls, y_train_cls)

# Make predictions and evaluate classifier
y_pred_cls = xgb_classifier.predict(X_test_cls)
accuracy = accuracy_score(y_test_cls, y_pred_cls)
print(f'XGBoost Classifier Accuracy: {accuracy:.4f}')

print('\nBuilding Regression Model (Predicting salary_lpa)...')

X_reg = filter_data.drop(columns=['salary_lpa', 'placed', 'job_role'])
y_reg = filter_data['salary_lpa']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

xgb_regressor = XGBRegressor(random_state=42)
xgb_regressor.fit(X_train_reg, y_train_reg)
y_pred_reg = xgb_regressor.predict(X_test_reg)
mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mse)
print(f'XGBoost Regressor RMSE: {rmse:.4f}')



import joblib

print('\nSaving models...')
joblib.dump(xgb_classifier, 'classifier.pkl')
joblib.dump(xgb_regressor, 'regressor.pkl')
print('Models saved successfully.')