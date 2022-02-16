import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
from sklearn.svm import SVC
from joblib import dump

df = pd.read_csv('heart.csv')

df = df.drop(df.index[449], axis=0)

df['Cholesterol'] = df['Cholesterol'].replace(
    0, np.int64(round(df['Cholesterol'].mean())))

IQR = 267-199
lower_limit_cholesterol = 199 - 1.5*(IQR)
upper_limit_cholesterol = 267 + 1.5*(IQR)

df = df[(df['Cholesterol'] > lower_limit_cholesterol) &
        (df['Cholesterol'] < upper_limit_cholesterol)]

df = pd.get_dummies(df, drop_first=True)

X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=101)

scaler = StandardScaler()

scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

log_model = LogisticRegression()
log_model.fit(scaled_X_train, y_train)
y_pred = log_model.predict(scaled_X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(log_model.score(scaled_X_test, y_test))

plot_confusion_matrix(log_model, scaled_X_test, y_test)
plt.show()

# model_params = {
#     'svm': {
#         'model': SVC(gamma='auto', probability=True),
#         'params': {
#             'C': [1, 10, 100, 1000],
#             'kernel': ['rbf', 'linear']
#         }},
#     'random_forest': {
#         'model': RandomForestClassifier(),
#         'params': {
#             'n_estimators': [1, 5, 10]
#         }
#     },
#     'logistic_regression': {
#         'model': LogisticRegression(solver='liblinear', multi_class='auto'),
#         'params': {
#             'C': [1, 5, 10]
#         }
#     }}


# scores = []
# best_estimators = {}

# for algo, mp in model_params.items():
#     grid_model = GridSearchCV(
#         mp['model'], mp['params'], cv=5, return_train_score=False)
#     grid_model.fit(scaled_X_train, y_train)
#     scores.append({
#         'model': algo,
#         'best_score': grid_model.best_score_,
#         'best_params': grid_model.best_params_
#     })
#     best_estimators[algo] = grid_model.best_estimator_

# model_details = pd.DataFrame(
#     scores, columns=['model', 'best_score', 'best_params'])

# dump(log_model, 'final_log_model.joblib')
