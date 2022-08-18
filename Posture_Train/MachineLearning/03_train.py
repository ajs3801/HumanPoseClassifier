from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score # Accuracy metrics

import pandas as pd
import pickle 

# custom here
PATH_CSV = "Posture_Data/MachineLearning/Dataset.csv"
SAVE_NAME = "ActionV8"

# Read Data(CSV)
df = pd.read_csv(PATH_CSV)

# 
X = df.drop('class', axis=1) # features
y = df['class'] # target value

# split to train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

pipelines = {
    'lr':make_pipeline(StandardScaler(), LogisticRegression()),
    'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
}

fit_models = {}
for algo, pipeline in pipelines.items():
  model = pipeline.fit(X_train, y_train)
  fit_models[algo] = model
  print("### TRAIN COMPLETE ### Algorithm : {}".format(algo))

print(fit_models)

for algo, model in fit_models.items():
    yhat = model.predict(X_test)
    print(algo, accuracy_score(y_test, yhat))
    
# Save the model
with open(SAVE_NAME+"_lr.pkl", 'wb') as f:
  pickle.dump(fit_models['lr'], f)

print('Save logisticRegresstion model..')

with open(SAVE_NAME+"_rc.pkl", 'wb') as f:
  pickle.dump(fit_models['rc'], f)

print('Save RidgeClassifier model..')

with open(SAVE_NAME+"_rf.pkl", 'wb') as f:
  pickle.dump(fit_models['rf'], f)

print('Save RandomForestClassifier model..')

with open(SAVE_NAME+"_gb.pkl", 'wb') as f:
  pickle.dump(fit_models['gb'], f)

print('Save GradientBoostingClassifier model..')