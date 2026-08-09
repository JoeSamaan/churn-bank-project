import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
#boosting
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.model_selection import RandomizedSearchCV
# deep learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.models import load_model

import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,ConfusionMatrixDisplay,confusion_matrix , classification_report
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier

import warnings
warnings.filterwarnings("ignore")

# to make data balance 
from imblearn.over_sampling import SMOTE


def load_data():
    df = pd.read_csv("Customer-Churn-Records.csv")
    return df
df = pd.read_csv('Customer-Churn-Records.csv')
df.head()
print("Head  \n",df.head())
df.isnull().sum()
df.info()
print("data info \n",df.info())
print("null value \n", df.isnull().sum())
df.describe()
print(" decs \n",df.describe())
print("Data type \n",df.dtypes)
df.columns
print("columns Names \n",df.columns)
df.shape
checkzerosinmydata = ['Age','Balance','Tenure','NumOfProducts','EstimatedSalary','Satisfaction Score']
print("check zeros in my data \n",(df[checkzerosinmydata]== 0).sum())
dataclean = df.copy()

mappingGender = {
   'Male': 1, 
   'Female': 0
}

dataclean["Gender"] = dataclean["Gender"].map(mappingGender)


dataclean = pd.get_dummies(
    dataclean,
    columns=["Geography", "Card Type"],
    drop_first=True
)


dataclean.drop(columns=['Surname', 'CustomerId', 'RowNumber','Complain'], inplace=True)
print("dataclean\n",dataclean.head())



X = dataclean.drop('Exited', axis=1)
y = dataclean['Exited']


x_train , x_test , y_train , y_test = train_test_split(X,y,random_state=43,test_size=0.2,stratify=y)


scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


smote = SMOTE(random_state=43)

X_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)
X_train_smote_Scaled, y_train_smote = smote.fit_resample(x_train_scaled, y_train)
plt.Figure(figsize=(2,4))
y.value_counts().plot(kind='bar')
plt.show()



plt.Figure(figsize=(2,4))
plt.hist(dataclean['Balance'] , bins=30)
plt.show()




# Deep Learning 
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_smote_Scaled.shape[1],)),
    Dropout(0.3),

    Dense(32, activation='relu'),
    Dropout(0.2),

    Dense(16, activation='relu'),

    Dense(1, activation='sigmoid')
])

early = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)



history = model.fit(
    X_train_smote_Scaled,
    y_train_smote,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early],
    verbose=1
    
)


model.save("DeepLearning.keras")
# model.save("DeepLearning.keras")

# print("Deep Learning Model Saved Successfully.")


pred_prob = model.predict(x_test_scaled)

PredDL = (pred_prob > 0.5).astype(int)

accDL = accuracy_score(y_test, PredDL)
recallDL = recall_score(y_test, PredDL)
preDL = precision_score(y_test, PredDL)
f1DL = f1_score(y_test, PredDL)

confDL = confusion_matrix(y_test, PredDL)
reportDL = classification_report(y_test, PredDL)
print(classification_report(y_test, PredDL))









XGB = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss",
    
)

XGB.fit(X_train_smote,y_train_smote)
PredXGB = XGB.predict(x_test)
accXGB = accuracy_score(y_test,PredXGB)
recallXGB = recall_score(y_test,PredXGB)
preXGB = precision_score(y_test,PredXGB)
f1XGB= f1_score(y_test,PredXGB)
confXGB = confusion_matrix(y_test,PredXGB)
sns.heatmap(confusion_matrix(y_test,PredXGB),annot=True)
reportXGB = classification_report(y_test,PredXGB )
print("accuracy  = ", accXGB,"\nf1 = " , f1XGB, "\nprecision_score = ",preXGB,"\nrecall = ", recallXGB , "\nconfusion_matrix = \n", confXGB)
print("full reprot  \n ",reportXGB)
ConfusionMatrixDisplay.from_predictions(y_test,PredXGB )
plot_importance(XGB)
Importance=plot_importance(XGB)
plt.show()



LG = LogisticRegression(max_iter=1000 , random_state=42)
LG.fit(X_train_smote_Scaled,y_train_smote)
PredLG = LG.predict(x_test_scaled)
accLG = accuracy_score(y_test,PredLG)
recallLG = recall_score(y_test,PredLG)
preLG = precision_score(y_test,PredLG)
f1LG= f1_score(y_test,PredLG)
confLG = confusion_matrix(y_test,PredLG)
sns.heatmap(confusion_matrix(y_test,PredLG),annot=True)
reportLG = classification_report(y_test,PredLG )
print("accuracy  = ", accLG,"\nf1 = " , f1LG , "\nprecision_score = ",preLG,"\nrecall = ", recallLG , "\nconfusion_matrix = \n", confLG )
print("full reprot  \n ",reportLG)
ConfusionMatrixDisplay.from_predictions(y_test,PredLG )



RF = RandomForestClassifier(n_estimators=100 , random_state=42)
RF.fit(X_train_smote,y_train_smote)
PredRF = RF.predict(x_test)
accRF = accuracy_score(y_test,PredRF)
recallRF = recall_score(y_test,PredRF)
preRF = precision_score(y_test,PredRF)
f1RF= f1_score(y_test,PredRF)
confRF = confusion_matrix(y_test,PredRF)
sns.heatmap(confusion_matrix(y_test,PredRF),annot=True)
reportRF = classification_report(y_test,PredRF )
print("accuracy  = ", accRF,"\nf1 = " , f1RF, "\nprecision_score = ",preRF,"\nrecall = ", recallRF , "\nconfusion_matrix = \n", confRF )
print("full reprot  \n ",reportRF)
ConfusionMatrixDisplay.from_predictions(y_test,PredRF )



DT = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=42)
DT.fit(X_train_smote,y_train_smote)
PredDT= DT.predict(x_test)
accDT = accuracy_score(y_test,PredDT)
recallDT = recall_score(y_test,PredDT )
preDT = precision_score(y_test,PredDT)
f1DT= f1_score(y_test,PredDT)
confDT = confusion_matrix(y_test,PredDT)
sns.heatmap(confusion_matrix(y_test,PredDT),annot=True)
reportDT = classification_report(y_test,PredDT )
print("accuracy  = ", accDT,"\nf1 = " , f1DT , "\nprecision_score = ",preDT,"\nrecall = ", recallDT , "\nconfusion_matrix = \n", confDT )
print("full reprot  \n ",reportDT)
ConfusionMatrixDisplay.from_predictions(y_test,PredDT )



comp = pd.DataFrame(
    {
        'Model' : [  'Deep Learning (TensorFlow)','XGB','LogisticRegression' ,'RandomForestClassifier' , 'DecisionTreeClassifier'],
        'accuracy' :[accDL  ,accXGB  ,accLG  , accRF , accDT ],
        'recall' : [     recallDL    ,recallXGB  ,recallLG  ,recallRF, recallDT ],
        'precision' : [  preDL     ,preXGB    ,preLG ,  preRF , preDT ],
        'f1' : [    f1DL      ,f1XGB     ,f1LG ,f1RF,f1DT],
    }

)
print(comp)


print(X.columns.tolist())
print(X.dtypes)
print(y.value_counts())
print(X.shape)

model = {
    "Deep Learning": model,
    "XGBoost": XGB,
    "Random Forest": RF,
    "Decision Tree": DT,
    "Logistic Regression": LG
}

X_column = X.columns.tolist()


results = {
    "Logistic Regression": PredLG,
    "Decision Tree": PredDT,
    "Random Forest": PredRF,
    "XGBoost": PredXGB,
    "Deep Learning (TensorFlow)": PredDL
}