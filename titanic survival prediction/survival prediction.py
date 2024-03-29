import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("TITANIC SURVIVAL PREDICTION\\Titanic-Dataset.csv")
print(df.head(10))
print(df.shape)
print(df.describe)
print(df['Survived'].value_counts())
sns.countplot(x=df['Survived'], hue=df['Pclass'])
plt.show()
print(df["Sex"])
sns.countplot(x=df['Sex'], hue=df['Survived'])
plt.show()
print(df.groupby('Sex')[['Survived']].mean())
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df['Sex']= labelencoder.fit_transform(df['Sex'])
print(df.head())
print(df['Sex'], df['Survived'])
sns.countplot(x=df['Sex'], hue=df["Survived"])
plt.show()
print(df.isna().sum())
df=df.drop(['Age'], axis=1)
print(df)
df_final = df
print(df_final.head(10))

'''Modelling'''
X= df[['Pclass', 'Sex']]
Y=df['Survived']
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
from sklearn.linear_model import LogisticRegression
log = LogisticRegression(random_state = 0)
log.fit(X_train, Y_train)

'''Prediction'''
pred = print(log.predict(X_test))
print(pred)
print(print(Y_test))
import warnings
warnings.filterwarnings("ignore")
value1 = int(input("Enter the person class: "))
value2 = int(input("Enter the passengerID: "))
res= log.predict([[value1,value2]])
if(res==0):
  print("So Sorry! Not Survived")
else:
  print("Survived")
