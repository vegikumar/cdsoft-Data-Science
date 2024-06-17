import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("sales prediction\\advertising.csv")
print(df.head())
sns.pairplot(df, x_vars=['TV', 'Radio','Newspaper'], y_vars='Sales', kind='scatter')
plt.show()
df['TV'].plot.hist(bins=10)
plt.show()
df['Radio'].plot.hist(bins=10, color="green", xlabel="Radio")
plt.show()
df['Newspaper'].plot.hist(bins=10,color="purple", xlabel="newspaper")
plt.show()
sns.heatmap(df.corr(),annot = True)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[['TV']], df[['Sales']], test_size = 0.3,random_state=0)
print(X_train)
print(y_train)
print(X_test)
print(y_test)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
res= model.predict(X_test)
print(res)
model.coef_
model.intercept_
plt.plot(res)
plt.scatter(X_test, y_test)
plt.plot(X_test, 7.144 + 0.055 * X_test, 'r')
plt.show()