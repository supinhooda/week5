from sklearn.datasets import load_iris
import pandas as pd
import pickle
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

df = load_iris()
dataset = pd.DataFrame(df.data)
dataset.columns = df.feature_names
target_mapping={0:'setosa', 1:'versicolor', 2:'virginica'}
dataset['Target'] = df['target']
dataset['Target'] = dataset['Target'].map(target_mapping)
X = dataset.drop(columns='Target')
Y = dataset['Target']

corr= X.corr()

# plt.figure(figsize=(10,10))
# plt.subplot(2,2,1)
# sns.heatmap(data = corr, annot= True)
# plt.yticks(rotation = 0)
# plt.xticks(rotation = 90)
# plt.subplot(2,2,2)
# sns.scatterplot(data=dataset, x='petal length (cm)', y='petal width (cm)', hue='Target', palette={0:'red', 1:'green', 2:'blue'})
# plt.subplot(2,2,3)
# sns.scatterplot(data=dataset, x='petal length (cm)', y='sepal length (cm)', hue='Target', palette={0:'red', 1:'green', 2:'blue'})
# plt.show()

# Model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

model= LogisticRegression()
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=42)

model.fit(X_train, Y_train)

filename = 'model.sav'
pickle.dump(model, open(filename, 'wb'))





