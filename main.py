import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
data=pd.read_csv("heart.csv")
print(data.head())
corrmat=data.corr()
top_corr_features=corrmat.index
plt.figure(figsize=(16,16))
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap='RdYlGn')
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
standardScaler = StandardScaler()
columns_to_scale = ['age', 'resting bp s', 'cholesterol','max heart rate', 'oldpeak']
dataset = pd.get_dummies(data, columns = ['sex', 'chest pain type', 
                                        'fasting blood sugar','resting ecg', 
                                        'exercise angina', 'ST slope'])
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])
dataset.head()
y = dataset['target']
X = dataset.drop(['target'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)


models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Support Vector Classifier": SVC(),
    "Random Forest Classifier":RandomForestClassifier(n_estimators=31),
    "XGBRegression":XGBRegressor(n_estimators=10),
    "Gradient Boosting Classifier": GradientBoostingClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "MLP Classifier": MLPClassifier()
}


for name, model in models.items():
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"{name} Accuracy: {score * 100:.2f}%")