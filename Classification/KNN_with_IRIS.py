# Classification with KNN 
# Import Libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Dataset 
iris = load_iris()
iris.feature_names
Data_iris = iris.data
Data_iris = pd.DataFrame(Data_iris, columns = iris.feature_names)
Data_iris['Labels']= iris.target

# plot 2-Dimensional
plt.scatter(x= Data_iris.iloc[:,2],
            y= Data_iris.iloc[:,3],
            c= Data_iris['Labels'],
            )
plt.xlabel('Petal Length(cm)')
plt.ylabel('Petal Width(cm)')
plt.show()

x = Data_iris.iloc[:,:-1]
y = Data_iris.iloc[:,4]
 
"""
KNN Classifiers 
"""
# Manhatan Distance with p = 1 and Euclidean Distance with p = 2
KNN = KNeighborsClassifier(n_neighbors= 6, metric= 'minkowski', p= 1)
KNN.fit(x,y)
x_N = np.array([[5.6, 3.4, 1.4, 0.1]])
x_N2 = np.array([[3.5, 4, 5.5, 2]])
KNN.predict(x_N)
KNN.predict(x_N2)

# Splitting train and test 
X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                    test_size = 0.2,
                                                    train_size = 0.8,
                                                    random_state = 88,
                                                    shuffle = True,
                                                    stratify = y
                                                    )
KNN = KNeighborsClassifier(n_neighbors= 6, metric= 'minkowski', p= 1)
KNN.fit(X_train,y_train)
y_predicted = KNN.predict(X_test)

# Evaluating Model 
accuracy_score(y_test, y_predicted)
