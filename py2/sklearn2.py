import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets, preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
iris = datasets.load_iris()
X, y = iris.data[:,:2], iris.target
#print(X,y)
X_train, X_test,y_train,y_test=train_test_split(X,y,random_state=33)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy_score(y_test, y_pred)
print(y_pred)
print(accuracy_score)


# Loading the Data
"""
Your data needs to be numeric and stored as Numpy arrays or SciPy sparse
matrices. Other types that are convertible to numeric arrays, such as Pandas
DataFrame,are also acceptable.
"""
X = np.random.random((10,5))
#print(X)
y = np.array(['M','M','F','F','M','F','M','M','F','F'])
print(y)
X[X < 0.7] = 0.5
print(X)


# Training And Test Data
X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=0)
print(y_test)
print(y_train)


# Preprocessing The Data
#Standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
standardized_X = scaler.transform(X_train)
standardized_X_test = scaler.transform(X_test)
#print(standardized_X)
#print(standardized_X_test)
# Normalization
from sklearn.preprocessing import Normalizer
scaler = Normalizer().fit(X_train)
normalized_X = scaler.transform(X_train)
normalized_X_test = scaler.transform(X_test)
print(normalized_X)
print(normalized_X_test)
# Binarization
from sklearn.preprocessing import Binarizer
binarizer = Binarizer(threshold=0.0).fit(X)
binary_X = binarizer.transform(X)
#print(binary_X)
# Create Your Model
# Supervised Learning Estimators
# Linear Regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression(normalize=True)
# Support Vector Machines(SVM)
from sklearn.svm import SVC
svc = SVC(kernel='linear')
# Naive Bayes
from sklearn.naive_bayes
gnb = GaussianNB()
# KNN
from sklearn import neighbors
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
# Unsupervised Learning estimators
# Principal Component Analysis(PCA)
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)

# K Means
from sklearn.cluster import KMeans
k_means = KMeans(n_clusters=3, random_state=0)

# Model Fitting
Supervised learning
lr.fit(X,y)
knn.fit(X_train, y_train)
svc.fit(X_train, y_train)

# Unsupervised Learning
k_means.fit(X_train)                          # Fit the model to the data
pca_model = pca.fit_transform(X_train)        # Fit to data, then transform it

# Evaluate Your Model's Performance
# Classification Metrics
# Accuracy Score
knn.score(X_test, y_test)                     # Estimator score method
from sklearn.metrics import accuracy_score    # Metric scoring functions
accuracy_score(y_test,y_pred)

# Classification Report
from sklearn.metrics import cassification_report
print(classification_report(y_test,y_pred))    # Precision,recall,f1-score and support

# Regression Metrics
# Mean Absolute Error
from sklearn.metrics import mean_absolute_error
y_true = [3, -0.5, 2]
mean_absolute_error(y_true, y_pred)

# Mean Squared Error
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,y_pred)

# R**2 score
from sklearn.metrics import r2_score
r2_score(y_true,y_pred)


# Clustering Metrics
# Adjusted Rand Index
from sklearn.metrics import adjusted_rand_score
adjusted_rand_score(y_true, y_pred)

# Homogeneity
from sklearn.metrics import homogeneity_score
homogeneity_score(y_true, y_pred)

# V-measure
from sklearn.metrics import v_measure_score
metrics.v_measure_score(y_true, y_pred)

# Cross-Validation
from sklearn.cross_validation import cross_val_score
print(cross_val_score(knn, X_train, y_train, cv=4))
print(cross_val_score(lr, X, y,cv=2))

# Tune Your Model
# Grid Search

