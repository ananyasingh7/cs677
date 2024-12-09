#%%
'''
Name: Ananya Singh
Class: CS677
Date: 12/08/2024
Homework Assignment #6
Description of Problem: Given a seeds dataset, we will implement k-means clustering and use it to construct a multi-label classifier to determine the variety of wheat
'''
#%%
'''
Note: Regarding Question 3.3 - 3.5, you only need to use the two randomly selected features selected in Question 3.2 for the remainder of the assignment (Questions 3.3 - 3.5). Given that you're to do this selection randomly in Question 3.2, I suggest that you make sure that when we re-run the code the same two features are selected using whatever function/code you decide to use to pick the two variables (use packages like random to accomplish this). That way, your analysis will match up when we re-run your code.
'''
#%%
'''
My Buid ends with 8 therefore my remainder is 2. 
R= 2: class L= 1 (negative) and L= 3 (positive)
'''
#%%
import pandas as pd
columns = ['area', 'perimeter', 'compactness', 'length', 'width', 'asymmetry', 'groove', 'class']
data = pd.read_csv('seeds_dataset.txt', sep='\s+', names=columns)
data.head()
#%%
data.info()
#%%
# filter for classes 1 and 3
filtered_data = data[data['class'].isin([1, 3])]
filtered_data.tail()
#%%
filtered_data['class'] = filtered_data['class'].map({1: 0, 3: 1})
filtered_data.tail()
#%%
filtered_data.head()
#%%
print(f"Number of class 0 (originally class 1): {len(filtered_data[filtered_data['class'] == 0])}")
print(f"Number of class 1 (originally class 3): {len(filtered_data[filtered_data['class'] == 1])}")
print(f"Total samples: {len(filtered_data)}")
#%%
from sklearn.model_selection import train_test_split

X = filtered_data.drop('class', axis=1)
y = filtered_data['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
#%%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
#%%
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC

linear_svm = SVC(kernel='linear', random_state=42)
linear_svm.fit(X_train_scaled, y_train)
linear_pred = linear_svm.predict(X_test_scaled)

print("Linear kernel SVM:")
print(f"Accuracy: {accuracy_score(y_test, linear_pred):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, linear_pred))
#%%
gaussian_svm = SVC(kernel='rbf', random_state=42)
gaussian_svm.fit(X_train_scaled, y_train)
gaussian_pred = gaussian_svm.predict(X_test_scaled)

print("Gaussian kernel SVM:")
print(f"Accuracy: {accuracy_score(y_test, gaussian_pred):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, gaussian_pred))
#%%
poly_svm = SVC(kernel='poly', degree=3, random_state=42)
poly_svm.fit(X_train_scaled, y_train)
poly_pred = poly_svm.predict(X_test_scaled)

print("polynomial kernel SVM of degree 3:")
print(f"Accuracy: {accuracy_score(y_test, poly_pred):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, poly_pred))
#%%

#%%
from sklearn.linear_model import LogisticRegression

logistic_regression = LogisticRegression(random_state=42, max_iter=1000)
logistic_regression.fit(X_train_scaled, y_train)
logistic_regression_pred = logistic_regression.predict(X_test_scaled)
#%%
print("Logistic Regression:")
print(f"Accuracy: {accuracy_score(y_test, logistic_regression_pred):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, logistic_regression_pred))
#%%
'''
Used this from prior assignments solutions that were provided to us
'''
import numpy as np
def compute_metrics(y_test,y_pred):
  '''
  Inputs: y_test, y_pred vectors

  Output: DataFrame of metrics
  '''

  # Find tn, fp, fn, tp using confusion_matrix
  cm = confusion_matrix(y_test,y_pred)
  tn, fp, fn, tp = cm.ravel()
  tpr = tp/(tp+fn)
  tnr = tn/(tn+fp)
  accuracy = np.trace(cm)/np.sum(cm) # the trace of a matrix is the sum of the diagonal elements

  return pd.DataFrame({'TP':[tp],'FP':[fp],'TN':[tn],'FN':[fn],'Accuracy':[accuracy],'TPR':[tpr],'TNR':[tnr]})
#%%
linear_metrics = compute_metrics(y_test, linear_pred)
linear_metrics.index = ['Linear SVM']

gaussian_metrics = compute_metrics(y_test, gaussian_pred)
gaussian_metrics.index = ['Gaussian SVM']

poly_metrics = compute_metrics(y_test, poly_pred)
poly_metrics.index = ['Polynomial SVM']

logistic_metrics = compute_metrics(y_test, logistic_regression_pred)
logistic_metrics.index = ['Logistic Regression']

all_metrics = pd.concat([linear_metrics, gaussian_metrics, poly_metrics, logistic_metrics])

all_metrics
#%%

#%%
# Take the original dataset with all 3 class labels
columns = ['area', 'perimeter', 'compactness', 'length', 'width', 'asymmetry', 'groove', 'class']
data = pd.read_csv('seeds_dataset.txt', sep='\s+', names=columns)
#%%
X = data.drop('class', axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#%%
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

distortions = []
for k in range(1, 9):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    distortions.append(kmeans.inertia_)
  
plt.figure(figsize=(10, 6))
plt.plot(range(1, 9), distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('Optimal k')
plt.grid(True)
plt.show()
#%%
import random
np.random.seed(42)
random.seed(42)
#%%
selected_features = random.sample(columns[:-1], 2)
print(selected_features)
#%%
X = data[selected_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)
centroids_scaled = kmeans.cluster_centers_
centroids = scaler.inverse_transform(centroids_scaled)
#%%
plt.figure(figsize=(10, 8))
#%%
# references: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
# https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.scatter.html                
fig, ax = plt.subplots(figsize=(12, 10))
scatter = ax.scatter(X['area'], 
                     X['asymmetry'],
                    c=cluster_labels,
                    cmap='Set1',
                    s=100,
                    alpha=0.7,
                    label='Data Points')

# plot centroids
ax.scatter(centroids[:, 0], 
           centroids[:, 1],
           color='black',       
           marker='*',  
           s=300,   
           label='Centroids',
           edgecolor='white', 
           linewidth=1)

for idx, row in data.iterrows():
    ax.text(row['area'], 
            row['asymmetry'], 
            str(row['class']),
            fontsize=8,
            bbox={'facecolor': 'white', 'alpha': 0.7, 'edgecolor': 'none', 'pad': 1})

ax.set_xlabel('Area')
ax.set_ylabel('Asymmetry')
ax.set_title('Area vs Asymmetry')
ax.grid(True, linestyle='--', alpha=0.3)

plt.colorbar(scatter, label='Cluster Assignment')
ax.legend()
plt.tight_layout()
plt.show()
#%%
for cluster in range(k):
    cluster_data = data[cluster_labels == cluster]
    classes = cluster_data['class'].tolist()
    count_one = classes.count(1)
    count_two = classes.count(2)
    count_three = classes.count(3)
    
    if count_one >= count_two and count_one >= count_three:
        majority_class = 1
    elif count_two >= count_one and count_two >= count_three:
        majority_class = 2
    else:
        majority_class = 3

    print(f"Cluster: {cluster} and Assigned Label: {majority_class}")
    print("Centroid:")
    for idx, feature in enumerate(selected_features): 
        print(f"   {feature}: {centroids[cluster][idx]:.4f}")
    print()
#%%

#%%
# rescale
features_subset = data[['area', 'asymmetry']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features_subset)
#%%
kmeans = KMeans(n_clusters=k, n_init=10, random_state=42) # run it multiple times to get the BEST result
cluster_labels = kmeans.fit_predict(X_scaled)
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
#%%
largest_clusters = {1: (0, 0), 2: (0, 0), 3: (0, 0)}

for cluster in range(k):
    cluster_data = data[cluster_labels == cluster]
    classes = cluster_data['class'].tolist()
    count_one = classes.count(1)
    count_two = classes.count(2)
    count_three = classes.count(3)
    
    if count_one >= count_two and count_one >= count_three:
        majority_class = 1
        if count_one > largest_clusters[1][1]:
            largest_clusters[1] = (cluster, count_one)
    elif count_two >= count_one and count_two >= count_three:
        majority_class = 2
        if count_two > largest_clusters[2][1]:
            largest_clusters[2] = (cluster, count_two)
    else:
        majority_class = 3
        if count_three > largest_clusters[3][1]:
            largest_clusters[3] = (cluster, count_three)

    print(f"Cluster: {cluster} and Assigned Label: {majority_class}")
    print("Centroid:")
    for idx, feature in enumerate(selected_features):
        print(f"   {feature}: {centroids[cluster][idx]:.4f}")
    print()
#%%
# get centroids for largest clusters
representative_centroids = {1: centroids[largest_clusters[1][0]], 2: centroids[largest_clusters[2][0]], 3: centroids[largest_clusters[3][0]]}
#%%
X_test = data[selected_features].values
predictions = []

for point in X_test:
    distances = {}
    
    # distances to clusters
    mu_A = representative_centroids[1]
    dist_to_A = np.linalg.norm(point - mu_A)
    distances[1] = dist_to_A
    
    mu_B = representative_centroids[2]
    dist_to_B = np.linalg.norm(point - mu_B)
    distances[2] = dist_to_B

    mu_C = representative_centroids[3]
    dist_to_C = np.linalg.norm(point - mu_C)
    distances[3] = dist_to_C

    min_distance = float('inf')
    nearest_cluster = None
    for cluster_label, distance in distances.items():
        if distance < min_distance:
            min_distance = distance
            nearest_cluster = cluster_label
    
    predictions.append(nearest_cluster)

predictions = np.array(predictions)
#%%
print(f"accuracy for all 3 classes: {accuracy_score(data['class'], predictions):.4f}")
#%%

#%%
binary_mask = data['class'].isin([1, 3])
binary_true = data.loc[binary_mask, 'class']
binary_pred = predictions[binary_mask]

# class 3 -> 1, class 1 -> 0
binary_true = (binary_true == 3).astype(int)
binary_pred = (binary_pred == 3).astype(int)

print(f"Accuracy: {accuracy_score(binary_true, binary_pred):.4f}")
#%%
print("Class 1 vs 3:")
compute_metrics(binary_true, binary_pred)
#%%
print("Confusion Matrix:")
print(confusion_matrix(binary_true, binary_pred))
#%%
new_k_means_classifier_metrics = compute_metrics(binary_true, binary_pred)
new_k_means_classifier_metrics.index = ['New Classifier']

all_metrics = pd.concat([all_metrics, new_k_means_classifier_metrics])
all_metrics
#%%
