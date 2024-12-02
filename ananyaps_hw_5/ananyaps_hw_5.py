#%%
'''
Name: Ananya Singh
Class: CS677
Date: 11/30/2024
Homework Assignment #5
Description of Problem: Given fetal cardiotocology data set, Naive Bayesian and Decision Tree Classification for identifying normal vs. non-normal fetus status based on fetal cardiograms
'''
#%%
import pandas as pd
CTG = 'CTG.xls'
data = pd.read_excel(CTG, sheet_name="Raw Data", engine="xlrd") 
data = data.drop(index=0)
data.reset_index(drop=True, inplace=True)
data = data[:2126]
data.reset_index(drop=True, inplace=True)
print(data.tail())
#%%
data.info()
#%%
data.loc[data['NSP'] == 2, 'NSP'] = 0
data.loc[data['NSP'] == 3, 'NSP'] = 0

print(data['NSP'].value_counts())
print(data.head(40))
#%%
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

selected_features = ['MSTV', 'Width', 'Mode', 'Variance']
X = data[selected_features]
y = data['NSP']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# scale the features
scaler = StandardScaler() # https://scikit-learn.org/dev/modules/generated/sklearn.preprocessing.StandardScaler.html
X_train_scaled = scaler.fit_transform(X_train) # use fit the scaler on the training data and then apply to test data
X_test_scaled = scaler.transform(X_test)

# training on Xtrain
nb_classifier = GaussianNB() # https://scikit-learn.org/dev/modules/generated/sklearn.naive_bayes.GaussianNB.html
nb_classifier.fit(X_train_scaled, y_train)

# predict class labels in Xtest
y_pred = nb_classifier.predict(X_test_scaled) # make predictions

accuracy = accuracy_score(y_test, y_pred)
print(f"{accuracy*100:.2f}%")
#%%
confusion_matrix_nb = confusion_matrix(y_test, y_pred) # https://scikit-learn.org/dev/modules/generated/sklearn.metrics.confusion_matrix.html
print(confusion_matrix_nb)
#%%
from sklearn.tree import DecisionTreeClassifier

dt_classifier = DecisionTreeClassifier(random_state=42) #https://scikit-learn.org/dev/modules/generated/sklearn.tree.DecisionTreeClassifier.html
dt_classifier.fit(X_train, y_train) # using same 50/50 split from earlier question

# make predictions based on Xtest
dt_pred = dt_classifier.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_pred)
print(f"{dt_accuracy*100:.2f}%")
#%%
dt_confusion_matrix = confusion_matrix(y_test, dt_pred)
print(dt_confusion_matrix)
#%%
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

X = data[selected_features]
y = data['NSP']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# take N = 1, . . . , 10 and d = 1, 2, . . . , 5
NValues= range(1, 11)
dValues = range(1, 6)
error_rates = np.zeros((len(NValues), len(dValues))) # stores error rates for different combinations of N and d 

'''
we need to test every possible combination of N and d (number of trees and max depth) params
for each "iteration":
    create a Random Forest model with the param
    train it
    make predictions on test data
    calculate the error rate
    store it in error_rates matrix for the combination
'''
for i, n_trees in enumerate(NValues):
    for j, max_depth in enumerate(dValues):
        rf = RandomForestClassifier( # https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.RandomForestClassifier.html
            n_estimators=n_trees,
            max_depth=max_depth,
            criterion='entropy', # use ”entropy” as splitting criteria
            random_state=42
        )
        rf.fit(X_train, y_train) # train rf on training data
        y_pred = rf.predict(X_test) # after training, model makes predictions on new data
        error_rate = 1 - accuracy_score(y_test, y_pred) # calculate the error rate 
        error_rates[i, j] = error_rate # store it in error_rates matrix for the combination

'''
heatmap is effective way to visualize and compare results with two different params
here we have 50 different combinations so now we can see all the results together
'''
plt.figure(figsize=(10, 8))
plt.imshow(error_rates) # https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
plt.colorbar(label='Error Rate')
plt.xlabel('d')
plt.ylabel('N')
plt.title('Error Rates')
plt.xticks(range(len(dValues)), dValues)
plt.yticks(range(len(NValues)), NValues)

for i in range(len(NValues)):
    for j in range(len(dValues)):
        plt.text(j, i, f'{error_rates[i,j]:.2f}')

plt.savefig('error_rates.pdf')
plt.close()

#%%
min_error = 9999999999999999 # set min to maximum amount
i_best_index = 0 
j_best_index = 0

for i in range(len(NValues)):
    for j in range(len(dValues)):
        if error_rates[i][j] < min_error:
            min_error = error_rates[i][j]
            i_best_index = i
            j_best_index = j

best_N = NValues[i_best_index]
best_d = dValues[j_best_index]
print(f"N={best_N}, d={best_d}")
#%%
best_accuracy = 1 - error_rates[i_best_index, j_best_index]
print(f"Best accuracy: {best_accuracy*100:.2f}%")
#%%
best_combo_rf = RandomForestClassifier(
    n_estimators=best_N,
    max_depth=best_d,
    criterion='entropy',
    random_state=42
)
best_combo_rf.fit(X_train, y_train)
best_pred = best_combo_rf.predict(X_test)
best_combo_confusion_matrix = confusion_matrix(y_test, best_pred)
print(best_combo_confusion_matrix)
#%%
# naive bayesian
true_positive = confusion_matrix_nb[1][1]
false_positive = confusion_matrix_nb[0][1]
true_negative = confusion_matrix_nb[0][0]
false_negative = confusion_matrix_nb[1][0]

if (true_positive + false_negative) > 0:
    true_positive_rate = true_positive / (true_positive + false_negative)
else:
    true_positive_rate = 0
if (true_negative + false_positive) > 0:
    true_negative_rate = true_negative / (true_negative + false_positive)
else:
    true_negative_rate = 0
    
if (true_positive + false_positive + true_negative + false_negative) > 0:
    accuracy = (true_positive + true_negative) / (true_positive + false_positive + true_negative + false_negative)
else:
    accuracy = 0
nb_results = [true_positive, false_positive, true_negative, false_negative, accuracy, true_positive_rate, true_negative_rate]

# decision tree
true_positive = dt_confusion_matrix[1][1]
false_positive = dt_confusion_matrix[0][1]
true_negative = dt_confusion_matrix[0][0]
false_negative = dt_confusion_matrix[1][0]

if (true_positive + false_negative) > 0:
    true_positive_rate = true_positive / (true_positive + false_negative)
else:
    true_positive_rate = 0
if (true_negative + false_positive) > 0:
    true_negative_rate = true_negative / (true_negative + false_positive)
else:
    true_negative_rate = 0
    
if (true_positive + false_positive + true_negative + false_negative) > 0:
    accuracy = (true_positive + true_negative) / (true_positive + false_positive + true_negative + false_negative)
else:
    accuracy = 0
dt_results = [true_positive, false_positive, true_negative, false_negative, accuracy, true_positive_rate, true_negative_rate]

# random forest (best combo)
true_positive = best_combo_confusion_matrix[1][1]
false_positive = best_combo_confusion_matrix[0][1]
true_negative = best_combo_confusion_matrix[0][0]
false_negative = best_combo_confusion_matrix[1][0]

if (true_positive + false_negative) > 0:
    true_positive_rate = true_positive / (true_positive + false_negative)
else:
    true_positive_rate = 0
if (true_negative + false_positive) > 0:
    true_negative_rate = true_negative / (true_negative + false_positive)
else:
    true_negative_rate = 0
    
if (true_positive + false_positive + true_negative + false_negative) > 0:
    accuracy = (true_positive + true_negative) / (true_positive + false_positive + true_negative + false_negative)
else:
    accuracy = 0
rf_results = [true_positive, false_positive, true_negative, false_negative, accuracy, true_positive_rate, true_negative_rate]

results = pd.DataFrame({
    'Model': ['naive bayesian', 'decision tree', 'random forest'],
    'TP': [nb_results[0], dt_results[0], rf_results[0]],
    'FP': [nb_results[1], dt_results[1], rf_results[1]],
    'TN': [nb_results[2], dt_results[2], rf_results[2]],
    'FN': [nb_results[3], dt_results[3], rf_results[3]],
    'accuracy': [f"{nb_results[4]*100:.2f}%", f"{dt_results[4]*100:.2f}%", f"{rf_results[4]*100:.2f}%"],
    'TPR': [nb_results[5], dt_results[5], rf_results[5]],
    'TNR': [nb_results[6], dt_results[6], rf_results[6]]
})

results
#%%
