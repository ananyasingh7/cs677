'''
Name: Ananya Singh
Class: CS677
Date: 17/04/2024
Homework Assignment #3
Description of Problem: Given a banknote authentication dataset from a repository, this assignment focuses on classifying real and counterfeit banknotes using k-nn and logistic regression.
'''

import pandas as pd

def analyze_banknote_data(filepath: str) -> pd.DataFrame:
    """
    analyzes banknote data from a CSV file and calculates statistics for each feature
    """
    try:
        # create results dataframe
        results_df = pd.read_csv(filepath, header=None)
        results_df.columns = ['f1', 'f2', 'f3', 'f4', 'class']
        results_df['color'] = results_df['class'].map({0: 'green', 1: 'red'})
        
        results = []
        
        # calculate stats for each class
        for class_num in [0, 1]:
            class_data = results_df[results_df['class'] == class_num]
            
            # calculate means
            f1_mean = class_data['f1'].mean().round(2)
            f2_mean = class_data['f2'].mean().round(2)
            f3_mean = class_data['f3'].mean().round(2)
            f4_mean = class_data['f4'].mean().round(2)
            
            # calculate standard deviations
            f1_std = class_data['f1'].std().round(2)
            f2_std = class_data['f2'].std().round(2)
            f3_std = class_data['f3'].std().round(2)
            f4_std = class_data['f4'].std().round(2)
            
            results.append({
                'Class': class_num,
                'μ(f1)': f1_mean,
                'σ(f1)': f1_std,
                'μ(f2)': f2_mean,
                'σ(f2)': f2_std,
                'μ(f3)': f3_mean,
                'σ(f3)': f3_std,
                'μ(f4)': f4_mean,
                'σ(f4)': f4_std,
            })
        
        # calculate stats for all of the data
        f1_mean_all = results_df['f1'].mean().round(2)
        f2_mean_all = results_df['f2'].mean().round(2)
        f3_mean_all = results_df['f3'].mean().round(2)
        f4_mean_all = results_df['f4'].mean().round(2)
        
        f1_std_all = results_df['f1'].std().round(2)
        f2_std_all = results_df['f2'].std().round(2)
        f3_std_all = results_df['f3'].std().round(2)
        f4_std_all = results_df['f4'].std().round(2)
        
        results.append({
            'Class': 'all',
            'μ(f1)': f1_mean_all,
            'σ(f1)': f1_std_all,
            'μ(f2)': f2_mean_all,
            'σ(f2)': f2_std_all,
            'μ(f3)': f3_mean_all,
            'σ(f3)': f3_std_all,
            'μ(f4)': f4_mean_all,
            'σ(f4)': f4_std_all,
        })
        
        # create and return the results dataframe
        results_df = pd.DataFrame(results)
        return results_df
    
    except Exception as e:
        print(f"Error processing file: {e}")
        return None


import seaborn as sns
import matplotlib.pyplot as plt

def simple_classifier(row) -> str:
    """
    f1 > -1 (ensures that banknotes with very low f1 values which are likely fake are filtered out)
    f2 > 3 (setting f2 > 3 filters out fake banknotes that have negative f2 values)
    f4 < -0.5 (setting f4 < -0.5 ensures that banknotes with strange f4 values are not classified as real)
    """
    if (row['f1'] > -1) and (row['f2'] > 3) and (row['f4'] < -0.5):
        return 'green'
    else:
        return 'red'

def calculate_metrics(df, column) -> tuple:
    '''
    calculates various types of classification metric and returns all the calculated metrics as a tuple
    same method was used in assignment #2
    '''
    valid_data = df.dropna(subset=['color', column])

    true_positive = sum((valid_data['color'] == 'red') & (valid_data[column] == 'red'))
    false_positive = sum((valid_data['color'] == 'green') & (valid_data[column] == 'red'))
    true_negative = sum((valid_data['color'] == 'green') & (valid_data[column] == 'green'))
    false_negative = sum((valid_data['color'] == 'red') & (valid_data[column] == 'green'))
    
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
    
    return true_positive, false_positive, true_negative, false_negative, accuracy, true_positive_rate, true_negative_rate

def get_train_and_test_data(X,y) -> tuple:
    '''
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    '''
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, stratify=y, random_state=20
    )
    return X_train, X_test, y_train, y_test

def analyze_banknote_classification(filepath):
    """
    performs classification analysis on banknote data with visualization and metrics table
    """
    try:
        results_df = pd.read_csv(filepath, header=None)
        results_df.columns = ['f1', 'f2', 'f3', 'f4', 'class']
        results_df['color'] = results_df['class'].map({0: 'green', 1: 'red'})
    
        # differentiate features/target using a fixed random state
        X = results_df[['f1', 'f2', 'f3', 'f4']]
        y = results_df['color']
        X_train, X_test, y_train, y_test = get_train_and_test_data(X, y)
        
        # back into dfs
        train_data = X_train.copy()
        train_data['color'] = y_train
        test_data = X_test.copy()
        test_data['color'] = y_test
        
        # create pairplots
        correct_notes = train_data[train_data['color'] == 'green']
        sns.pairplot(correct_notes.drop('color', axis=1), diag_kind='hist')
        plt.savefig('good_bills.pdf')
        plt.close()
        
        fake_notes = train_data[train_data['color'] == 'red']
        sns.pairplot(fake_notes.drop('color', axis=1), diag_kind='hist')
        plt.savefig('fake_bills.pdf')
        plt.close()

        # make predictions
        test_data['Prediction'] = test_data.apply(simple_classifier, axis=1)
        
        # calculate metrics
        tp, fp, tn, fn, acc, tpr, tnr = calculate_metrics(test_data, 'Prediction')
        
        # create results df
        results = pd.DataFrame({
            'TP': [tp],
            'FP': [fp],
            'TN': [tn],
            'FN': [fn],
            'accuracy': [round(acc, 2)],
            'TPR': [round(tpr, 2)],
            'TNR': [round(tnr, 2)]
        })
        
        return results
    
    except Exception as e:
        print(f"Error processing file: {e}")
        return None


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def calculate_metrics_for_knn(df, column) -> tuple:
    valid_data = df.copy()
    true_positive = sum((valid_data['True_Label'] == 1) & (valid_data[column] == 1))
    false_positive = sum((valid_data['True_Label'] == 0) & (valid_data[column] == 1))
    true_negative = sum((valid_data['True_Label'] == 0) & (valid_data[column] == 0))
    false_negative = sum((valid_data['True_Label'] == 1) & (valid_data[column] == 0))

    if (true_positive + false_negative) > 0:
        true_positive_rate = true_positive / (true_positive + false_negative)
    else:
        true_positive_rate = 0

    if (true_negative + false_positive) > 0:
        true_negative_rate = true_negative / (true_negative + false_positive)
    else:
        true_negative_rate = 0

    total = true_positive + false_positive + true_negative + false_negative
    accuracy = (true_positive + true_negative) / total if total > 0 else 0

    return true_positive, false_positive, true_negative, false_negative, accuracy, true_positive_rate, true_negative_rate


def analyze_banknote_knn(filepath):
    """
    performs k-NN classification analysis on data with different k values and computes metrics
    """
    try:
        results_df = pd.read_csv(filepath, header=None)
        results_df.columns = ['f1', 'f2', 'f3', 'f4', 'class']
        X = results_df[['f1', 'f2', 'f3', 'f4']]
        y = results_df['class']

        X_train, X_test, y_train, y_test = get_train_and_test_data(X, y)

        k_values = [3, 5, 7, 9, 11]
        accuracies = []
        predictions = []

        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k) # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
            accuracies.append(accuracy)
            predictions.append(y_pred)
            print(f" k={k}, accuracy={round(accuracy, 4)}")

        # plot accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, accuracies, marker='o')
        plt.xlabel('k')
        plt.ylabel('accuracy')
        plt.title('k-NN Classifier')
        plt.savefig('knn.pdf')
        plt.close()

        # find optimal k and its predictions
        max_accuracy = accuracies[0] 
        optimal_index = 0
        for i in range(len(accuracies)):
            if accuracies[i] >= max_accuracy:
                max_accuracy = accuracies[i]
                optimal_index = i
        
        optimal_k = k_values[optimal_index]
        y_pred = predictions[optimal_index]
        
        print()
        print(f"Optimal k: {optimal_k}")

        results_df = pd.DataFrame({
            'True_Label': y_test,
            'Predicted': y_pred
        })

        tp, fp, tn, fn, acc, tpr, tnr = calculate_metrics_for_knn(results_df, 'Predicted')
        results = pd.DataFrame({
            'k*': [optimal_k],
            'TP': [tp],
            'FP': [fp],
            'TN': [tn],
            'FN': [fn],
            'accuracy': [round(acc, 2)],
            'TPR': [round(tpr, 2)],
            'TNR': [round(tnr, 2)]
        })

        # my BUID ends with 3968
        buid_features = pd.DataFrame([[3, 9, 6, 8]], columns=['f1', 'f2', 'f3', 'f4'])
        knn = KNeighborsClassifier(n_neighbors=optimal_k)
        knn.fit(X_train, y_train)
        if knn.predict(buid_features)[0] == 0:
            buid_prediction = 'green' 
        else: 
            buid_prediction = 'red'
        return results, optimal_k, buid_prediction

    except Exception as e:
        print(f"Error processing file: {e}")
        return None, None, None


def analyze_feature_importance_for_knn(filepath):
    """
    analyzes the importance of each feature by removing one at a time and evaluating k-NN performance
    """
    try:
        results_df = pd.read_csv(filepath, header=None)
        results_df.columns = ['f1', 'f2', 'f3', 'f4', 'class']
        X = results_df[['f1', 'f2', 'f3', 'f4']]
        y = results_df['class']

        X_train, X_test, y_train, y_test = get_train_and_test_data(X, y)
        
        # optimal_k was calculated in the previous function
        
        results = []        
        X_train_drop_f1 = X_train[['f2', 'f3', 'f4']]
        X_test_drop_f1 = X_test[['f2', 'f3', 'f4']]
        knn = KNeighborsClassifier(n_neighbors=optimal_k) # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        knn.fit(X_train_drop_f1, y_train) 
        y_pred = knn.predict(X_test_drop_f1)
        accuracy = accuracy_score(y_test, y_pred) # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
        results.append({
            'Dropped Feature': 'f1',
            'Accuracy': round(accuracy, 4)
        })
        
        X_train_drop_f2 = X_train[['f1', 'f3', 'f4']]
        X_test_drop_f2 = X_test[['f1', 'f3', 'f4']]
        knn = KNeighborsClassifier(n_neighbors=optimal_k)
        knn.fit(X_train_drop_f2, y_train)
        y_pred = knn.predict(X_test_drop_f2)
        accuracy = accuracy_score(y_test, y_pred)
        results.append({
            'Dropped Feature': 'f2',
            'Accuracy': round(accuracy, 4)
        })

        X_train_drop_f3 = X_train[['f1', 'f2', 'f4']]
        X_test_drop_f3 = X_test[['f1', 'f2', 'f4']]
        knn = KNeighborsClassifier(n_neighbors=optimal_k)
        knn.fit(X_train_drop_f3, y_train)
        y_pred = knn.predict(X_test_drop_f3)
        accuracy = accuracy_score(y_test, y_pred)
        results.append({
            'Dropped Feature': 'f3',
            'Accuracy': round(accuracy, 4)
        })

        X_train_drop_f4 = X_train[['f1', 'f2', 'f3']]
        X_test_drop_f4 = X_test[['f1', 'f2', 'f3']]
        knn = KNeighborsClassifier(n_neighbors=optimal_k)
        knn.fit(X_train_drop_f4, y_train)
        y_pred = knn.predict(X_test_drop_f4)
        accuracy = accuracy_score(y_test, y_pred)
        results.append({
            'Dropped Feature': 'f4',
            'Accuracy': round(accuracy, 4)
        })

        results_df = pd.DataFrame(results)
        return results_df
    
    except Exception as e:
        print(f"Error processing file: {e}")
        return None


from sklearn.linear_model import LogisticRegression
def analyze_banknote_logistic(filepath):
    """
    performs logistic regression analysis on banknote data and computes metrics
    """
    try:
        results_df = pd.read_csv(filepath, header=None)
        results_df.columns = ['f1', 'f2', 'f3', 'f4', 'class']
        X = results_df[['f1', 'f2', 'f3', 'f4']]
        y = results_df['class']

        X_train, X_test, y_train, y_test = get_train_and_test_data(X, y)
        
        # train logistic regression model
        log_reg = LogisticRegression(random_state=20)
        log_reg.fit(X_train, y_train)
        
        # make predictions
        y_pred = log_reg.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Logistic regression accuracy: {round(accuracy, 4)}")
        
        results_df = pd.DataFrame({
            'True_Label': y_test,
            'Predicted': y_pred
        })

        tp, fp, tn, fn, acc, tpr, tnr = calculate_metrics_for_knn(results_df, 'Predicted')
        results = pd.DataFrame({
            'TP': [tp],
            'FP': [fp],
            'TN': [tn],
            'FN': [fn],
            'accuracy': [round(acc, 3)],
            'TPR': [round(tpr, 3)],
            'TNR': [round(tnr, 3)]
        })
        
        # my BUID ends with 3968
        buid_features = pd.DataFrame([[3, 9, 6, 8]], columns=['f1', 'f2', 'f3', 'f4'])
        if log_reg.predict(buid_features)[0] == 0:
            buid_prediction = 'green' 
        else: 
            buid_prediction = 'red'
        
        return results, buid_prediction
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return None, None

def analyze_feature_importance_for_logistic(filepath):
    """
    analyzes the importance of each feature by removing one at a time and evaluating logistic regression performance
    """
    try:
        results_df = pd.read_csv(filepath, header=None)
        results_df.columns = ['f1', 'f2', 'f3', 'f4', 'class']
        X = results_df[['f1', 'f2', 'f3', 'f4']]
        y = results_df['class']

        X_train, X_test, y_train, y_test = get_train_and_test_data(X, y)

        results = []        
        
        X_train_drop_f1 = X_train[['f2', 'f3', 'f4']]
        X_test_drop_f1 = X_test[['f2', 'f3', 'f4']]
        log_reg = LogisticRegression(random_state=20)
        log_reg.fit(X_train_drop_f1, y_train)
        y_pred = log_reg.predict(X_test_drop_f1)
        accuracy = accuracy_score(y_test, y_pred)
        results.append({
            'Dropped Feature': 'f1',
            'Accuracy': round(accuracy, 4)
        })
        
        X_train_drop_f2 = X_train[['f1', 'f3', 'f4']]
        X_test_drop_f2 = X_test[['f1', 'f3', 'f4']]
        log_reg = LogisticRegression(random_state=20)
        log_reg.fit(X_train_drop_f2, y_train)
        y_pred = log_reg.predict(X_test_drop_f2)
        accuracy = accuracy_score(y_test, y_pred)
        results.append({
            'Dropped Feature': 'f2',
            'Accuracy': round(accuracy, 4)
        })

        X_train_drop_f3 = X_train[['f1', 'f2', 'f4']]
        X_test_drop_f3 = X_test[['f1', 'f2', 'f4']]
        log_reg = LogisticRegression(random_state=20)
        log_reg.fit(X_train_drop_f3, y_train)
        y_pred = log_reg.predict(X_test_drop_f3)
        accuracy = accuracy_score(y_test, y_pred)
        results.append({
            'Dropped Feature': 'f3',
            'Accuracy': round(accuracy, 4)
        })

        X_train_drop_f4 = X_train[['f1', 'f2', 'f3']]
        X_test_drop_f4 = X_test[['f1', 'f2', 'f3']]
        log_reg = LogisticRegression(random_state=20)
        log_reg.fit(X_train_drop_f4, y_train)
        y_pred = log_reg.predict(X_test_drop_f4)
        accuracy = accuracy_score(y_test, y_pred)
        results.append({
            'Dropped Feature': 'f4',
            'Accuracy': round(accuracy, 4)
        })

        results_df = pd.DataFrame(results)
        return results_df
    
    except Exception as e:
        print(f"Error processing file: {e}")
        return None