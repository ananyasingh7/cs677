'''
Name: Ananya Singh
Class: CS677
Date: 10/04/2024
Homework Assignment #2
Description of Problem: Given stock data, we need to discover patterns in the stock behavior and we will make predictions on this.
'''

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

spy_df = pd.read_csv('SPY.csv')
sbux_df = pd.read_csv('SBUX.csv')

spy_df["True Label"] = np.where(spy_df["Return"] >= 0, '+', '-')
sbux_df["True Label"] = np.where(sbux_df["Return"] >= 0, '+', '-')

training_data_for_spy = spy_df[spy_df["Year"].isin([2016, 2017, 2018])]
size_of_training_data_for_spy = len(training_data_for_spy)
size_of_positive_days_for_spy = len(training_data_for_spy[training_data_for_spy["True Label"] == '+'])
probability_next_day_up_for_spy = size_of_positive_days_for_spy/size_of_training_data_for_spy

training_data_for_sbux = sbux_df[sbux_df["Year"].isin([2016, 2017, 2018])]
size_of_training_data_for_sbux = len(training_data_for_sbux)
size_of_positive_days_for_sbux = len(training_data_for_sbux[training_data_for_sbux["True Label"] == '+'])
probability_next_day_up_for_sbux = size_of_positive_days_for_sbux/size_of_training_data_for_sbux

labels_for_spy = training_data_for_spy["True Label"].tolist()
labels_for_sbux = training_data_for_sbux["True Label"].tolist()

def calculate_up_after_consecutive_down_days(labels: [], k: int, ticker: str) -> float:
    '''
    Calculates the probability of an up day occurring after seeing k consecutive down days in the stock price
    '''
    total_consecutive_down_days_for_k = 0
    down_days_then_up_for_k = 0
    size_of_labels = len(labels)
    
    # iterates through the labels array which stops k positions before the end
    for i in range(size_of_labels-k):
        if_down_pattern = True 
        for j in range(k):
            if labels[i+j] != '-': # ensures all next k days are down 
                if_down_pattern = False
                break # break loop if any day isn't down
        
        if if_down_pattern:
            total_consecutive_down_days_for_k += 1
            if i + k < len(labels) and labels[i+k] == '+': # check if next day is up
                down_days_then_up_for_k += 1
    
    probability = down_days_then_up_for_k / total_consecutive_down_days_for_k if total_consecutive_down_days_for_k > 0 else 0
    print(f"Probability of up day after {k} down days for ticker {ticker} : {probability*100:.2f}%")
    return probability

def calculate_up_after_consecutive_up_days(labels: [], k: int, ticker: str) -> float:
    '''
    Calculates the probability of an up day occurring after seeing k consecutive up days in the stock price
    '''
    total_consecutive_up_days_for_k = 0
    up_days_then_up_for_k = 0
    size_of_labels = len(labels)
    
    # iterates through the labels array which stops k positions before the end
    for i in range(size_of_labels-k):
        if_up_pattern = True
        for j in range(k):
            if labels[i+j] != '+': # ensures all next k days are up 
                if_up_pattern = False
                break # break loop if any day isn't up
                
        if if_up_pattern:
            total_consecutive_up_days_for_k += 1
            if i + k < len(labels) and labels[i+k] == '+': # check if next day is up
                up_days_then_up_for_k += 1
    
    probability = up_days_then_up_for_k / total_consecutive_up_days_for_k if total_consecutive_up_days_for_k > 0 else 0
    print(f"Probability of up day after {k} up days for ticker {ticker} after consecutive up days is : {probability*100:.2f}%")
    return probability

def predicting_labels_by_window(training_labels: [], test_labels: [], W: int, ticker: str, default_probability_up_days: float) -> list:
    '''
    Makes predictions using a sliding window approach of size W and builds pattern frequencies from training data
    '''
    size_of_training_labels = len(training_labels)
    size_of_test_labels = len(test_labels) 
    pattern_frequencies = {}
    
    # build pattern frequencies from training data
    for i in range(size_of_training_labels - W):
        pattern = ''.join(training_labels[i:i+W])
        next_day_after_pattern = training_labels[i+W]
        
        if pattern not in pattern_frequencies:
            pattern_frequencies[pattern] = {'+': 0, '-': 0}
        pattern_frequencies[pattern][next_day_after_pattern] += 1
    
    # calculate default probability p* from training data
    default_probability_up_days = training_labels.count('+') / size_of_training_labels
    
    predictions = []
    for i in range(size_of_test_labels - W):
        pattern = ''.join(test_labels[i:i+W])
        
        # if pattern is never seen in training, use default
        if pattern not in pattern_frequencies:
            predictions.append('+' if default_probability_up_days >= 0.5 else '-')
            continue
            
        up_count = pattern_frequencies[pattern]['+']
        down_count = pattern_frequencies[pattern]['-']
        
        if up_count > down_count:
             predictions.append('+')
        elif up_count < down_count:
            predictions.append('-')
        else:
            predictions.append('+' if default_probability_up_days >= 0.5 else '-')
    
    # added NaN padding to match test data length to fix the issue of must have equal len keys and value when setting with an iterable
    full_predictions = [np.nan] * W + predictions
    full_predictions += [np.nan] * (size_of_test_labels - len(full_predictions))
    return full_predictions

def calculate_accuracy(df, W) -> float:
    '''
    Calculates prediction accuracy for a given window size W
    '''
    test_data = df[df['Year'].isin([2019, 2020])].copy()
    correct = 0
    total = 0
    
    for index, row in test_data.iterrows():
        true_label = row['True Label']
        prediction_label = row[f'Predicted_Label_W{W}']
        
        # only count not NaN predictions
        if pd.notna(prediction_label):
            total += 1
            if true_label == prediction_label:
                correct += 1
    
    accuracy = (correct / total) * 100
    
    print(f"\nResults for W={W}:")
    print(f"Correct predictions: {correct}")
    print(f"Total predictions: {total}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    return accuracy

def ensemble_predictions(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Implements ensemble voting strategy using predictions from different window sizes
    '''
    test_years = [2019, 2020]
    test_data = df[df['Year'].isin(test_years)].copy()
    ensemble_labels = []
    
    for idx, row in test_data.iterrows():
        # skip if we don't have valid predictions
        if pd.isna(row['Predicted_Label_W2']):
            ensemble_labels.append(np.nan)
            continue
    
        positive_count = 0
        for W in [2, 3, 4]:
            if row[f'Predicted_Label_W{W}'] == '+':
                positive_count += 1
        
        if positive_count < 2:
            ensemble_labels.append('-')
        else:
            ensemble_labels.append('+')
            
    test_data['Ensemble_Label'] = ensemble_labels
    return test_data 

def calculate_accuracies(df: pd.DataFrame) -> float:
    '''
    Calculates accuracy metrics specifically for the ensemble predictions
    '''

    # check for valid predictions across all W values used in ensemble
    test_years = [2019, 2020]
    test_data = df[df['Year'].isin(test_years)]
    
    valid_predictions = test_data[
        pd.notna(test_data['Predicted_Label_W2']) & 
        pd.notna(test_data['Predicted_Label_W3']) & 
        pd.notna(test_data['Predicted_Label_W4']) &
        pd.notna(test_data['Ensemble_Label'])
    ]
    
    total_predictions = len(valid_predictions)
    
    correct_predictions = 0
    for idx, row in valid_predictions.iterrows():
        if row['Ensemble_Label'] == row['True Label']:
            correct_predictions += 1
            
    # calculate separate accuracies for positive and negative labels
    positive_predictions = valid_predictions[valid_predictions['True Label'] == '+']
    negative_predictions = valid_predictions[valid_predictions['True Label'] == '-']
    
    positive_correct = 0
    for idx, row in positive_predictions.iterrows():
        if row['Ensemble_Label'] == row['True Label']:
            positive_correct += 1
            
    negative_correct = 0
    for idx, row in negative_predictions.iterrows():
        if row['Ensemble_Label'] == row['True Label']:
            negative_correct += 1
    
    return (correct_predictions / total_predictions) * 100

def calculate_metrics(df: pd.DataFrame, prediction_column: str) -> any:
    '''
    Calculates various types of classification metric and returns all the calculated metrics as a tuple
    '''

    # filter rows where both the prediction and true label are not NaN
    valid_data = df.dropna(subset=['True Label', prediction_column])
    
    true_positive = sum((valid_data['True Label'] == '+') & (valid_data[prediction_column] == '+'))
    false_positive = sum((valid_data['True Label'] == '-') & (valid_data[prediction_column] == '+'))
    true_negative = sum((valid_data['True Label'] == '-') & (valid_data[prediction_column] == '-'))
    false_negative = sum((valid_data['True Label'] == '+') & (valid_data[prediction_column] == '-'))
    
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

def create_statistics_table(df_with_ensemble: pd.DataFrame, ticker) -> pd.DataFrame:
    '''
    Creates a statistics table for all prediction methods and retrns a new dataframe
    '''

    statistics_table = []
    
    for W in [2, 3, 4, 'ensemble']:
        pred_column = 'Ensemble_Label' if W == 'ensemble' else f'Predicted_Label_W{W}'
        tp, fp, tn, fn, accuracy, tpr, tnr = calculate_metrics(df_with_ensemble, pred_column)
        statistics_table.append([W, tp, fp, tn, fn, accuracy, tpr, tnr])
    
    results_df = pd.DataFrame(
        statistics_table, 
        columns=['W', 'TP', 'FP', 'TN', 'FN', 'Accuracy', 'TPR', 'TNR']
    )
    return results_df

def plot_portfolio_growth(df: pd.DataFrame, ticker: str, W: int) -> None:
    '''
    Shows portfolio performance for various different strategies
    '''
    initial_investment=100
    data = df[df['Year'].isin([2019, 2020])].copy()
    
    data_with_ensemble = ensemble_predictions(df)
    data['Ensemble_Label'] = data_with_ensemble['Ensemble_Label']

    data['Date'] = pd.to_datetime(data['Date'])
    start_date = data['Date'].iloc[0]
    end_date = data['Date'].iloc[-1] 
    
    buy_and_hold_value = [initial_investment]
    for daily_return in data['Return']:
        new_value = buy_and_hold_value[-1] * (1 + daily_return)
        buy_and_hold_value.append(new_value)
        
    ensemble_portfolio_value = [initial_investment]
    for daily_return, prediction in zip(data['Return'], data['Ensemble_Label']):
        if prediction == '+':
            new_value = ensemble_portfolio_value[-1] * (1 + daily_return)
        else:
            new_value = ensemble_portfolio_value[-1]
        ensemble_portfolio_value.append(new_value)

    predicted_portfolio_value = [initial_investment]
    for daily_return, prediction in zip(data['Return'], data[f'Predicted_Label_W{W}']):
        if prediction == '+':  # Update only on positive predictions
            new_value = predicted_portfolio_value[-1] * (1 + daily_return)
        else:
            new_value = predicted_portfolio_value[-1]
        predicted_portfolio_value.append(new_value)
    
    
    plt.figure(figsize=(20, 10))
    plt.plot(data['Date'], buy_and_hold_value[1:], label='Buy-and-Hold Strategy', 
             linestyle='-', color='yellow')
    plt.plot(data['Date'], ensemble_portfolio_value[1:], label='Ensemble Strategy', 
             linestyle=':', color='red', marker='^', markersize=2)
    plt.plot(data['Date'], predicted_portfolio_value[1:], label=f'Predicted Strategy (W={W})', 
             linestyle='--', color='purple', marker='o', markersize=2)
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    plt.xlim(start_date, end_date)
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.title(f'{ticker} Portfolio Growth (2019-2020)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
