#%%
'''
Name: Ananya Singh
Class: CS677
Date: 11/25/2024
Homework Assignment #4
Description of Problem: Given heart failure clinical records from a repository, this assignment focuses on using linear models to analyze relationships. Given 4 features, our goal is to establish relationships with using various
    different linear models e.g. linear regression etc. 
'''
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

data = pd.read_csv('heart_failure_clinical_records_dataset.csv')
df_0 = data[data['DEATH_EVENT'] == 0]
df_1 = data[data['DEATH_EVENT'] == 1]

features = ['creatinine_phosphokinase', 'serum_creatinine', 'serum_sodium', 'platelets']

M0 = df_0[features].corr()
M1 = df_1[features].corr()

# https://seaborn.pydata.org/generated/seaborn.heatmap.html
plt.figure(figsize=(8, 8))
sns.heatmap(M0, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='.2f') 
plt.title('Correlation Matrix Surviving')
plt.savefig('correlation_matrix_surviving.pdf')
plt.close()

plt.figure(figsize=(8, 8))
sns.heatmap(M1, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='.2f') 
plt.title('Correlation Matrix Deceased')
plt.savefig('correlation_matrix_deceased.pdf')
plt.close()

def analyze_linear_regression(df, if_death: bool):
    '''
    y= ax + b (simple linear regression)
    '''
    X = df['serum_sodium'].values.reshape(-1, 1) # scikit.LinearRegression expects the input features to be in 2D array
    y = df['serum_creatinine'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42) # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    
    model = LinearRegression() 
    model.fit(X_train, y_train)

    a = model.coef_[0] # coef_array of shape (n_features, ) or (n_targets, n_features)
    b = model.intercept_ # float or array of shape (n_targets,)
    print(f"Weights: a: {a:.2f}, b: {b:.2f}")
    
    # make predictions
    y_pred = model.predict(X_test)
    
    # calculate SSE
    residuals = y_test - y_pred
    sse = np.sum(residuals**2)
    print(f"SSE: {sse:.2f}")
    
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_with_legend.html#sphx-glr-gallery-lines-bars-and-markers-scatter-with-legend-py
    plt.figure(figsize=(12, 7))
    plt.scatter(X_test, y_test, color='blue', label='Actual Values', alpha=0.5)
    plt.plot(X_test, y_pred, color='red', label='Predicted Values')
    plt.xlabel('Serum Sodium')
    plt.ylabel('Serum Creatinine')
    plt.title(f'Simple Linear Regression Death: {if_death} (y = ax + b)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'linear_regression_death_{if_death}.pdf')
    plt.close()
    
    return a, b, sse, X_test, y_test, y_pred

print(f"Simple Linear Regression (y = ax + b)")
print()
print(f"Results for Survived patients: ")
results_0 = analyze_linear_regression(df_0, False)
print()
print(f"Results for Died patients: ")
results_1 = analyze_linear_regression(df_1, True)

def analyze_quadratic_regression(df, if_death: bool):
    '''
    y = ax^2 + bx + c (quadratic regression)
    '''
    
    # get original X and transform to include x^2
    X_original = df['serum_sodium'].values.reshape(-1, 1)
    X_squared = X_original**2
    # combine X and X^2
    X = np.hstack([X_original, X_squared])  # https://www.geeksforgeeks.org/numpy-hstack-in-python/
    y = df['serum_creatinine'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    b, a = model.coef_ 
    c = model.intercept_
    print(f"Weights: a: {a:.2f}, b: {b:.2f}, c: {c:.2f}")
    
    # make predictions
    y_pred = model.predict(X_test)
    
    # calculate SSE
    residuals = y_test - y_pred
    sse = np.sum(residuals**2)
    print(f"SSE: {sse:.2f}")
    
    X_test_orig = X_test[:, 0]  # get original x values from test set
    sort_idx = np.argsort(X_test_orig) # need to sort X values to get a "smoother" curve
    X_test_sorted = X_test_orig[sort_idx]
    y_pred_sorted = y_pred[sort_idx]
    
    plt.figure(figsize=(12, 7))
    plt.scatter(X_test_orig, y_test, color='blue', label='Actual Values', alpha=0.5)
    plt.plot(X_test_sorted, y_pred_sorted, color='red', label='Predicted Values')
    plt.xlabel('Serum Sodium')
    plt.ylabel('Serum Creatinine')
    plt.title(f'Quadratic Regression Death: {if_death} (y = ax² + bx + c)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'quadratic_regression_death_{if_death}.pdf')
    plt.close()
    
    return a, b, c, sse, X_test_orig, y_test, y_pred

print(f"Quadratic Regression (y = ax² + bx + c)")
print()
print(f"Results for Survived patients: ")
results_0_quad = analyze_quadratic_regression(df_0, False)
print()
print(f"Results for Died patients: ")
results_1_quad = analyze_quadratic_regression(df_1, True)

def analyze_cubic_regression(df, if_death: bool):
    '''
    y = ax^3 + bx^2 + cx + d (cubic regression)
    '''
    
    # get original X and transform to include x^2 and x^3
    X_orig = df['serum_sodium'].values.reshape(-1, 1)
    X_squared = X_orig**2
    X_cubed = X_orig**3
    X = np.hstack([X_orig, X_squared, X_cubed])  # combine X, X^2, and X^3
    y = df['serum_creatinine'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # c for x, b for x^2, a for x^3
    c, b, a = model.coef_ 
    d = model.intercept_
    print(f"Weights: a: {a:.2f}, b: {b:.2f}, c: {c:.2f}, d: {d:.2f}")
    
    # make predictions
    y_pred = model.predict(X_test)
    
    # calculate SSE
    residuals = y_test - y_pred
    sse = np.sum(residuals**2)
    print(f"SSE: {sse:.2f}")
    
    X_test_orig = X_test[:, 0]  # get original x values from test set
    sort_idx = np.argsort(X_test_orig)  # need to sort X values to get a "smoother" curve
    X_test_sorted = X_test_orig[sort_idx]
    y_pred_sorted = y_pred[sort_idx]

    plt.figure(figsize=(12, 7))
    plt.scatter(X_test_orig, y_test, color='blue', label='Actual Values', alpha=0.5)
    plt.plot(X_test_sorted, y_pred_sorted, color='red', label='Predicted Values')
    plt.xlabel('Serum Sodium')
    plt.ylabel('Serum Creatinine')
    plt.title(f'Cubic Regression Death: {if_death} (y = ax³ + bx² + cx + d)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'cubic_regression_death_{if_death}.pdf')
    plt.close()
    
    return a, b, c, d, sse, X_test_orig, y_test, y_pred

print(f"Cubic Regression (y = ax³ + bx² + cx + d)")
print()
print(f"Results for Survived patients: ")
results_0_cubic_spline = analyze_cubic_regression(df_0, False)
print()
print(f"Results for Died patients: ")
results_1_cubic_spline = analyze_cubic_regression(df_1, True)

def analyze_log_regression(df, if_death: bool):
    '''
    y = a log(x) + b (GLM - generalized linear model)
    '''
    
    # get original X and transform to log(X)
    X_orig = df['serum_sodium'].values.reshape(-1, 1)
    # since log is undefined for non-positive values, ensure all values are positive
    if np.any(X_orig <= 0):
        X_orig = X_orig + 1  # adds 1 to all values to make sure they are above 0 no matter what
        
    X_log = np.log(X_orig)
    y = df['serum_creatinine'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X_log, y, test_size=0.5, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    a = model.coef_[0]
    b = model.intercept_
    print(f"Weights: a: {a:.2f}, b: {b:.2f}")
    
    # make predictions
    y_pred = model.predict(X_test)
    
    # calculate SSE
    residuals = y_test - y_pred
    sse = np.sum(residuals**2)
    print(f"SSE: {sse:.2f}")
    
    # for plotting, we need the original X values (not log-transformed)
    X_test_orig = np.exp(X_test)  # convert back to original scale
    sort_idx = np.argsort(X_test_orig.flatten())  # need to sort X values to get a "smoother" curve
    X_test_sorted = X_test_orig.flatten()[sort_idx]
    y_pred_sorted = y_pred[sort_idx]
    
    plt.figure(figsize=(12, 7))
    plt.scatter(X_test_orig, y_test, color='blue', label='Actual Values', alpha=0.5)
    plt.plot(X_test_sorted, y_pred_sorted, color='red', label='Predicted Values')
    plt.xlabel('Serum Sodium')
    plt.ylabel('Serum Creatinine')
    plt.title(f'Logarithmic Regression Death: {if_death} (y = a log(x) + b)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'log_regression_death_{if_death}.pdf')
    plt.close()
    
    return a, b, sse, X_test_orig.flatten(), y_test, y_pred

print(f"Logarithmic Regression (y = a log(x) + b)")
print()
print(f"Results for Survived patients: ")
results_0_log = analyze_log_regression(df_0, False)
print()
print(f"Results for Died patients: ")
results_1_log = analyze_log_regression(df_1, True)

def analyze_loglog_regression(df, if_death: bool):
    '''
    log(y) = a log(x) + b (GLM - generalized linear model)
    '''
    
    # get original X and Y and transform both to log scale
    X_orig = df['serum_sodium'].values.reshape(-1, 1)
    y_orig = df['serum_creatinine'].values.reshape(-1, 1)
    
    # ensure all values are positive for both X and Y
    if np.any(X_orig <= 0) or np.any(y_orig <= 0):
        X_orig = X_orig + 1
        y_orig = y_orig + 1
    
    # transform to log scale
    X_log = np.log(X_orig)
    y_log = np.log(y_orig).ravel()  # flatten y to 1D array
    
    X_train, X_test, y_train, y_test = train_test_split(X_log, y_log, test_size=0.5, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # scalars
    a = float(model.coef_[0])  # convert to float
    b = float(model.intercept_)  # convert to float
    print(f"Weights: a: {a:.2f}, b: {b:.2f}")
    
    # make predictions
    y_pred_log = model.predict(X_test)
    
    # transform predictions back to original scale
    y_pred = np.exp(y_pred_log)
    y_test_orig = np.exp(y_test)
    
    # calculate SSE in original scale
    residuals = y_test_orig - y_pred
    sse = np.sum(residuals**2)
    print(f"SSE: {sse:.2f}")
    
    # for plotting, we need the original X values
    X_test_orig = np.exp(X_test)
    sort_idx = np.argsort(X_test_orig.flatten())
    X_test_sorted = X_test_orig.flatten()[sort_idx]
    y_pred_sorted = y_pred[sort_idx]
    
    plt.figure(figsize=(12, 7))
    plt.scatter(X_test_orig, y_test_orig, color='blue', label='Actual Values', alpha=0.5)
    plt.plot(X_test_sorted, y_pred_sorted, color='red', label='Predicted Values')
    plt.xlabel('Serum Sodium')
    plt.ylabel('Serum Creatinine')
    plt.title(f'Log-Log Regression Death: {if_death} (log(y) = a log(x) + b)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'loglog_regression_death_{if_death}.pdf')
    plt.close()
    
    return a, b, sse, X_test_orig.flatten(), y_test_orig, y_pred

print(f"Log-Log Regression (log(y) = a log(x) + b)")
print()
print(f"Results for Survived patients: ")
results_0_loglog = analyze_loglog_regression(df_0, False)
print()
print(f"Results for Died patients: ")
results_1_loglog = analyze_loglog_regression(df_1, True)

def create_summary_table():
    '''
    create a summary table of SSE values for all models
    '''
    
    models = [
        'y = ax + b',
        'y = ax^2 + bx + c',
        'y = ax^3 + bx^2 + cx + d',
        'y = a log(x) + b',
        'log(y) = a log(x) + b'
    ]

    sse_0 = [
        results_0[2],          
        results_0_quad[3],     
        results_0_cubic_spline[4], 
        results_0_log[2],     
        results_0_loglog[2]   
    ]
    
    sse_1 = [
        results_1[2],         
        results_1_quad[3],    
        results_1_cubic_spline[4],  
        results_1_log[2],      
        results_1_loglog[2]   
    ]
    
    summary_df = pd.DataFrame({
        'Model': models,
        'SSE (death_event=0)': [f"{x:.2f}" for x in sse_0],
        'SSE (death_event=1)': [f"{x:.2f}" for x in sse_1]
    })
    return summary_df

print("Summary of SSE values for all models:")
summary_table = create_summary_table()
summary_table