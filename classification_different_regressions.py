# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 19:25:08 2023

@author: Patrycja Czemerych
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import pandas as pd


DIRECTORY = 'csv_T1'

file_types = ['output_D_corr.csv', 'output_D_fi.csv', 'output_D_mi.csv',
              'output_S_corr.csv', 'output_S_fi.csv', 'output_S_mi.csv',
              'output_M_corr.csv', 'output_M_fi.csv', 'output_M_mi.csv',
              'output_N_corr.csv', 'output_N_fi.csv', 'output_N_mi.csv']

excel_file_path = 'C:/Users/HP/Desktop/regression_results_both_T1.xlsx'
excel_writer = pd.ExcelWriter(excel_file_path, engine='xlsxwriter')

for i, TYPE_NORM in enumerate(file_types):
    file_path = f'C:/Users/HP/Desktop/{DIRECTORY}/{TYPE_NORM}'

    br_feature_dict = {}
    gr_feature_dict = {}

    try:
        with open(file_path) as csvfile:
            data = csv.reader(csvfile)
            header = next(data)  # Read the header row to get parameter names
            for row in data:
                roi_name = row[-1]
                features = row[:-1]  # Exclude the ROI category column

                # Extract ROI category (br or gr) and age from the ROI name
                roi_category = roi_name[:2]  # Extract the first two characters (br or gr)
                age = roi_name[2:]  # Extract the remaining characters (age)

                # Organize features based on ROI category and age
                roi_features = dict(zip(header, features))

                if roi_category == 'br':
                    br_feature_dict.setdefault(age, []).append(roi_features)
                elif roi_category == 'gr':
                    gr_feature_dict.setdefault(age, []).append(roi_features)

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except PermissionError:
        print(f"Permission denied: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

    # Extract features and target variable for br and gr
    br_features = []
    br_ages = []
    for age, features_list in br_feature_dict.items():
        for features in features_list:
            br_features.append([float(value) for value in features.values()])
            br_ages.append(int(age))

    gr_features = []
    gr_ages = []
    for age, features_list in gr_feature_dict.items():
        for features in features_list:
            gr_features.append([float(value) for value in features.values()])
            gr_ages.append(int(age))

    # Combine features and ages for both br and gr
    all_features = np.vstack([br_features, gr_features])
    all_ages = np.concatenate([br_ages, gr_ages])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(all_features, all_ages, test_size=0.2, random_state=42)

    # Regression models and predictions
    models = {
        'Linear Regression': LinearRegression(),
        'LASSO Regression': LassoCV(max_iter=100000),
        'Decision Tree Regression': DecisionTreeRegressor(),
        'Support Vector Regression': SVR(),
        'Ridge Regression': RidgeCV(alphas=[0.1, 1.0, 10.0]),
        'Random Forest Regression': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    results = {'Model': [], 'MSE': [], 'R-squared': []}


    plt.figure(figsize=(15, 10))
    for j, (model_name, model) in enumerate(models.items()):
        plt.subplot(2, 3, j + 1)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate MSE and R-squared for each model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results['Model'].append(model_name)
        results['MSE'].append(mse)
        results['R-squared'].append(r2)


        plt.scatter(y_test, y_pred, label=model_name)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', label='Perfect Prediction')
        plt.scatter(np.unique(y_test), [np.mean(y_pred[y_test == age]) for age in np.unique(y_test)], color='black', marker='o', label='Mean Predicted Age')
        plt.title(f'{model_name} - {TYPE_NORM[:-4]}')
        plt.xlabel('Actual Age')
        plt.ylabel('Predicted Age')
        plt.legend()


    results_df = pd.DataFrame(results)
    results_df.to_excel(excel_writer, sheet_name=TYPE_NORM[:-4], index=False)

    plt.tight_layout()
    plt.show()


excel_writer.save()
