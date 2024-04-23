# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 19:25:08 2023

@author: Patrycja Czemerych
"""

import pandas as pd
import csv
from sklearn.feature_selection import mutual_info_regression
from skfeature.function.similarity_based import fisher_score
import numpy as np

DIRECTORY = 'csv_T2'
DIRECTORY2 = 'T2_features_txt'
csv_files = ['output_D.csv', 'output_S.csv', 'output_M.csv', 'output_N.csv']

for TYPE_NORM in csv_files:
    file_path = f'C:/Users/HP/Desktop/{DIRECTORY}/{TYPE_NORM}'

    # Extract normalization type (D, S, M, or N) from the file name
    letter = TYPE_NORM.split('_')[-1].split('.')[0]

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

    # Convert lists to NumPy arrays
    br_features = np.array(br_features)
    gr_features = np.array(gr_features)

    # Combine features and ages into a DataFrame for bone region (br)
    train_data_br = pd.DataFrame(data=br_features, columns=header[:-1])
    train_data_br['Category'] = br_ages

    # Calculate correlation between features and age for bone region (br)
    correlation_matrix_br = train_data_br.corr()['Category'].abs()
    selected_features_corr_br = correlation_matrix_br.nlargest(11).index[1:]

    # Calculate Mutual Information between features and age for bone region (br)
    mi_scores_br = mutual_info_regression(br_features, br_ages)
    selected_features_mi_br = pd.Series(mi_scores_br, index=header[:-1]).nlargest(11).index

    # Perform Fisher Score for bone region (br)
    fisher_scores_br = fisher_score.fisher_score(br_features, br_ages)
    selected_features_fisher_br = pd.Series(fisher_scores_br, index=header[:-1]).nlargest(11).index

    # Print selected features for bone region (br)
    print(f"For bone region ({letter}):\n")
    output_file_path_corr_br = f'C:/Users/HP/Desktop/{DIRECTORY2}/selected_features_{letter}_corr_br.txt'
    print("Top 10 selected features - Correlation:")
    with open(output_file_path_corr_br, 'w') as output_file_br:
        for feature_corr_br in selected_features_corr_br[:10]:
            print(feature_corr_br)
            output_file_br.write(feature_corr_br + '\n')

    output_file_path_mi_br = f'C:/Users/HP/Desktop/{DIRECTORY2}/selected_features_{letter}_mi_br.txt'
    print("\nTop 10 selected features - Mutual Information:")
    with open(output_file_path_mi_br, 'w') as output_file_br:
        for feature_mi_br in selected_features_mi_br[:10]:
            print(feature_mi_br)
            output_file_br.write(feature_mi_br + '\n')

    output_file_path_fi_br = f'C:/Users/HP/Desktop/{DIRECTORY2}/selected_features_{letter}_fi_br.txt'
    print("\nTop 10 selected features - Fisher Score:")
    with open(output_file_path_fi_br, 'w') as output_file_br:
        for feature_fi_br in selected_features_fisher_br[:10]:
            print(feature_fi_br)
            output_file_br.write(feature_fi_br + '\n')

    # Combine features and ages into a DataFrame for growth region (gr)
    train_data_gr = pd.DataFrame(data=gr_features, columns=header[:-1])
    train_data_gr['Category'] = gr_ages

    # Calculate correlation between features and age for growth region (gr)
    correlation_matrix_gr = train_data_gr.corr()['Category'].abs()
    selected_features_corr_gr = correlation_matrix_gr.nlargest(11).index[1:]

    # Calculate Mutual Information between features and age for growth region (gr)
    mi_scores_gr = mutual_info_regression(gr_features, gr_ages)
    selected_features_mi_gr = pd.Series(mi_scores_gr, index=header[:-1]).nlargest(11).index

    # Perform Fisher Score for growth region (gr)
    fisher_scores_gr = fisher_score.fisher_score(gr_features, gr_ages)
    selected_features_fisher_gr = pd.Series(fisher_scores_gr, index=header[:-1]).nlargest(11).index

    # Print selected features for growth region (gr)
    print("\n\nFor growth region ({letter}):\n")
    output_file_path_corr_gr = f'C:/Users/HP/Desktop/{DIRECTORY2}/selected_features_{letter}_corr_gr.txt'
    print("Top 10 selected features - Correlation:")
    with open(output_file_path_corr_gr, 'w') as output_file_gr:
        for feature_corr_gr in selected_features_corr_gr[:10]:
            print(feature_corr_gr)
            output_file_gr.write(feature_corr_gr + '\n')

    output_file_path_mi_gr = f'C:/Users/HP/Desktop/{DIRECTORY2}/selected_features_{letter}_mi_gr.txt'
    print("\nTop 10 selected features - Mutual Information:")
    with open(output_file_path_mi_gr, 'w') as output_file_gr:
        for feature_mi_gr in selected_features_mi_gr[:10]:
            print(feature_mi_gr)
            output_file_gr.write(feature_mi_gr + '\n')

    output_file_path_fi_gr = f'C:/Users/HP/Desktop/{DIRECTORY2}/selected_features_{letter}_fi_gr.txt'
    print("\nTop 10 selected features - Fisher Score:")
    with open(output_file_path_fi_gr, 'w') as output_file_gr:
        for feature_fi_gr in selected_features_fisher_gr[:10]:
            print(feature_fi_gr)
            output_file_gr.write(feature_fi_gr + '\n')
