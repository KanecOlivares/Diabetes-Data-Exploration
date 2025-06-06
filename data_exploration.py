import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import sklearn
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from typing import Tuple

import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from medi_nn import MediNN

import warnings
warnings.filterwarnings('ignore')

RED = '\033[91m'
GREEN = '\033[92m'
BLUE = '\033[94m'
RESET = '\033[0m'

# 75 and 25 training vs testing
# Randomeness / Reproduable shouldnt matter too much

class GeneralError(Exception):
       """
       Just a class for any exception one wants to raise 
       """
       def __init__(self, message):
           super().__init__(message)

def error(msg):
    raise(GeneralError(msg))

# Data setup
def check_valid_keys(data, features_to_drop):
    print(data.columns)
    
    """
    ['encounter_id', 'patient_nbr', 'race', 'gender', 'age', 'weight',
        'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
        'time_in_hospital', 'payer_code', 'medical_specialty',
        'num_lab_procedures', 'num_procedures', 'num_medications',
        'number_outpatient', 'number_emergency', 'number_inpatient', 'diag_1',
        'diag_2', 'diag_3', 'number_diagnoses', 'max_glu_serum', 'A1Cresult',
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
        'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
        'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
        'tolazamide', 'examide', 'citoglipton', 'insulin',
        'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone',
        'metformin-pioglitazone', 'change', 'diabetesMed', 'readmitted']
    """

    for feature in features_to_drop:
        RED = '\033[31m'
        GREEN = '\033[32m'
        RESET = '\033[0m'
        if feature in data.columns:
            print(f"{GREEN}{feature} is recognized!!!{RESET}")
        else:
            print(f"{RED}{feature} is NOT recognized!!!{RESET}")


def setup():

    data = pd.read_csv('data/diabetic_data.csv')
    # Either too little info, or they have too many unique keys reduction of noise
    features_to_drop = ['weight', 'encounter_id', 'patient_nbr', 'readmitted', 'payer_code', 'medical_specialty', 
                        'diag_1', 'diag_2', 'diag_3', 'examide', 'citoglipton']
    
    X = data.drop(features_to_drop, axis=1)
    data['readmitted'] = data['readmitted'].map({
    'NO': 0,
    '>30': 1,
    '<30': 2
    })

    for col in data.select_dtypes(include=['object']).columns:
        data[col] = data[col].astype('category').cat.codes
    
    y_A1C = data['A1Cresult']
    y_readmitted = data['readmitted']
    y_med_spec = data['medical_specialty']
    y_change = data['change']
    # data = standardize_data(data)
    return data, X, y_A1C, y_readmitted, y_med_spec, y_change


# Exploration

def compute_range_instances(weight_column) -> dict:
    """
    Computes the amount of instances each range has. Will be used for a bar graph. 
    There is an error check to see if the weight is present in the weight_range hash map
    if not it will raise an error. Guarrenting there are no ranges that are not counted for
    """

    # Ranges used in the data set
    weight_range = {'?': 0, '[0-25)': 0, '[25-50)': 0, '[50-75)' : 0, '[50-75)': 0, '[75-100)': 0,
                    '[100-125)': 0, '[125-150)': 0, '[150-175)': 0, '[175-200)': 0, '>200': 0}


    for weight in weight_column:
        if weight not in weight_range:
            message = f'{weight} is not a key in weight_range inside function weight_exploration'
            raise error(message)
        else:
            weight_range[weight] += 1
    
    return weight_range

def plot_weight_bar(weight_range) -> None:
    key_list = list(weight_range.keys())
    value_list = list(weight_range.values())
    plt.figure(figsize=(10, 6))
    plt.bar(key_list, value_list)
    plt.savefig("weight_plot.png")
    plt.close()
    return

def weight_exploration(data) -> None:
    weight_column = data['weight']
    weight_range = compute_range_instances(weight_column)
    plot_weight_bar(weight_range)
    return

def standardize_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes the data using StandardScaler from sklearn. Assumes that the input DataFrame contains numeric features.
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return pd.DataFrame(scaled_data, columns=data.columns, index=data.index)

def get_numeric_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with only numeric features from the input DataFrame.
    """
    numeric_features = data.select_dtypes(include='number')
    return numeric_features

def plot_correlation_matrix(data: pd.DataFrame) -> None:
    """
    Plots the correlation matrix of the input DataFrame. Assumes that the DataFrame contains numeric features.
    """
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(30, 20)) # width
    fig.colorbar(ax.matshow(corr, cmap='coolwarm'))
    
    ticks = np.arange(len(corr.columns))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)

    for (i, j), val in np.ndenumerate(corr.values):
        ax.text(j, i, f"{val:.2f}", ha='center', va='center', color='black')
    
    plt.title('Correlation Matrix Heatmap')
    plt.savefig("correlation_matrix_heatmap.png")
    plt.close()
    
def plot_readmission_time_in_hospital(data: pd.DataFrame) -> None:
    crosstab = pd.crosstab(data['discharge_disposition_id'], data['readmitted'])

    # Optional: sort by disposition id
    crosstab = crosstab.sort_index()

    # Plot stacked bar chart
    crosstab.plot(kind='bar', stacked=True, figsize=(10,7), colormap='tab20')


    plt.title('Readmission Counts per Discharge Disposition ID')
    plt.xlabel('Discharge Disposition ID')
    plt.ylabel('Number of Patients')
    plt.legend(title='Readmission Status')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("discharge_dispo_readmission_counts.png")
    plt.close()

    GREEN = '\033[32m'
    RESET = '\033[0m' # Resets all formatting
    for feature in data.columns:
        if feature == 'readmitted':
            continue  # skip target column
        counts = pd.crosstab(data[feature], data['readmitted'])
        percentages = counts.div(counts.sum(axis=1), axis=0) * 100
        # print(percentages)
        # print(percentages.keys())
        # sorted_percentages = percentages.sort_values(by=2, ascending=True)
        pd.set_option('display.float_format', lambda x: f'{x:5.2f}%')  # format as %
        print(GREEN + f"\nReadmission Percentages per {feature}:\n" + RESET)
        print(percentages)
        # print(sorted_percentages)
        

    """
    Readmission Percentages per Discharge Disposition ID:
        readmitted                  <30    >30      NO
        discharge_disposition_id                      
        1                         9.30% 35.72%  54.98%
        2                        16.07% 31.39%  52.54%
        3                        14.66% 35.23%  50.11%
        4                        12.76% 34.11%  53.13%
        5                        20.86% 29.56%  49.58%
        6                        12.70% 41.56%  45.74%
        7                        14.45% 35.47%  50.08%
        8                        13.89% 35.19%  50.93%
        9                        42.86%  9.52%  47.62%
        10                        0.00% 66.67%  33.33%
        11                        0.00%  0.00% 100.00%
        12                       66.67%  0.00%  33.33%
        13                        4.76%  9.02%  86.22%
        14                        6.45%  1.88%  91.67%
        15                       44.44% 28.57%  26.98%
        16                        0.00% 54.55%  45.45%
        17                        0.00% 35.71%  64.29%
        18                       12.44% 27.82%  59.74%
        19                        0.00%  0.00% 100.00%
        20                        0.00%  0.00% 100.00%
        22                       27.70% 26.04%  46.26%
        23                        7.28% 34.95%  57.77%
        24                       14.58% 33.33%  52.08%
        25                        9.30% 38.02%  52.68%
        27                        0.00% 20.00%  80.00%
        28                       36.69% 24.46%  38.85%
    """

def evaluate(model, dataloader, criterion, device='cpu'):
    model.eval()  # eval mode
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)  # sum loss
            
            # Get predicted classes
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def create_dataloader(X, y, batch_size=64, shuffle=False):
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def plot_loss_curve(model):
    plt.plot(model.loss_curve_)
    plt.title("Loss per Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("neural_network_loss_curve.png")
    plt.close()

def print_evals(model, X_train, X_val, X_test, y_train, y_val, y_test):

    print_scores(model, X_train, y_train, "Training")
    print_scores(model, X_val, y_val, "Validation")
    print_scores(model, X_test, y_test, "Testing")

    # Printing Distrubutions
    print("Train classes:", np.bincount(y_train))
    print("Val classes:", np.bincount(y_val))

    #
    plot_loss_curve(model)


def main():
    seed = 1234
    np.random.seed(seed)
    data, _, y_A1C, y_readmitted, y_med_spec, y_change = setup()

    # uncomment these!!!
    # weight_exploration(data)
    # numerical_standard_data = get_numeric_features(data)
    # plot_correlation_matrix(numerical_standard_data)
    # plot_readmission_time_in_hospital(data)


    medi_nn = MediNN(data, y_readmitted)
    medi_nn.train()
    medi_nn.print_evals()


# CHECK THE PLOTSSSS FOR REGRESSION
# DO NEURAL NETWORKS

def print_scores(model, x, y, score_type: str = "NOT GIVEN"):
    y_pred = model.predict(x)
    accuracy = accuracy_score(y, y_pred)
    print(f"{GREEN}{score_type} Accuracy: {accuracy:.4f}{RESET}")

def nn_num_med():
    seed = 1234
    np.random.seed(seed)
    data, _, y_A1C, y_readmitted, y_med_spec, y_change = setup()

    # Dropping because this is the target
    X = data.drop('readmitted', axis=1)

    # Dropping additional features not dropped in setup due to fine tuning this model
    features_to_drop = ["patient_nbr", "discharge_disposition_id", "admission_source_id", "payer_code", "medical_specialty"]
    X = data.drop(features_to_drop, axis=1)

    # Setting up training split 70% training 30% for Test/Val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_readmitted, test_size=0.3, random_state=seed, stratify=y_readmitted
    )
    # 15% Test 15% Validation
    X_val, X_test, y_val, y_test = train_test_split(
    X_val, y_val, test_size=0.5, random_state=seed, stratify=y_val)

    mlp = MLPClassifier(hidden_layer_sizes= (10, 2),
        activation = "relu",
        solver = "sgd",
        alpha = 0.1,
        batch_size=32,
        learning_rate = 'adaptive',
        learning_rate_init = 0.0001,
        early_stopping=True,
        validation_fraction=0.25,
        max_iter = 100,
        n_iter_no_change = 20,
        verbose=True)
    
    scaler = StandardScaler().fit(X_train)  # Learn mean and std from training set
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    y_train.value_counts(normalize=True)
    y_val.value_counts(normalize=True)
    y_test.value_counts(normalize=True)

    mlp.fit(X_train, y_train)

    print_evals(mlp, X_train, X_val, X_test, y_train, y_val, y_test)



main()

# nn_num_med()
    

    
