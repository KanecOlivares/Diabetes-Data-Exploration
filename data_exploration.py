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
from typing import Tuple

import warnings
warnings.filterwarnings('ignore')

# 75 and 25 training vs testing
# Randomeness / Reproduable shouldnt matter too much

class GeneralError(Exception):
       def __init__(self, message):
           super().__init__(message)

def error(msg):
    raise(GeneralError(msg))

"""
Columns to drop: 
    Weight: not enough 
        Run function to figure out how many (percentage wise has unknown weight )
        Bar graph: Show percentage unknown, show percentage of certain range, and show it

    encounter_id: Unique identifier of an encounter
    paitent_nbr: Unique identifier of a paitent

Columns to check out  
    discharge_disposition_id: Integer identifier corresponding to 29 distinct values, for example, discharged to home, expired, and not available
        input

    time_in_hospital: Integer number of days between admission and discharge
        input

    admission_source_id: Integer identifier corresponding to 21 distinct values, for example, physician referral, emergency room, and transfer 
    from a hospital

    num_lab_procedures: Number of lab tests performed during the encounter

    num_procedures: Number of procedures (other than lab tests) performed during the encounter

    num_medications: Number of distinct generic names administered during the encounter

    number_outpatient: Number of outpatient visits of the patient in the year preceding the encounter

    number_emergency: Number of emergency visits of the patient in the year preceding the encounter

    number_inpatient: Number of inpatient visits of the patient in the year preceding the encounter

Output

    A1Cresult: 	Indicates the range of the result or if the test was not taken. Values: >8 if the result was greater than 8%, >7 if the result
    was greater than 7% but less than 8%, normal if the result was less than 7%, and none if not measured

    Readmitted:

    medical_speciality:

    Change: Indicates if there was a change in diabetic medications (either dosage or generic name). Values: change and no change


Note: The rest of the columns from 23 to 49 are features of different drugs the paitent was on or persctibed
will leave it in there maybe there might be a correlation. Column 50 is readmission. 

Accuracy: 
    80 - 85% on validation


Intresting to check out:

    payer_code: Integer identifier corresponding to 23 distinct values, for example, Blue Cross/Blue Shield, Medicare, and self-pay

    medical_specialty: Integer identifier of a specialty of the admitting physician, corresponding to 84 distinct values, for example, cardiology,
    internal medicine, family/general practice, and surgeon

Citation: 
Beata Strack, Jonathan P. DeShazo, Chris Gennings, Juan L. Olmo, Sebastian Ventura, Krzysztof J. Cios, and John N. Clore, “Impact of HbA1c Measurement
on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records,” BioMed Research International, vol. 2014, Article ID 781670, 11
pages, 2014.
"""
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
    features_to_drop = ['weight', 'encounter_id', 'patient_nbr']
    # check_valid_keys(data, features_to_drop) # Use for debugging
    X = data.drop(features_to_drop, axis=1)
    y_A1C = data['A1Cresult']
    y_readmitted = data['readmitted']
    y_med_spec = data['medical_specialty']
    y_change = data['change']
    

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
    plt.bar(key_list, value_list)
    plt.show()
    return

def weight_exploration(data) -> None:
    weight_column = data['weight']
    weight_range = compute_range_instances(weight_column)
    plot_weight_bar(weight_range)
    return



def main():
    seed = 1234
    np.random.seed(seed)
    data, X, y_A1C, y_readmitted, y_med_spec, y_change = setup()
    weight_exploration(data)

main()
    

    
