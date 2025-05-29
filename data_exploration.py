import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from typing import Tuple

import warnings
warnings.filterwarnings('ignore')

seed = 1234
np.random.seed(seed)

data = pd.read_csv('data/diabetic_data.csv')
X = data.drop('target_column', axis=1)
y = data['target_column']
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