import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

RED = '\033[91m'
GREEN = '\033[92m'
BLUE = '\033[94m'
RESET = '\033[0m'

def save_fig(plt, directory: str, filename) -> None:
    """
    Saves a given plt in to the machine's directory and filename. If directory 
    does not exist it will double check if the directory should be made then make 
    or not make it 

    Arguments:
    plt: plot object to plot and close
    """
    if not os.path.isdir(directory):
        message = f'{RED} {directory} does not exist in current path: {os.getcwd()}{RESET}.\n'
        message += f"Would you like to create {BLUE} {directory} {RESET}: Y/N: "
        if input(message).lower() == 'y':
            print(f'{GREEN}Made {directory}/{filename}{RESET}')
            os.makedirs(directory, exist_ok=True)
        else:
            print(f'{RED}Did not make directory exiting...{RESET}')
            return
            
    plt.savefig(f"{directory}/{filename}")
    plt.close()


class MediKNN():
    def __init__(self, data, target, k):
        """
        Setups up model, training, validation, testing data. 
        data: the db
        target: column we are predicting 
        """
        self.seed = 1234
        self.data = data
        self.target = target
        self.k = k  # Number of neighbors for KNN
        self.model = self.model_creation()
        self.X = self.compute_X()
        # 70% Training 30% Validation/Testing
        self.X_train, self.X_val, self.y_train, self.y_val = self.get_split(0.3, self.X, self.target)
        # 15 % Validation 15% Testing
        self.X_val, self.X_test, self.y_val, self.y_test = self.get_split(0.5, self.X_val, self.y_val)
        self.transform_and_scale()

    def model_creation(self):    
        """
        Creates the model and returns it. Mess with the actual hyperparms here
        """
        # KNN Classifier with 5 neighbors
        return KNeighborsClassifier(n_neighbors=self.k)

    def compute_X(self):
        """
        In this case we are dropping readmitted as it is the target. The other features
        were too noisy i.e too many unique values or had little to no information. 
        """
        X = self.data.drop('readmitted', axis=1)
        features_to_drop = ["patient_nbr", "discharge_disposition_id", "admission_source_id", 
                            "payer_code", "medical_specialty"]
        X = self.data.drop(features_to_drop, axis=1)
        return X
    
    def get_split(self, split_percentage, X, y_target):
        """
        Gives split of one_pair and second pair. X and Y. First pair is the (1 - split_percentage)
        Other is the split_percentage. i.e Testing/Validation
        """
        return train_test_split(
            X, y_target, test_size=split_percentage, random_state=self.seed, stratify=y_target
        )
    
    def transform_and_scale(self) -> None:
        """
        Transforms and scales the input and output 
        """
        scaler = StandardScaler().fit(self.X_train)  # Learn mean and std from training set
        self.X_train = scaler.transform(self.X_train)
        self.X_val = scaler.transform(self.X_val)
        self.X_test = scaler.transform(self.X_test)

        self.y_train.value_counts(normalize=True)
        self.y_val.value_counts(normalize=True)
        self.y_test.value_counts(normalize=True)

    def train(self) -> None:
        self.model.fit(self.X_train, self.y_train) 

    def return_evals(self) -> tuple:
        return_tuple = (
        self.return_scores(self.model, self.X_train, self.y_train, "Training"),
        self.return_scores(self.model, self.X_val, self.y_val, "Validation"),
        self.return_scores(self.model, self.X_test, self.y_test, "Testing")
        )

        # Printing Distrubutions
        print("Train classes:", np.bincount(self.y_train))
        print("Val classes:", np.bincount(self.y_val))

        return return_tuple


    def plot_loss_curve(self, k_vals, train_error, val_error, test_error) -> None:
        figure, axes = plt.subplots(1, figsize=(6,6))
        axes.plot(k_vals, train_error, color='red', label='Train Error')
        axes.plot(k_vals, val_error, color='blue', label='Validation Error')
        axes.plot(k_vals, test_error, color='green', label='Test Error')
        axes.set_xlabel('K Value')
        axes.set_ylabel('Error Rate')
        axes.set_title('Error Rate vs K Value')
        axes.legend()
        save_fig(plt, "knn_figs", "knn_error_rate_vs_k_value.png")


    def return_scores(self, model, x, y, score_type: str = "NOT GIVEN") -> float:
        y_pred = model.predict(x)
        accuracy = accuracy_score(y, y_pred)
        print(f"{GREEN}{score_type} Accuracy: {accuracy:.4f}{RESET}")
        return accuracy