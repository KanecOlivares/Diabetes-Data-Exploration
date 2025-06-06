
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score



RED = '\033[91m'
GREEN = '\033[92m'
BLUE = '\033[94m'
RESET = '\033[0m'

def save_fig(plt, directory: str, filename):
    os.makedirs(directory, exist_ok=True)
    plt.savefig(f"{directory}/{filename}")
    plt.close()

class MediNN():
    def __init__(self, data, target):
        self.seed = 1234
        self.data = data
        self.target = target
        self.model = self.model_creation()
        self.X = self.compute_X()
        # 70% Training 30% Valdation/Testing
        self.X_train, self.X_val, self.y_train, self.y_val = self.get_split(0.3, self.X, self.target)
        # 15 % Validation 15% Testing
        self.X_val, self.X_test, self.y_val, self.y_test = self.get_split(0.5, self.X_val, self.y_val)
        self.transform_and_scale()

    def model_creation(self):
        return MLPClassifier(hidden_layer_sizes= (10, 2),
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
    
    def compute_X(self):
        X = self.data.drop('readmitted', axis=1)
        features_to_drop = ["patient_nbr", "discharge_disposition_id", "admission_source_id", 
                            "payer_code", "medical_specialty"]
        X = self.data.drop(features_to_drop, axis=1)
        return X

    def get_split(self, split_percentage, X, y_target):
        return train_test_split(
            X, y_target, test_size=split_percentage, random_state=self.seed, stratify=y_target
        )
    
    def transform_and_scale(self) -> None:
        scaler = StandardScaler().fit(self.X_train)  # Learn mean and std from training set
        self.X_train = scaler.transform(self.X_train)
        self.X_val = scaler.transform(self.X_val)
        self.X_test = scaler.transform(self.X_test)

        self.y_train.value_counts(normalize=True)
        self.y_val.value_counts(normalize=True)
        self.y_test.value_counts(normalize=True)

    def train(self) -> None:
        self.model.fit(self.X_train, self.y_train)

    def plot_loss_curve(self) -> None:
        plt.plot(self.model.loss_curve_)
        plt.title("Loss per Iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.grid(True)
        save_fig(plt, "nn_figs", "neural_network_loss_curve.png")

    def print_evals(self) -> None:
        self.print_scores(self.model, self.X_train, self.y_train, "Training")
        self.print_scores(self.model, self.X_val, self.y_val, "Validation")
        self.print_scores(self.model, self.X_test, self.y_test, "Testing")

        # Printing Distrubutions
        print("Train classes:", np.bincount(self.y_train))
        print("Val classes:", np.bincount(self.y_val))

        # Saves locally a grpah of the loss curve
        self.plot_loss_curve()

    def print_scores(self, model, x, y, score_type: str = "NOT GIVEN"):
        y_pred = model.predict(x)
        accuracy = accuracy_score(y, y_pred)
        print(f"{GREEN}{score_type} Accuracy: {accuracy:.4f}{RESET}")