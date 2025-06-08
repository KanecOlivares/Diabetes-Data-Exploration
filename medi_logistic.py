import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

RED = '\033[91m'
GREEN = '\033[92m'
BLUE = '\033[94m'
RESET = '\033[0m'

def save_fig(plt, directory: str, filename: str) -> None:
    if not os.path.isdir(directory):
        message = (
            f"{RED} {directory} does not exist in current path: {os.getcwd()}{RESET}.\n"
            f"Would you like to create {BLUE}{directory}{RESET}: Y/N: "
        )
        if input(message).lower() == "y":
            print(f"{GREEN}Made {directory}/{filename}{RESET}")
            os.makedirs(directory, exist_ok=True)
        else:
            print(f"{RED}Did not make directory—exiting...{RESET}")
            return
    plt.savefig(f"{directory}/{filename}")
    plt.close()

class MediLogistic:
    def __init__(self, data, target):
        self.seed = 1234
        self.data = data
        self.target = target  # Removed label noise injection
        self.X = self.compute_X()
        self.X_train, self.X_val, self.y_train, self.y_val = self.get_split(0.3, self.X, self.target)
        self.X_val, self.X_test, self.y_val, self.y_test = self.get_split(0.5, self.X_val, self.y_val)
        self.transform_and_scale()

    def model_creation(self, max_iter):
        return LogisticRegression(penalty="l1", solver="saga", multi_class="multinomial", max_iter=max_iter, n_jobs=-1)

    def compute_X(self):
        X = self.data.drop("readmitted", axis=1)
        features_to_drop = [
            "patient_nbr",
            "discharge_disposition_id",
            "admission_source_id",
            "payer_code",
            "medical_specialty",
        ]
        return self.data.drop(features_to_drop, axis=1)

    def get_split(self, split_percentage, X, y_target):
        return train_test_split(
            X,
            y_target,
            test_size=split_percentage,
            random_state=self.seed,
            stratify=y_target,
        )

    def transform_and_scale(self):
        scaler = StandardScaler().fit(self.X_train)
        self.X_train = scaler.transform(self.X_train)
        self.X_val = scaler.transform(self.X_val)
        self.X_test = scaler.transform(self.X_test)

    def train(self):
        self.model = self.model_creation(max_iter=50)
        self.model.fit(self.X_train, self.y_train)
        self.print_scores(self.model, self.X_train, self.y_train, "Training")
        self.print_scores(self.model, self.X_val, self.y_val, "Validation")
        self.print_scores(self.model, self.X_test, self.y_test, "Testing")

    def print_evals(self):
        self.plot_iter_learning_curve()

    def plot_iter_learning_curve(self):
        iter_range = np.linspace(1, 50, 10, dtype=int)
        train_accuracies = []
        val_accuracies = []
        test_accuracies = []

        X_full = self.X_train
        y_full = self.y_train

        for iter_count in iter_range:
            model = LogisticRegression(
                penalty="l1",
                solver="saga",
                multi_class="multinomial",
                max_iter=iter_count,
                n_jobs=-1
            )
            model.fit(X_full, y_full)
            train_acc = self.print_scores(model, X_full, y_full, f"Training (iter={iter_count})")
            val_acc = self.print_scores(model, self.X_val, self.y_val, f"Validation (iter={iter_count})")
            test_acc = self.print_scores(model, self.X_test, self.y_test, f"Testing (iter={iter_count})")
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            test_accuracies.append(test_acc)
            

        plt.plot(iter_range, train_accuracies, label="Training", marker='o')
        plt.plot(iter_range, val_accuracies, label="Validation", marker='s')
        plt.plot(iter_range, test_accuracies, label="Testing", marker='^')
        plt.title("Accuracy vs. Max Iterations")
        plt.xlabel("Max Iterations")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.grid(True)
        plt.legend()
        save_fig(plt, "logistic_figs", "logistic_iteration_curve.png")

    def print_scores(self, model, x, y, score_type: str = "NOT GIVEN"):
        y_pred = model.predict(x)
        accuracy = accuracy_score(y, y_pred)
        print(f"{GREEN}{score_type} Accuracy: {accuracy:.4f}{RESET}")
        return accuracy
