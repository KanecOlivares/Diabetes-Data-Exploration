
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

class mediForest():
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
        # self.transform_and_scale()
    
    def model_creation(self):
        return RandomForestClassifier(n_estimators=20, max_depth=15, random_state=self.seed)
    
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
    
    def train(self) -> None:
        return self.model.fit(self.X_train, self.y_train)
    

    def print_evals(self) -> None:
        self.print_scores(self.model, self.X_train, self.y_train, "Training")
        self.print_scores(self.model, self.X_val, self.y_val, "Validation")
        self.print_scores(self.model, self.X_test, self.y_test, "Testing")

        # Printing Distrubutions
        print("Train classes:", np.bincount(self.y_train))
        print("Val classes:", np.bincount(self.y_val))

        self.plot_depth_comparison("max_depth_scores")

        # Max Depth
        max_depths = [estimator.tree_.max_depth for estimator in self.model.estimators_]
        print("Maximum depth among all trees:", max(max_depths))
        # Min split
        print("min_samples_split:", self.model.min_samples_split)
        # Min samples
        print("min_sampel:", self.model.min_samples_leaf)

    def print_scores(self, model, x, y, score_type: str = "NOT GIVEN"):
        y_pred = model.predict(x)
        accuracy = accuracy_score(y, y_pred)
        print(f"{GREEN}{score_type} Accuracy: {accuracy:.4f}{RESET}")
    
    def get_score(self, x, y):
        y_pred = self.model.predict(x)
        return accuracy_score(y, y_pred)
        
        


    def plot_depth_comparison(self, filename):
        tr_scores, val_scores, test_scores = self.compute_scores()
        x = list(range(1, 21))
        plt.figure(figsize=(10, 6))
        plt.plot(x, tr_scores, label='Training Accuracy', color='blue')
        plt.plot(x, val_scores, label='Validation Accuracy', color='orange')
        plt.plot(x, test_scores, label='Testing Accuracy', color='green')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.title('Accuracy over Iterations')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        save_fig(plt, "tree_figs", f"{filename}.png")


    def compute_scores(self):

        training_score = []
        validation_score = []
        testing_score = []

        # Checking n_estimator scores
        ### REMEMBER IT PASSES BY REFRENCE NOT BY VALUE ###
        # self.n_estimator_score(training_score, validation_score, testing_score)

        # Max Depth Scores
        self.max_depth_score(training_score, validation_score, testing_score)

        return training_score, validation_score, testing_score
    
    def n_estimator_score(self, training_score, validation_score, testing_score):
        for i in range(1, 101):
            print(f"Finished with iteration: {i}")
            self.model = RandomForestClassifier(n_estimators=i, random_state=self.seed)
            self.train()
            training_score.append(self.get_score(self.X_train, self.y_train))
            validation_score.append(self.get_score(self.X_val, self.y_val))
            testing_score.append(self.get_score(self.X_test, self.y_test))
        

    def max_depth_score(self, training_score, validation_score, testing_score):
        for i in range(1, 21):
            print(f"Finished with iteration: {i}")
            self.model = RandomForestClassifier(n_estimators=20, random_state=self.seed, max_depth=i)
            self.train()
            training_score.append(self.get_score(self.X_train, self.y_train))
            validation_score.append(self.get_score(self.X_val, self.y_val))
            testing_score.append(self.get_score(self.X_test, self.y_test))
