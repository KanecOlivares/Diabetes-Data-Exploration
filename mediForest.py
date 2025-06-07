
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

RED = '\033[91m'
GREEN = '\033[92m'
BLUE = '\033[94m'
RESET = '\033[0m'

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
        return RandomForestClassifier(n_estimators=5, random_state=self.seed)
    
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