# Imports
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Database class
class ClinicalDatabase:
    def __init__(self):
        dataset = load_breast_cancer(as_frame=True)
        self.data = dataset.frame.drop(columns=["target"])
        self.labels = dataset.frame["target"]
        self.feature_names = list(self.data.columns)

    def get_features(self, feature_list):
        return self.data[feature_list]

    def get_labels(self):
        return self.labels

# *** Model D - Logistic Regression ***

# Untrained model
class ModelD_LR:
    def __init__(self, database: ClinicalDatabase):
        self.model_type = "LogisticRegression"
        self.database = database
        self.trained_model = None

    def build_model(self):
        X = self.database.get_features(self.database.feature_names)
        y = self.database.get_labels()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model_d = LogisticRegression(max_iter=2000, solver='lbfgs')
        model_d.fit(X_train, y_train)
        print("ModelD_LR successfully trained.")
        self.trained_model = TrainedModelD(model_d, X_train, y_train, list(X_train.columns))
        return self.trained_model

# Trained Model
class TrainedModelD:
    def __init__(self, model_d, X_train, y_train, feature_names):
        self.model = model_d
        self.X_train = X_train
        self.y_train = y_train
        self.feature_names = feature_names
        self.explainer = CeterisExplainerD(self)
        print("TrainedModelD instantiated.")

    def predict(self, X):
        return self.model.predict(X)

# Ceteris Paribus Explainer
class CeterisExplainerD:
    def __init__(self, trained_model_d):
        self.trained_model = trained_model_d
        print("CeterisExplainerD instantiated and connected to TrainedModelD.")

    def explain_instance(self, instance_id=0, feature_name="mean radius"):
        import dalex as dx
        import matplotlib.pyplot as plt
        model_d = self.trained_model.model
        X_train = self.trained_model.X_train
        y_train = self.trained_model.y_train
        explainer = dx.Explainer(model_d, X_train, y_train, label="Logistic Regression (Clinical)")
        instance = X_train.iloc[[instance_id]]
        print(f"Generating Ceteris Paribus plot for instance {instance_id}, feature '{feature_name}'")
        cp_profile = explainer.predict_profile(instance, variables=[feature_name])
        ax = cp_profile.plot() 
        plt.title(f"Ceteris Paribus Plot for '{feature_name}' (instance {instance_id})")
        plt.show()
        return cp_profile

# *** Main ***

if __name__ == "__main__":
    db = ClinicalDatabase()

    model_d = ModelD_LR(db)
    trained_model_d = model_d.build_model()
    sample_instance_d = trained_model_d.X_train.iloc[0]
    print("Selected instance:")
    print(sample_instance_d)

    print("Instance real target:")
    print(trained_model_d.y_train.iloc[0])

    print("Model prediction for the instance:")
    print(trained_model_d.predict(sample_instance_d.to_frame().T))

    print("Model Accuracy:")
    print(trained_model_d.model.score(trained_model_d.X_train, trained_model_d.y_train))

    trained_model_d.explainer.explain_instance(
        instance_id=0,
        feature_name="mean radius"
    )