import pandas as pd
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


def load_dataset(filename: str, names: list[str]) -> pd.DataFrame:
    """Loads a dataset from the given URL"""
    dataset = pd.read_csv(filename, names=names)
    return dataset


def show_some_dataset_info(dataset: pd.DataFrame) -> None:
    """Displays some information about the dataset"""
    print(f"Shape: \n {dataset.shape}")
    print(f"Head: \n {dataset.head(10)}")
    print(f"Description: \n {dataset.describe()}")
    print(f"Group by class: \n 
          {dataset.groupby('class').size()}")
