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
    print(f"Group by class: \n {dataset.groupby('class').size()}")


def show_univariate_plot(dataset: pd.DataFrame) -> None:
    """Plot Univariate Dataset"""
    dataset.plot(kind="box", subplots=True, layout=(2, 2), sharex=False, sharey=False)
    plt.show()


def show_histogram(dataset: pd.DataFrame) -> None:
    """Plot Histogram"""
    dataset.hist()
    plt.show()


def show_multivariate_plot(dataset: pd.DataFrame) -> None:
    """Plot Multivariate Dataset"""
    scatter_matrix(dataset)
    plt.show()


def split_out_validation_dataset(dataset: pd.DataFrame) -> tuple:

    array = dataset.values
    x = array[:, 0:4]  # Get all data but just the 0 to 3 dimensions
    y = array[:, 4]  # Get all data but just the 4th dimension
    X_train, X_validation, Y_train, Y_validation = train_test_split(
        x, y, test_size=0.20, random_state=1
    )

    return X_train, X_validation, Y_train, Y_validation


def evaluate_models(X_train, Y_train) -> None:
    # Spot Check Algorithms
    models = []
    models.append(("LR", LogisticRegression(solver="liblinear")))
    models.append(("LDA", LinearDiscriminantAnalysis()))
    models.append(("KNN", KNeighborsClassifier()))
    models.append(("CART", DecisionTreeClassifier()))
    models.append(("NB", GaussianNB()))
    models.append(("SVM", SVC(gamma="auto")))

    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        cv_results = cross_val_score(
            model, X_train, Y_train, cv=kfold, scoring="accuracy"
        )
        results.append(cv_results)
        names.append(name)
        print(f"{name}: {cv_results.mean()} ({cv_results.std()})")

    # plt.boxplot(results, labels=names)
    # plt.title("Algorithm Comparison")
    # plt.show()


def make_predictions_model(model, X_train, Y_train, X_validation, Y_validation) -> None:

    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)

    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))
