import ML
from sklearn.svm import SVC


def run():
    url = "../assets/iris.csv"
    names = ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]
    dataset = ML.load_dataset(url, names)
    # ML.show_some_dataset_info(dataset)
    # ML.show_univariate_plot(dataset)
    # ML.show_histogram(dataset)
    # ML.show_multivariate_plot(dataset)
    X_train, X_validation, Y_train, Y_validation = ML.split_out_validation_dataset(
        dataset
    )
    # ML.evaluate_models(X_train, Y_train)
    used_model = SVC(gamma="auto")
    ML.make_predictions_model(used_model, X_train, Y_train, X_validation, Y_validation)


if __name__ == "__main__":
    run()
