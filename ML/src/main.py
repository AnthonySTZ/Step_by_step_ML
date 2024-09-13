import ML


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


if __name__ == "__main__":
    run()
