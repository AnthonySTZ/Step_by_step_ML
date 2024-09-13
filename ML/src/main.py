import ML


def run():
    url = "../assets/iris.csv"
    names = ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]
    dataset = ML.load_dataset(url, names)
    print(dataset)


if __name__ == "__main__":
    run()
