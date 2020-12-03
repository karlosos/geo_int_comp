import pandas as pd


def load_data():
    df = pd.read_csv("./data/wraki_utm.txt", delimiter=" ", header=None)
    df = df.drop(df.columns[[0, 1, 3]], axis=1)
    df.columns = ["X", "Y", "Z"]
    return df


def main():
    df = load_data()
    print(df.head())


if __name__ == "__main__":
    main()
