import numpy as np
import pandas as pd


def build_1_to_1_data(size=1000):
    np.random.seed(13492)
    x = np.random.randint(24, size=size)
    p = 0.5 + (0.45 * np.sin(-np.pi + (x * np.pi / 12)))
    threshold = np.random.uniform(size=size)
    y = (threshold < p).astype(int)
    return pd.DataFrame({"x": x, "y": y})


def build_1_to_24_data(size=1000):
    np.random.seed(13492)
    x = np.random.randint(24, size=size)
    p = 0.5 + (0.45 * np.sin(-np.pi + (x * np.pi / 12)))
    threshold = np.random.uniform(size=size)
    df = pd.DataFrame({"x": x})
    df = pd.get_dummies(df["x"], prefix="y")
    df *= (threshold < p).reshape(-1, 1)
    df["x"] = x
    return df


def build_24_to_1_data(size=1000):
    df = build_1_to_1_data(size)
    df = pd.get_dummies(df, columns=["x"])
    return df


def build_24_to_24_data(size=1000):
    df = build_1_to_24_data(size)
    df = pd.get_dummies(df, columns=["x"])
    return df
