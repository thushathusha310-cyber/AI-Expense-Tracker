import pandas as pd
from sklearn.linear_model import LinearRegression


def predict_next_expense(df):

    if df is None or len(df) < 2:
        raise ValueError("Not enough expenses")

    data = df.copy()

    data["Amount"] = pd.to_numeric(
        data["Amount"],
        errors="coerce"
    )

    data = data.dropna(
        subset=["Amount"]
    )

    if len(data) < 2:
        raise ValueError("Not enough valid expenses")

    data["Transaction_Number"] = range(
        1,
        len(data) + 1
    )

    X = data[["Transaction_Number"]]
    y = data["Amount"]

    model = LinearRegression()

    model.fit(X, y)

    next_number = [[len(data) + 1]]

    prediction = model.predict(
        next_number
    )[0]

    return max(float(prediction), 0)