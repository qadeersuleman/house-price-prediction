from pathlib import Path
import joblib
import pandas as pd


def predict(example_row: dict):
    model_path = Path(__file__).resolve().parent / 'model.pkl'
    model = joblib.load(model_path)
    df = pd.DataFrame([example_row])
    return model.predict(df)[0]


if __name__ == '__main__':
    example = {
        'LotArea': 8450,
        'OverallQual': 7,
        'YearBuilt': 2003,
        'TotalBsmtSF': 856,
        'GrLivArea': 1710,
        'FullBath': 2,
        'BedroomAbvGr': 3
    }
    print(predict(example))