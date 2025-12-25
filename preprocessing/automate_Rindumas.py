import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(path):
    return pd.read_csv(path)

def preprocessing(df):
    df = df.copy()

    # 1. Missing values
    df = df.dropna()

    # 2. Duplikat
    df = df.drop_duplicates()

    # 3. Drop ID
    df = df.drop(columns=['nameOrig', 'nameDest'])

    # 4. Encode type
    le = LabelEncoder()
    df['type'] = le.fit_transform(df['type'])

    # 5. Scaling numeric
    num_cols = [
        'step',
        'amount',
        'oldbalanceOrg',
        'newbalanceOrig',
        'oldbalanceDest',
        'newbalanceDest'
    ]

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # 6. Outlier detection (IQR)
    Q1 = df['amount'].quantile(0.25)
    Q3 = df['amount'].quantile(0.75)
    IQR = Q3 - Q1

    df = df[
        (df['amount'] >= Q1 - 1.5 * IQR) &
        (df['amount'] <= Q3 + 1.5 * IQR)
    ]

    # 7. Binning (0 = low, 1 = medium, 2 = high, 3 = very high)
    df['amount_bin'] = pd.qcut(df['amount'], q=4, labels=False)

    return df

def save_data(df, output_path):
    df.to_csv(output_path, index=False)

def main():
    input_path = "onlinepaymentfraud_dataset.csv"
    output_path = "preprocessing/onlinepaymentfraud_preprocessing.csv"

    df = load_data(input_path)
    df_clean = preprocessing(df)
    save_data(df_clean, output_path)

if __name__ == "__main__":
    main()