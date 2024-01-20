import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler



class Data:
    methods = {
        "log": lambda x: np.log(x),
        "sqrt": lambda x: np.sqrt(x),
        "shift": lambda x: x.shift(1),
        "diff_2": lambda x: x.diff(2),  # Second-order difference
        "diff_log": lambda x: np.log(x.diff(1)),
        "diff": lambda x: x.diff(1),
        "pct_change": lambda x: x.pct_change(1),
        "rolling_mean": lambda x: x.rolling(1).mean(),
        "rolling_std": lambda x: x.rolling(1).std(),
        # Square root transformation after differencing
        "power_0.5": lambda x: x.diff(1) ** 0.5,
        # Cube root transformation after differencing
        "power_0.33": lambda x: x.diff(1) ** (1 / 3)
    }

    def __init__(self, csv_path='/content/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv'):
        self.csv_path = csv_path

    def clean_data(self, df):
        df = df.dropna()
        df = df.reset_index(drop=True)
        return df

    def classify_data(self, df):
        price = df['Close']
        log_price = np.log(price)
        min_indice = np.argmin(log_price)
        max_indice = np.argmax(log_price)
        return min_indice, max_indice

    def add_features(self, data):
        # Volatility Measures
        data['Price_Range'] = data['High'] - data['Low']
        data['Price_Std_Dev'] = data['Close'].rolling(window=30).std()

        # Relative Changes
        data['Close_Pct_Change'] = data['Close'].pct_change()

        # Technical Indicators (Simple Moving Average)
        data['SMA_7'] = data['Close'].rolling(window=7).mean()
        data['SMA_30'] = data['Close'].rolling(window=30).mean()

        # Volume-Related Ratios
        data['Volume_Price_Ratio'] = data['Volume_(Currency)'] / data['Close']
        return data

    def Update_Y(self, data):
        data['Close_diff'] = data['Close'].diff(1)
        data = data.drop(columns=['Close'])
        data = data.dropna()
        data = data.reset_index(drop=True)
        return data

    def split_data(self, df, target_column='Close', train_size=0.7, val_size=0.2):
        """Splits the data into training, validation, and test sets.

        Args:
            df: The data to split.
            target_column: The name of the target column.
            train_size: The proportion of the data to use for training.
            val_size: The proportion of the data to use for validation.

        Returns:
            X_train, Y_train, X_val, Y_val, X_test, Y_test: The training, validation, and test sets.
        """

        n = len(df)
        train_end = int(n * train_size)
        val_end = int(n * (train_size + val_size))

        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]

        # Splitting into X and Y
        train_X, train_Y = train_df.drop(
            columns=[target_column]), train_df[target_column].diff(1).dropna()
        val_X, val_Y = val_df.drop(
            columns=[target_column]), val_df[target_column].diff(1).dropna()
        test_X, test_Y = test_df.drop(
            columns=[target_column]), test_df[target_column].diff(1).dropna()

        # Adjust X datasets to match the size of Y datasets
        train_X = train_X.iloc[1:]
        val_X = val_X.iloc[1:]
        test_X = test_X.iloc[1:]

        return train_X, train_Y, val_X, val_Y, test_X, test_Y

    def btc_trend_show_all(self, data):
        num_cols = len(data.columns)
        num_rows = (num_cols + 1) // 2  # Calculate the number of rows needed

        fig, axes = plt.subplots(num_rows, 2, figsize=(15, 4*num_rows))

        for i, col in enumerate(data.columns):
            row = i // 2
            col_num = i % 2
            axes[row, col_num].plot(data[col])
            axes[row, col_num].set_xlabel('Days')
            axes[row, col_num].set_ylabel(col)
            axes[row, col_num].set_title(f'Bitcoin {col} over time')

        # Hide empty subplots if any
        for i in range(num_cols, num_rows * 2):
            row = i // 2
            col_num = i % 2
            fig.delaxes(axes[row, col_num])

        plt.tight_layout()
        plt.show()

    def data_slices(self, data, up=True):
        min_indice, max_indice = self.classify_data(data)
        data_up = data.iloc[min_indice:max_indice]
        data_down = data.iloc[max_indice:]
        if up:
            return data_up
        else:
            return data_down
    def normalize_data(self, data):
        """Normalize data using Min-Max Scaling."""
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        return pd.DataFrame(scaled_data, columns=data.columns)

    def full_process(self, up=True):
        df = pd.read_csv(self.csv_path)
        data = df.copy()
        data = data.drop(columns=['Timestamp'])
        data = self.clean_data(data)
        
        if up:
            data = self.data_slices(data, up=True)
            data = self.add_features(data)
            data = self.normalize_data(data)  # Normalize the data here
            price = data['Close']
            data = self.data_slices(data, up=True)
            X_train, Y_train, X_val, Y_val, X_test, Y_test = self.split_data(data)
            return X_train, Y_train, X_val, Y_val, X_test, Y_test, data, price
        else:
            data = self.data_slices(data, up=False)
            data = self.add_features(data)
            data = self.normalize_data(data)  # Normalize the data here
            price = data['Close']
            data = self.data_slices(data, up=False)
            X_train, Y_train, X_val, Y_val, X_test, Y_test = self.split_data(data)
            return X_train, Y_train, X_val, Y_val, X_test, Y_test, data, price
