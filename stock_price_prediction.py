import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Step 1: Define the stock ticker and time period
ticker = "AAPL"  # Example: Apple Inc.
print(f"Fetching data for {ticker}...")  # Indicate that fetching is in progress
data = yf.download(ticker, start="2015-01-01", end="2024-01-11")

# Step 2: Check if data was fetched
if data.empty:
    print("No data fetched for the given ticker.")
else:
    print("Data fetched successfully!")  # Indicate success

    # Step 3: Visualize the closing price
    plt.figure(figsize=(12, 6))
    plt.plot(data['Close'], label='Close Price')
    plt.title('Apple Inc. (AAPL) Closing Price')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()

    # Step 4: Feature Engineering
    data['SMA_20'] = data['Close'].rolling(window=20).mean()  # 20-day simple moving average
    data['Daily_Return'] = data['Close'].pct_change()  # Daily returns
    data.dropna(inplace=True)  # Drop rows with NaN values

    # Step 5: Prepare data for modeling
    # Define features (X) and labels (y)
    X = data[['Open', 'High', 'Low', 'Volume', 'SMA_20']]
    y = data['Close']

    # Step 6: Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 7: Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Step 8: Make predictions
    predictions = model.predict(X_test)

    # Step 9: Evaluate model performance
    mae = mean_absolute_error(y_test, predictions)
    print(f'Mean Absolute Error: {mae}')

    # Step 10: Visualize predictions
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test, label='Actual Price')
    plt.plot(y_test.index, predictions, label='Predicted Price', linestyle='--')
    plt.title('AAPL Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()
