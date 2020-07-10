# main.py - contains the driver code for predicting stock closing prices 24 hours later.
# based on: https://medium.com/@randerson112358/stock-price-prediction-using-python-machine-learning-e82a039ac2bb
# Most of core code came from above source. This program adapts it into short term predictions and also puts it inside
# a wrapper program to interact with, with the addition of different modes and multi-stock prediction.

# ---------------------------------------------------------------------------------------------------------------------

# Import statements
import multiprocessing as mp
import time
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random
import matplotlib.backends.backend_pdf


# ---------------------------------------------------------------------------------------------------------------------

# parameters: stock (for i mode, which stock(s) to look at), save_graphs (True for saving, False for not)
def predict_next_day_close(stock, save_graphs):
    todays_date = datetime.date(datetime.now())
    yesterdays_date = datetime.strftime(datetime.now() - timedelta(1), '%Y-%m-%d')
    df = web.DataReader(stock, data_source='yahoo', start='2012-01-01', end=todays_date)
    data = df.filter(['Close'])
    dataset = data.values
    training_data_len = math.ceil(len(dataset) * .8)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Create the scaled training data set
    train_data = scaled_data[0:training_data_len, :]

    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])

    # Convert x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data into the shape accepted by the LSTM
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # actual mode creation
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    # Test data set
    test_data = scaled_data[training_data_len - 60:, :]

    # Create the x_test and y_test data sets
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    # Convert x_test to a numpy array
    x_test = np.array(x_test)

    # Reshape the data into the shape accepted by the LSTM
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Getting the models predicted price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)  # Undo scaling

    # Calculate/Get the value of RMSE and normalize
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2))) / np.mean(y_test)

    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions

    # creates and saved grpahs if wanted
    if (save_graphs):
        plt.figure(figsize=(16, 8))
        plt.title(stock + " Prediction")
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price USD ($)', fontsize=18)
        plt.plot(train['Close'])
        plt.plot(valid[['Close', 'Predictions']])
        plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
        # plt.show()

    # Get the quote
    quote = web.DataReader(stock, data_source='yahoo', start='2012-01-01', end=todays_date)

    # Create a new dataframe
    new_df = quote.filter(['Close'])

    # Get the last 60 day Closing price
    last_60_days = new_df[-60:].values

    # Scale the data to be values between 0 and 1
    last_60_days_scaled = scaler.transform(last_60_days)

    # Create an empty list
    X_test = []

    # Append the past 365 days
    X_test.append(last_60_days_scaled)

    # Convert the X_test data set to a numpy array
    X_test = np.array(X_test)

    # Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Get the predicted scaled price
    pred_price = model.predict(X_test)

    # undo the scaling
    pred_close = scaler.inverse_transform(pred_price)

    old_close = df['Close'].iloc[-1]
    old_close_date = df.axes[0].tolist()[-1]
    # print("Old Close Date", old_close_date)

    # creates info to view and append to DataFrame
    pred_profit = pred_close[0][0] - old_close
    pred_mvmt_perc = round((pred_profit / old_close) * 100, 4)
    # print("Old Close(Today): ", old_close)
    # print("Predicted close(Tomorrow): ", pred_close[0][0])
    # print("Predicted Profit: ", pred_profit)
    # print("Model NRMSE: ", rmse)
    # print("Predicted Movmement Percent: ", pred_mvmt_perc)

    return stock, old_close, round(pred_close[0][0], 3), round(pred_profit, 3), rmse, pred_mvmt_perc, plt


# ---------------------------------------------------------------------------------------------------------------------
# Main method.
# Parameters: mode(user mode), how_many(for r mode), and stock(for i mode)
# calls the predict method appropriatly for each mode.

def main(mode, how_many, stock):
    df_sp500 = pd.read_csv("sp500.csv")

    stock_list = df_sp500['Symbol'].tolist()
    randoms = []

    if mode.lower() == "one stock":
        for each_stock in stock:
            randoms.append(each_stock)
        p = mp.Pool(40)
    elif mode.lower() == "run":
        for i in range(int(how_many)):
            stocks_random = random.choice(stock_list)
            randoms.append(stocks_random)

    old_closes = []
    new_closes = []
    pred_profits = []
    pred_movement_percents = []
    rmses = []
    i = 1
    plots = []

    for each_stock in randoms:
        print("Predicting: ", each_stock, "(Stock Number: ", i, ")")
        # Boolean Value Controls graph: True displays predicted/actual graph, False does not
        stock, old_close, pred_close, pred_profit, rmse, pred_mvmt_perc, graph = predict_next_day_close(each_stock,
                                                                                                        True)
        old_closes.append(old_close)
        new_closes.append(pred_close)
        pred_profits.append(pred_profit)
        rmses.append(rmse)
        pred_movement_percents.append(pred_mvmt_perc)
        plots.append(graph)
        i = i + 1

    df_pred = pd.DataFrame(randoms, columns=['Stock'])

    df_pred['Today Close '] = old_closes
    df_pred['Tomorrow Close (Predicted)'] = new_closes
    df_pred['Predicted Profit'] = pred_profits
    df_pred["Predicted Profit @ 5 Shares"] = df_pred['Predicted Profit'] * 5
    df_pred["Predicted Profit @ 10 Shares"] = df_pred['Predicted Profit'] * 10
    df_pred["Predicted Profit @ 20 Shares"] = df_pred['Predicted Profit'] * 20
    df_pred["Predicted Profit @ 50 Shares"] = df_pred['Predicted Profit'] * 50
    df_pred["Predicted Profit @ 100 Shares"] = df_pred['Predicted Profit'] * 100
    df_pred['RMSE'] = rmses

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    result = df_pred.sort_values(by='Predicted Profit', ascending=False, )
    print(result)

    avg_rmse = round(df_pred["RMSE"].mean(), 5)

    print("Average Model RMSE: ", avg_rmse)
    print("Saving DataFrame with relevant information to stock_report.txt")
    time.sleep(2)
    print("Results Saved.")
    result.to_pickle("stock_report.txt")
    return plots


# ---------------------------------------------------------------------------------------------------------------------

# Code to provide user interaction and seperation into multiple modes.
to_end = False
while to_end is False:

    # quit code
    mode = input("Which mode would you like to Enter? Run(r), Individual Stocks(i), quit(q)")
    if mode == "q":
        to_end = True
    print("--------------------------------------------------------")

    # run mode code
    if mode.lower() == "r":
        print("Entered Run Mode.")
        how_many = input("How Many stocks from the S&P 500 list would you like to predict? (randomly chosen)")
        graphs = main("run", how_many, "")
        pdf = matplotlib.backends.backend_pdf.PdfPages("predicted_graphs.pdf")
        for fig in range(1, len(graphs) + 1):
            pdf.savefig(fig)
        pdf.close()

    # individual stock mode:
    elif mode.lower() == "i":
        print("Entered Individual Stocks Mode.")
        which_stocks = input("Enter Stock Tickers to predict, separated by a comma, like this: TWTR,AAPL,GE").split(",")
        graphs = main("one stock", 0, which_stocks)
        pdf = matplotlib.backends.backend_pdf.PdfPages("predicted_graphs.pdf")
        for fig in range(1, len(graphs) + 1):
            pdf.savefig(fig)
        pdf.close()

    # Quit Mode: Quits the Program (redundant, as a precaution)
    elif mode.lower() == "q":
        print("Process Quit.")
        to_end = True

    # mode was not recognized
    else:
        print("Unrecognized Input, please try again.")

# ---------------------------------------------------------------------------------------------------------------------
