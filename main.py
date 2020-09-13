# main.py - contains the driver code for predicting stock opening prices 2 days from now
# based on: https://medium.com/@randerson112358/stock-price-prediction-using-python-machine-learning-e82a039ac2bb
# Most of core code came from above source. This program adapts it into short term predictions and also puts it inside
# a wrapper program to interact with, with the addition of different modes and multi-stock prediction.


# Program predicts opening price two days in advance. Here is how to use:
# Run model after hours on current day(today), after the prices have stablized - sometime between 6pm and 9:00am EST
# Stocks will be queued to buy on market open at 9:30 AM. Once 9:30 hits, stocks will be bought.
# The day will play out and markets will close at 4:00 and after hours at 6:00pm.
# Sell stocks next day at market open at predicted market open price.
#
# EXAMPLE:
# Evening, 1/1/2020: Run Model after Hours to see what stocks will go up by morning of 1/3/2020. Stocks are queued
# Morning, 1/2/2020: Stocks are bought at open prices. Re-Run program for two days from now.
# Evening, 1/2/2020: Stocks have their day.
# Morning, 1/3/2020: Sell stocks at predicted open price. Repeat each day. Make Bank.
#

# ---------------------------------------------------------------------------------------------------------------------

# Import statements
import random
import time
import math
from itertools import chain

import pandas_datareader as web
import numpy as np
import pandas as pd
from matplotlib import pyplot

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.backends.backend_pdf
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ---------------------------------------------------------------------------------------------------------------------


def up_down(df):
    ups = []
    downs = []

    for index, row in df.iterrows():
        predicted_profit = row["Predicted Profit"]
        if predicted_profit >= 0:
            ups.append('UP')
        else:
            downs.append("DOWN")

    return len(ups), len(downs)



# parameters: stock (for i mode, which stock(s) to look at), save_graphs (True for saving, False for not)
def predict_open(stock, save_graphs, verbose, to_verify):
    todays_date = datetime.date(datetime.now())
    yesterdays_date = datetime.strftime(datetime.now() - timedelta(1), '%Y-%m-%d')
    tomorrows_date = datetime.strftime(datetime.now() + timedelta(1), '%Y-%m-%d')
    if(verbose):
        print('*** Fetching Most Up-To-Date Stock Information...')
    df = web.DataReader(stock, data_source='yahoo', start='2012-01-01', end=todays_date)
    data = df.filter(['Close'])
    data_y = df.filter(['Open'])
    dataset = data.values
    dataset_y = data_y.values
    data_length = len(dataset)
    training_data_len = math.ceil(len(dataset) * .80)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    scaled_data_y = scaler.fit_transform(dataset_y)

    if(verbose):
        print('*** Data Fetched. Training will commence...')

    # Create the scaled training data set
    train_data = scaled_data[0:training_data_len, :]
    train_data_y = scaled_data_y[0:training_data_len, :]


    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []


    #max = len(train_data) - 1
    # max = int(len(train_data) / 50)
    max = 2
    # print("Days in : ", max)


    for i in range(max, len(train_data)):
        x_train.append(train_data[i - max:i, 0])
        y_train.append(train_data_y[i, 0])
    # Convert x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
    # print("Shape of x_train: ", x_train.shape)
    # print("Shape of y_train:", y_train.shape)

    # Reshape the data into the shape accepted by the LSTM
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    #x_train.shape[1]
    # actual mode creation
    model = Sequential()
    model.add(LSTM(units=128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.20))
    model.add(LSTM(units=64, return_sequences=False))
    model.add(Dropout(0.20))
    model.add(Dense(units=32))
    model.add(Dense(units=16))
    model.add(Dense(units=1))
    if (verbose):
        model.summary()

    model.compile(optimizer='adam', loss='mean_squared_error')

    if to_verify:
        history = model.fit(x_train, y_train, batch_size=1, epochs=10)
    else:
        history = model.fit(x_train, y_train, batch_size=1, epochs=3)

    # Test data set
    test_data = scaled_data[training_data_len - max:, :]

    # Create the x_test and y_test data sets
    x_test = []
    y_test = dataset_y[training_data_len:, :]
    for i in range(max, len(test_data)):
        x_test.append(test_data[i - max:i, 0])

    # Convert x_test to a numpy array
    x_test = np.array(x_test)

    # Reshape the data into the shape accepted by the LSTM
    x_test_2 = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


    # Getting the models predicted price values
    predictions = model.predict(x_test_2)
    predictions = scaler.inverse_transform(predictions)  # Undo scaling


    shift = 0.0005
    pred = predictions + (predictions * shift)

    # Calculate/Get the value of RMSE and normalize
    rmse = np.sqrt(np.mean(((pred - y_test) ** 2))) / np.mean(y_test)

    train = data_y[:training_data_len]
    valid = data_y[training_data_len:]
    valid2 = valid.copy(deep=True)
    valid2['Predictions'] = predictions + (predictions * shift)

    # creates and saved graphs if wanted
    if save_graphs:
        if verbose:
            print('*** Creating and Exporting Graph...')
        plt.figure(figsize=(16, 8))
        plt.title(stock + " Open Price in Two Days, Based on entire History")
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Open Price USD ($)', fontsize=18)
        plt.plot(train['Open'])
        plt.plot(valid2[['Open', 'Predictions']])
        plt.legend(['Train', 'Val', 'Predictions'], loc='lower left')
        # plt.show()
        if verbose:
            print('*** Graph Created and Saved...')

    if verbose:
        print('*** Summarizing Results...')



    # Get the quote
    quote = web.DataReader(stock, data_source='yahoo', start='2012-01-01', end=todays_date)

    # Create a new dataframe
    new_df = quote.filter(['Close'])

    # Get the last 2 day Open price
    # Determines the number of TimeStamps you want - how far back into the past you need to go
    last_days = new_df[-max:].values

    # Scale the data to be values between 0 and 1
    last_days_scaled = scaler.transform(last_days)

    # Create an empty list
    final_pred = []

    # Append the past 2 days
    final_pred.append(last_days_scaled)

    # Convert the X_test data set to a numpy array
    final_pred = np.array(final_pred)

    # Reshape the data
    final_pred_np = np.reshape(final_pred, (final_pred.shape[0], final_pred.shape[1], 1))


    # Get the predicted scaled price
    pred_price = model.predict(final_pred_np)

    # undo the scaling
    pred_open = scaler.inverse_transform(pred_price)

    # get todays close (or most recent)
    old_close = df['Close'].iloc[-1]
    old_close_date = df.axes[0].tolist()[-1]
    #print("Old Close Date", old_close_date)

    # creates info to view and append to DataFrame
    pred_profit = pred_open[0][0] - old_close
    pred_mvmt_perc = round((pred_profit / old_close) * 100, 5)
    # print("Old Close Price (Today): ", old_close)
    # print("Predicted Open (2 Days from Now): ", pred_open[0][0])
    # print("Predicted Profit: ", pred_profit)
    # print("Model NRMSE: ", rmse)
    # print("Predicted Movmement Percent: ", pred_mvmt_perc)
    if (verbose):
        print('*** Done.')

    return stock, old_close, round(pred_open[0][0], 3), round(pred_profit, 3), rmse, pred_mvmt_perc, plt


# ---------------------------------------------------------------------------------------------------------------------
# Make Table method.
# Parameters: mode(user mode), how_many(for r mode), and stock(for i mode)
# calls the predict method appropriatly for each mode.

def make_table(the_list, to_verify):

    old_closes = []
    new_closes = []
    pred_profits = []
    pred_movement_percents = []
    rmses = []
    i = 1
    plots = []

    for each_stock in the_list:
        print("------------------------------------------------------------")
        print("Predicting: ", each_stock, "(Stock Number: ", str(i) + ")")
        # Boolean Value Controls graph: True displays predicted/actual graph, False does not

        # graphs, verbose, to_verfiy
        if to_verify:
            stock, old_close, pred_close, pred_profit, rmse, pred_mvmt_perc, graph = predict_open(each_stock,
                                                                                       True, False, True)
        else:
            stock, old_close, pred_close, pred_profit, rmse, pred_mvmt_perc, graph = predict_open(each_stock,
                                                                                                  True, False, False)
        old_closes.append(old_close)
        new_closes.append(pred_close)
        pred_profits.append(pred_profit)
        rmses.append(rmse)
        pred_movement_percents.append(pred_mvmt_perc)
        plots.append(graph)
        i = i + 1

    df_pred = pd.DataFrame(the_list, columns=['Stock'])

    df_pred['Today Open '] = old_closes
    df_pred['Predicted Open in Two days'] = new_closes
    df_pred['Predicted Profit'] = pred_profits
    df_pred['Profit Percentage'] = pred_movement_percents
    df_pred["Predicted Profit @ 5 Shares"] = df_pred['Predicted Profit'] * 5
    df_pred["Predicted Profit @ 10 Shares"] = df_pred['Predicted Profit'] * 10
    df_pred["Predicted Profit @ 20 Shares"] = df_pred['Predicted Profit'] * 20
    df_pred["Predicted Profit @ 50 Shares"] = df_pred['Predicted Profit'] * 50
    df_pred["Predicted Profit @ 100 Shares"] = df_pred['Predicted Profit'] * 100
    df_pred["Predicted Profit @ 500 Shares"] = df_pred['Predicted Profit'] * 500
    df_pred["Predicted Profit @ 1000 Shares"] = df_pred['Predicted Profit'] * 1000
    df_pred['Test RMSE'] = rmses

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    result = df_pred.sort_values(by='Profit Percentage', ascending=False, )
    print(result)

    avg_rmse = round(df_pred["Test RMSE"].mean(), 5)

    print("Average Model RMSE: ", avg_rmse)
    print("Saving DataFrame with relevant information to stock_report.txt")
    #time.sleep(2)
    print("Results Saved.")
    result.to_pickle("stock_report.txt")
    return plots

# ---------------------------------------------------------------------------------------------------------------------
# verify method.
# Parameters: mode(user mode), how_many(for r mode), and stock(for i mode)
# calls the predict method appropriatly for each mode.
def verify(the_list):
    print("Verifying stocks entered...")
    runs = 5
    new_list = []
    for stock in the_list:
        new_list.append([stock] * runs)
    flat = list(chain.from_iterable(new_list))
    graphs = make_table(flat, True)
    new_df = pd.read_pickle("stock_report.txt")
    for stock in the_list:
        print("--------------------------------------------------------")
        print("Verification Report for " + str(stock) + ": ")
        rslt_df = new_df[new_df['Stock'] == stock]
        print(rslt_df)
        avg_rmse = round(rslt_df["Test RMSE"].mean(), 4)
        ups, downs = up_down(rslt_df)
        up_prop = round(ups / (ups + downs), 3) * 100
        print(str(ups) + " ups and " + str(downs) + " downs ")
        print("Up Percentage: " + str(up_prop)+ "%")
        print("Average Model RMSE for " + str(stock) + ": " + str(avg_rmse))
        if up_prop >= 0.80:
            to_invest = "yes"
        else:
            to_invest = "no"
        print(" **** BOTTOM LINE (INVEST YES/NO): ", to_invest, "***")
    return graphs



# ---------------------------------------------------------------------------------------------------------------------
# call_methods method.
# Parameters: mode(user mode), how_many(for r mode), and stock(for i/v mode)
# returns: numpy gra[h objects of the stocks predicted
# calls the predict method appropriatly for each mode.

def call_methods(mode, stock, how_many):
    # read in stock lists
    the_list = []
    df_sp500 = pd.read_csv("sp500.csv")
    yeet = df_sp500['Symbol'].tolist()

    # assign the list of stocks according to the mode:
    if mode.lower() == "one stock" or mode.lower() == "verify":
        stock_list = []
        for each_stock in stock:
            stock_list.append(each_stock)
            the_list = stock_list
    elif mode.lower() == "run":
        randoms = []
        for i in range(int(how_many)):
            stocks_random = random.choice(yeet)
            randoms.append(stocks_random)
        the_list = randoms
    else:
        the_list = []

    if mode.lower() == "verify":
        graphs = verify(the_list)
        return graphs
    else:
        graphs = make_table(the_list, False)
        return graphs

# ---------------------------------------------------------------------------------------------------------------------

# Code to provide user interaction and seperation into multiple modes.

def main():

    to_end = False
    while to_end is False:

        # quit code
        mode = input("Which mode would you like to Enter? Run(r), Individual Stocks(i), verify(v), quit(q)")
        if mode == "q":
            to_end = True
        print("--------------------------------------------------------")

        # run mode code
        if mode.lower() == "r":
            print("Entered Run Mode.")
            how_many = input("How Many stocks from the NYSE list would you like to predict? (randomly chosen)")
            graphs = call_methods("run", "", how_many)
            pdf = matplotlib.backends.backend_pdf.PdfPages("predicted_graphs.pdf")
            for fig in range(1, len(graphs) + 1):
                pdf.savefig(fig)
            pdf.close()

        # individual stock mode:
        elif mode.lower() == "i":
            print("Entered Individual Stocks Mode.")
            which_stocks = input("Enter Stock Tickers to predict, separated by a comma, like this: TWTR,AAPL,GE").replace(" ", "").split(",")
            graphs = call_methods("one stock", which_stocks, 0)
            pdf = matplotlib.backends.backend_pdf.PdfPages("predicted_graphs.pdf")
            for fig in range(1, len(graphs) + 1):
                pdf.savefig(fig)
            pdf.close()

        elif mode.lower() == "v":
            print("Entered Verfication Mode")
            which_stocks = input("Enter Stock Tickers to predict, separated by a comma, like this: TWTR,AAPL,GE").replace(" ", "").split(",")
            result = call_methods("verify", which_stocks, 0)
            to_end = True

        # Quit Mode: Quits the Program (redundant, as a precaution)
        elif mode.lower() == "q":
            print("Process Quit.")
            to_end = True

        # mode was not recognized
        else:
            print("Unrecognized Input, please try again.")
# ---------------------------------------------------------------------------------------------------------------------
main()