# mlstocks
Predicts Closing Stock Prices 24 Hours Later


LSTM Neural Network Based on Randerson112358's post on https://medium.com/@randerson112358/stock-price-prediction-using-python-machine-learning-e82a039ac2bb
Adapted to Predict 24 hour closing differnces instead of long term differences like in the article. Uses Yahoo Stocks as source.

Repo Root has a few important files:

1) main.py - the actual NN predictor and the interface used to interact with it
2) sp500.csv - CSV of the S&P500 stocks that the predictor uses to read in which stocks to predict
3) stock_report.txt - initially blank, this is where the DataFrame of stock predictions and other important
   information will be after the predictor is run.
4) view_report.py - a short script to read in the predictor results and pretty-print the DataFrame.
5) predicted_graphs.png (Will appear after running) - where the graphs of the predicted stocks are saved to. You can see the training results,
   testing resutls, and actual results in these graphs
   
main.py has two modes so far (more may be added in the future) that interact via the python prompter:
1) Run Mode (r): Asks the user for an input of how_many and randomly picks that many stocks out of the SP500 list to predict. Prints and saves a DataFrame
   with all the predicted info and other info to relevant files, mentioned above.
2) Individial Stock Mode (i): Outputs the same info and runs the same methods as run mode, but asks the user for which stocks they want to specifically
   look at. Takes in as many stocks as wanted, in this form: TWTR,SNAP,TSLA


Intended Use: markets Open at 9:30 AM and close 4:00pm, Monday-Friday. So, this is desined to be ran AFTER the markets have closed and yahoo's data
              is updated. There are still after hour transactions, like for robinhood, that can take place up to 6:00pm and open at 9:00am.
              So, I have found the best time to run this is around 8/9pm, to predict the next days 4:00pm closing price for each stock wanted.
              



Other Info:

1) Normalized RMSE(NMRSE) is used to evaulte mode performance/ confidence in the stock prediction. I have found that anything less then 0.1
   is usally pretty accurate the next day, with above 0.1 being more unstable and less reliable, uslaly for stocks that have resecenlty had their IPO.
2) On my 2018 MacBook Pro, each predict takes about 30 seconds, so factor this in when running large lists of stocks. I am currenlty working on optimizing
   and hope to get it way down.
