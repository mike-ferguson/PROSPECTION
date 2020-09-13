![alt text](https://res.cloudinary.com/jerrick/image/upload/fl_progressive,q_auto,w_1024/iyhg6zsvgfjdhgophzji.jpg)


PROSPECTION 
Version 1.1

This model Predicts Closing Stock Prices, 24 hours after last run. Uses all past data, going back as far as January 2012 (If stock is old enough).
This program looks at the S&P 500 and predicts all the stocks, and sorts by which stock it predicts will go up in 1 day by the highest percent.
It also will filter out in the final report (dataframe sorted by predicted profit percentage) any stock whose Test RMSE value was above a cutoff,
the defualt being 0.05. See below for other modes.

LSTM Neural Network Based on Randerson112358's post on https://medium.com/@randerson112358/stock-price-prediction-using-python-machine-learning-e82a039ac2bb
Adapted to Predict 24 hour closing differnces instead of long term differences like in the article. Uses Yahoo Stocks as source.

Repo Root has a few important files:

1) main.py - the actual NN predictor and the interface used to interact with it
2) sp500.csv - CSV of the S&P500 stocks that the predictor uses to read in which stocks to predict
3) stock_report.txt - initially blank, this is where the DataFrame of stock predictions and other important
   information will be after the predictor is run.
4) view_report.py - a short script to read in the predictor results and pretty-print the DataFrame.
5) predicted_graphs.png (Will appear after running) - where the graphs of the predicted stocks are saved to. You can see the training results,
   testing results, and actual results in these graphs.
   
   
main.py has three modes so far (more may be added in the future) that interact via the python prompter:

1) Run Mode (r): USed when you just want to see what stocks are mostl likley going to have a higher closing price.
   Asks the user for an input of how_many and randomly picks that many stocks out of the S&P 500 list to predict. 
   Prints and saves a DataFrame with all the predicted info and other info to relevant files, mentioned above.
2) Individial Stock Mode (i): Used when you want to assess specific stocks' likely closing performace, 
   Outputs the same info and runs the same methods as run mode, but asks the user for which stocks they 
   want to specifically look at. Takes in as many stocks as wanted, in this form: TWTR,SNAP,TSLA
3) Verify (v): Used to run the model many times as a sort of vailidation. Increases the number of epochs and runs the model many times. 
   Using a pre-defined threshold, outputs whether or not the user should buy the stock: i.e, if 4/5 predictions are up, then buy, etc.


Intended Use: Markets open at 9:30am and close 4:00pm, Monday-Friday. So, this is designed to be ran AFTER the markets have closed and Yahoo's data
              is updated. Some brokers use after-hour trading, like Robinhood, so "extended" hours are 9:00am-6:00pm.
              So, I have found the best time to run this is around 8/9pm, to predict the next days 4:00pm closing price for each stock wanted.


Other Info:

1) Normalized RMSE (NMRSE) is used to evaluate model performance/confidence in the stock prediction. I have found that anything less then 0.1
   is usually pretty accurate the next day, with above 0.1 being more unstable and less reliable, usually for stocks that have recently had their IPO,
   or for stocks that have crazy price fluctuations, like Tesla or some other penny stocks.
2) On my 2018 MacBook Pro, each individual predict takes about 2 seconds (serial, not parallized yet), so factor this in when running large lists of stocks. 
   I am currenlty working on optimizing and hope to get the time way down.


How To Use: I created this program to be very easy to get started and use in a black-box way, without needing to know too much ML, again using most of the source above, so credit to them for the actual model and most of the code. Here is how to use:
1) Clone repo locally
2) Run main.py in a terminal or in an IDE, follow the prompts. When done, make sure to enter q to quit and kill the program.
3) run veiw_results.py in a terminal/IDE to look at the results. That's it! It's that easy :) TO THE MOON!
