import pandas as pd


df_old = pd.read_pickle("stock_report.txt")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(df_old)
