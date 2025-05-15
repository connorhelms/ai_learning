# Find a simple CSV dataset online (e.g., from Kaggle or a government data portal, look for something small and easy to understand). 
# Load it into a Pandas DataFrame and use head(), info(), and describe() to explore it.
import pandas as pd

try:
    csv_dataframe = pd.read_csv('month1\week2\exercise6.csv')
    print(f"CSV DataFrame: \n{csv_dataframe}")
    print("---"*30)
    print(f"CSV DataFrame head: \n")
    print(csv_dataframe.head())
    print("---"*30)
    print(f"CSV DataFrame info: \n")
    print(csv_dataframe.info())
    print("---"*30)
    print(f"CSV DataFrame description: \n")
    print(csv_dataframe.describe())
    print("---"*30)
except FileNotFoundError:
    print("ERROR: File not found.")

