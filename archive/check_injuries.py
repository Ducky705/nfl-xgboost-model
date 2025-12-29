
import nfl_data_py as nfl
import pandas as pd

try:
    print("Fetching injury data...")
    injuries = nfl.import_injuries([2023])
    print("Columns:", injuries.columns.tolist())
    print(injuries.head())
    print("Unique statuses:", injuries['report_status'].unique())
except Exception as e:
    print(f"Error fetching injuries: {e}")
