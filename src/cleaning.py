import pandas as pd

# Load and convert to Parquet as per notes
df = pd.read_csv('data/Walmart_Store_sales.csv')
df.to_parquet('data/walmart_data.parquet')

# Load the parquet file
df = pd.read_parquet('data/walmart_data.parquet')

# Date Parsing (DD-MM-YYYY)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# Cleaning: remove duplicates and negative sales
df = df.drop_duplicates()
df = df[df['Weekly_Sales'] > 0]

# Sorting and Forward Fill missing values
df = df.sort_values(by=['Store', 'Date'])
df = df.ffill()

# Calendar Features
df['Month'] = df['Date'].dt.month
df['Day_of_Week'] = df['Date'].dt.dayofweek

# Save for modeling phase
df.to_csv('data/cleaned_temp.csv', index=False)
print("Step 1: Cleaning and Parquet conversion done.") 