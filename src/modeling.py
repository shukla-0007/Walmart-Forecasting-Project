import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

df = pd.read_csv('data/cleaned_temp.csv')

# Regression Split
features = ['Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Month', 'Day_of_Week']
X = df[features]
y = df['Weekly_Sales']

train_size = int(len(df) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Linear Regression (Baseline)
lr = LinearRegression().fit(X_train, y_train)
lr_mape = mean_absolute_percentage_error(y_test, lr.predict(X_test))

# Random Forest (Improved)
rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_mape = mean_absolute_percentage_error(y_test, rf_preds)

print(f"Regression MAPE (LR): {lr_mape:.2%}")
print(f"Regression MAPE (RF): {rf_mape:.2%}")

# SARIMAX (Picking Store 1 for the 4-month/4-week logic)
s1 = df[df['Store'] == 1].copy()
s1['Date'] = pd.to_datetime(df['Date'])
s1 = s1.set_index('Date').sort_index()

# Train on approx 4 months, predict 4 weeks
ts_train = s1.iloc[:16] # Weekly data: 16 weeks ~ 4 months
ts_test = s1.iloc[16:20] # 4 weeks

model = SARIMAX(ts_train['Weekly_Sales'], exog=ts_train[['Holiday_Flag', 'CPI']], order=(1,1,1)).fit()
forecast = model.get_forecast(steps=4, exog=ts_test[['Holiday_Flag', 'CPI']])

# EXPORT FOR POWER BI
df.to_csv('/Users/sigma-7/Documents/VS-Code/Walmart Forecasting Project/outputs/walmart_cleaned_for_powerbi.csv', index=False) 
print("Step 2: Modeling complete. Data exported to output folder.") 