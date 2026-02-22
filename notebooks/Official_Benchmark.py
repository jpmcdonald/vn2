import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INDEX = ["Store", "Product"]

# Step 1 - Extract data

sales = pd.read_csv("Week 0 - 2024-04-08 - Sales.csv").set_index(INDEX)
in_stock = pd.read_csv("Week 0 - In Stock.csv").set_index(INDEX)
state = pd.read_csv("Week 0 - 2024-04-08 - Initial State.csv").set_index(INDEX)
sales.columns = pd.to_datetime(sales.columns)
in_stock.columns = pd.to_datetime(in_stock.columns)

sales[~in_stock] = np.nan 
#These are shortages, we'll put missing data

# Step 2 - Make a Seasonal Moving Average Forecast

# Step 2a - Compute Seasonal Factors

# We compute *simple* multiplicative weekly seasonal parameters
season = sales.mean().rename("Demand").to_frame()
season["Week Number"] = season.index.isocalendar().week
season = season.groupby("Week Number").mean() 
#Seasonal parameters (multiplicative) per week
season = season / season.mean() 
#Normalize to one.

# Step 2b - Un-seasonalize Demand

sales_weeks = sales.columns.isocalendar().week
sales_no_season = sales / (season.loc[sales_weeks.values]).values.reshape(-1)

# Let's plot to take a look
ax = sales.sum().plot(label="Sales", legend=True)
sales_no_season.sum().plot(ax=ax, label="Sales No Season", legend=True)
plt.show()

# Step 2c we make a forecast using a 13 weeks moving average (the number is arbitrary)
base_forecast = sales_no_season.iloc[:,-13:].mean(axis=1) 
# That's the unseasonalized moving average of the last 8 weeks
# We need a forecast for 3 weeks.
f_periods = pd.date_range(start=sales.columns[-1], periods=10, inclusive="neither", freq="W-MON")
forecast = pd.DataFrame(data=base_forecast.values.reshape(-1,1).repeat(len(f_periods), axis=1), 
                        columns=f_periods,
                        index=sales.index)
# We need to seasonalize this for future forecast. 
forecast = forecast * (season.loc[f_periods.isocalendar().week.values]).values.reshape(-1)

# Let's plot to take a look
ax = sales.sum().plot(label="Sales", legend=True)
sales_no_season.sum().plot(ax=ax, label="Sales No Season", legend=True)
forecast.sum().plot(ax=ax, label="Forecast", legend=True)
plt.legend()
plt.show()

# Step 3 use a forecast-driven order-up-to policy with 4 weeks as coverage.

order_up_to = forecast.iloc[:,:4].sum(axis=1)
net_inventory = state[["In Transit W+1", "In Transit W+2", "End Inventory"]].sum(axis=1)
order = (order_up_to - net_inventory).clip(lower=0).round(0).astype(int)
order.to_csv("Benchmark Order 0.csv")