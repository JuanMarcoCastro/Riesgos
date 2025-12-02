
import pandas as pd
import numpy as np
from scipy import stats

# Load data
df = pd.read_csv('precios_historicos.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Select columns
cols = ['Date', 'NVDA.O', 'AAPL.O', 'LMT', 'MSFT.O', 'SIEGn.DE Usd', 'GOOGL.O', 'INTC.O']
df = df[cols]
df = df.rename(columns={'SIEGn.DE Usd': 'SIEGn.DE'})

# Filter for the first window to check counts
start_date = '2021-08-23'
end_date = '2025-09-08'

mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
window_df = df.loc[mask]
print(f"First window rows: {len(window_df)}")
print(f"First date: {window_df['Date'].iloc[0]}")
print(f"Last date: {window_df['Date'].iloc[-1]}")

# Define stocks
stocks = ['NVDA.O', 'AAPL.O', 'LMT', 'MSFT.O', 'SIEGn.DE', 'GOOGL.O', 'INTC.O']

# Calculate Portfolio Value (assuming 1 share of each)
df['Portfolio_Value'] = df[stocks].sum(axis=1)
df['Portfolio_Return'] = df['Portfolio_Value'].pct_change()

# Z-scores
z_95 = stats.norm.ppf(0.95)
z_99 = stats.norm.ppf(0.99)

results = []

# Find index of start and end date
try:
    start_idx = df[df['Date'] == start_date].index[0]
    end_idx = df[df['Date'] == end_date].index[0]
except IndexError:
    print("Start or End date not found exactly. Using closest.")
    # This logic needs to be robust if dates are missing, but user implies they exist.
    # I'll rely on the sorted dataframe and integer indexing.
    
# Re-locate based on sorted integer index
df = df.reset_index(drop=True)
start_idx_int = df[df['Date'] == start_date].index[0]
end_idx_int = df[df['Date'] == end_date].index[0]

print(f"Start Index: {start_idx_int}, End Index: {end_idx_int}")
print(f"Window size: {end_idx_int - start_idx_int + 1}")

window_size = 1000 # User said "mil datos"

# Iterate
# We need a window of 1000 data points.
# If the user wants the calculation for 8/9/2025 using the previous 1000 days (including 8/9/2025?), 
# usually VaR is calculated using historical data to predict *tomorrow's* risk, or describing the risk *of* the portfolio held today based on history.
# "sacar todo esto desde 23/8/2021 a 8/9/2025... mil días... quiero calcular todo con márgenes de mil días históricos"
# "luego se calcula desde 24/8/2021 a 9/9/2025"
# This implies a rolling window of size 1000.

# Let's assume the window size is fixed at 1000.
# The first window ends at '2025-09-08'.
# We will iterate from that index until the end of the dataframe.

current_end_idx = end_idx_int

while current_end_idx < len(df):
    # Define window indices
    # Window should be 1000 days.
    # If current_end_idx is the last day, start is current_end_idx - 999
    current_start_idx = current_end_idx - 1000 + 1
    
    if current_start_idx < 0:
        print(f"Not enough data for window ending at {current_end_idx}")
        current_end_idx += 1
        continue
        
    window_data = df.iloc[current_start_idx : current_end_idx + 1]
    
    # Current Date (for which VaR is reported)
    current_date = df.iloc[current_end_idx]['Date']
    
    # Portfolio Value Today (V0)
    V0 = df.iloc[current_end_idx]['Portfolio_Value']
    
    # Returns in this window
    # Note: pct_change gives NaN for the first item.
    # We should calculate returns on the window data or use the pre-calculated returns.
    # Using pre-calculated returns:
    window_returns = df.iloc[current_start_idx : current_end_idx + 1]['Portfolio_Return'].dropna()
    
    # Parametric VaR
    mu = window_returns.mean()
    sigma = window_returns.std()
    
    # VaR = V0 * (mu - z * sigma) -> This is usually negative.
    # User wants positive values (Loss).
    # VaR_95 = V0 * (z_95 * sigma - mu) ? Or just V0 * z_95 * sigma (assuming mu=0)?
    # Notebook used: V0 * (rendimiento_promedio - z * desviacion_estandar) which is negative.
    # Then abs().
    
    var_param_95 = abs(V0 * (mu - z_95 * sigma))
    var_param_99 = abs(V0 * (mu - z_99 * sigma))
    
    # Non-Parametric VaR (Historical)
    # Calculate PnL vector
    pnl_vector = window_returns * V0
    
    # Percentiles
    # 5% percentile for 95% confidence
    # 1% percentile for 99% confidence
    var_hist_95 = abs(np.percentile(pnl_vector, 5))
    var_hist_99 = abs(np.percentile(pnl_vector, 1))
    
    results.append({
        'Date': current_date,
        'VaR_Param_95_USD': var_param_95,
        'VaR_Param_95_Pct': (var_param_95 / V0) * 100,
        'VaR_Param_99_USD': var_param_99,
        'VaR_Param_99_Pct': (var_param_99 / V0) * 100,
        'VaR_Hist_95_USD': var_hist_95,
        'VaR_Hist_95_Pct': (var_hist_95 / V0) * 100,
        'VaR_Hist_99_USD': var_hist_99,
        'VaR_Hist_99_Pct': (var_hist_99 / V0) * 100
    })
    
    current_end_idx += 1

results_df = pd.DataFrame(results)
print(results_df.head())
print(results_df.tail())
