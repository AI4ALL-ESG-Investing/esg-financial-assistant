# Data Cleaning Notebook

# Import python libraries
import pandas as pd
import yfinance as yf
import logging
from sklearn.preprocessing import MinMaxScaler

logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# Clean ESG dataset
df = pd.read_csv("public-company-esg-ratings-dataset.csv")
df.head()

# Drop unnecessary columns
df.drop(columns=["logo", "weburl", "environment_grade", "environment_level", 
                 "social_grade", "social_level", "governance_grade", 
                 "governance_level", "total_grade", "total_level", 
                 "cik", "last_processing_date"], inplace=True)


# Check for missing values
df.isna().any()

# Identify rows with missing industry values
row_na = df[df["industry"].isna()]

# Print names of companies with missing industry
print(row_na["name"])

# Map industries to companies
industry_map = {
    'Armada Acquisition Corp I': 'Financial Services',
    'Acri Capital Acquisition Corp': 'Financial Services',
    'ACE Convergence Acquisition Corp': 'Technology',
    'Edoc Acquisition Corp': 'Healthcare',
    'AF Acquisition Corp': 'Financial Services',
    'AIB Acquisition Corp': 'Financial Services',
    'Sports Ventures Acquisition Corp': 'Media & Entertainment',
    'Alignment Healthcare LLC': 'Healthcare',
    'Health Assurance Acquisition Corp': 'Healthcare',
    'Healthcare Services Acquisition Corp': 'Healthcare',
    'Artisan Acquisition Corp': 'Financial Services',
    'Powered Brands': 'Consumer Goods',
    'Concord Acquisition Corp': 'Financial Services'
}

df['industry'] = df.apply(
    lambda row: industry_map[row['name']] if pd.isna(row['industry']) and row['name'] in industry_map else row['industry'],
    axis=1
)

# Check for missing values again
df.isna().any()

# Fetch unique tickers from the DataFrame
tickers = df["ticker"].unique()

# fetch historical data
try:
    historical_data = yf.download(tickers, period="6mo")['Close']
except Exception as e:
    print(f"An unknown error occurred while downloading historical data: {e}")
    historical_data = pd.DataFrame()  # Create an empty DataFrame in case of error


results = []

# Loop 
for ticker in tickers:
    try:
        stock = yf.Ticker(ticker)
        
        # Latest price
        latest_price = historical_data[ticker].iloc[-1] if ticker in historical_data.columns else None
        
        # beta
        beta = stock.info.get('beta', None)
        
        # 6-month performance
        if ticker in historical_data.columns:
            six_month_performance = (historical_data[ticker].iloc[-1] - historical_data[ticker].iloc[0]) / historical_data[ticker].iloc[0]
        else:
            six_month_performance = None
        
        # Append
        results.append({
            'ticker': ticker,
            'Latest_Price': latest_price,
            'Beta': beta,
            '6mo_Performance': six_month_performance
        })
        
    except Exception as e:
        print(f"An unknown error occurred for {ticker}: {e}")

results_df = pd.DataFrame(results)

# Merge 
stock_merged = pd.merge(df, results_df, on="ticker", how="inner")

filtered = stock_merged.dropna(subset=["ESG Score", "6mo_Performance", "Beta"]).copy()

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(filtered[["ESG Score", "6mo_Performance", "Beta"]])

# composite score: weights can be adjusted
composite = (
    0.4 * scaled_features[:, 0] +    # ESG Score
    0.4 * scaled_features[:, 1] -    # 6mo Performance
    0.2 * scaled_features[:, 2]      # Beta (risk), penalized
)

filtered["composite_score"] = composite

# Merge back 
stock_final = pd.merge(stock_merged, filtered[["ticker", "composite_score"]], on="ticker", how="left")

print(stock_merged.head(20))
