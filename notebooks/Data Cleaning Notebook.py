import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import numpy as np
import logging
import time

# Set custom values, default as 1/3 each
ENV_WEIGHT = 1/3
SOC_WEIGHT = 1/3
GOV_WEIGHT = 1/3

# Optionally, prompt user for weights
# try:
#     ENV_WEIGHT = float(input('Enter weight for environment_score (default 0.33): ') or 1/3)
#     SOC_WEIGHT = float(input('Enter weight for social_score (default 0.33): ') or 1/3)
#     GOV_WEIGHT = float(input('Enter weight for governance_score (default 0.33): ') or 1/3)
# except Exception as e:
#     print('Invalid input, using default weights.')

# Normalize weights
weight_sum = ENV_WEIGHT + SOC_WEIGHT + GOV_WEIGHT
ENV_WEIGHT /= weight_sum
SOC_WEIGHT /= weight_sum
GOV_WEIGHT /= weight_sum
print(f"Using weights - Environment: {ENV_WEIGHT:.2f}, Social: {SOC_WEIGHT:.2f}, Governance: {GOV_WEIGHT:.2f}")

# turn off warning
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# Load ESG dataset
df = pd.read_csv("public-company-esg-ratings-dataset.csv")

# Drop columns
df.drop(columns=["logo", "weburl", "environment_grade", "environment_level", 
                 "social_grade", "social_level", "governance_grade", 
                 "governance_level", "total_grade", "total_level", 
                 "cik", "last_processing_date", "total_score"], errors='ignore', inplace=True)

# Fill in missing industries
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

tickers = df["ticker"].dropna().unique().tolist()
end_date = datetime.today()
start_date = end_date - timedelta(days=180)

print(f"📊 Processing {len(tickers)} tickers from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

# Download data
print("📈 Downloading S&P 500 data for beta calculation...")
sp500 = yf.download("^GSPC", start=start_date, end=end_date, progress=False, auto_adjust=False)

if sp500 is None or sp500.empty:
    print("⚠️ Warning: Could not download S&P 500 data. Beta calculations will be skipped.")
    sp500_returns = None
else:
    # Handle MultiIndex columns or missing 'Adj Close'
    if isinstance(sp500.columns, pd.MultiIndex):
        sp500_adj_close = sp500["Adj Close"].dropna()
    else:
        sp500_adj_close = sp500['Adj Close'].dropna()
    
    # Calculate daily returns
    sp500_returns = np.log(sp500_adj_close / sp500_adj_close.shift(1)).dropna()
    print(f"✅ S&P 500 data downloaded successfully. {len(sp500_returns)} days of returns data.")

# Step 3: Process each ticker individually for better error handling
results = []
successful_tickers = 0
failed_tickers = 0
no_data_tickers = 0
insufficient_data_tickers = 0
other_errors = 0

print(f"\n🔄 Processing individual tickers...")
for i, ticker in enumerate(tickers):
    if (i + 1) % 50 == 0:
        print(f"   Progress: {i + 1}/{len(tickers)} tickers processed")
    
    try:
        # Download data
        ticker_data = yf.download(
            ticker,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            auto_adjust=True,
            progress=False
        )
        
        if ticker_data is None or ticker_data.empty:
            raise ValueError("No data downloaded")
        
        # Get close price
        close_prices = ticker_data['Close'].dropna()
        
        if len(close_prices) < 30:
            raise ValueError(f"Insufficient data: only {len(close_prices)} days available")
        
        # Calculate latest price
        latest_price = float(close_prices.iloc[-1])
        
        # Calculate 6-month performance
        start_price = float(close_prices.iloc[0])
        perf_6mo = (latest_price - start_price) / start_price
        
        # Calculate beta if S&P 500 data is available
        beta = None
        if sp500_returns is not None and len(sp500_returns) > 0:
            # Calculate stock returns
            stock_returns = np.log(close_prices / close_prices.shift(1)).dropna()
            
            # Align stock returns with market returns
            aligned_data = pd.concat([stock_returns, sp500_returns], axis=1).dropna()
            aligned_data.columns = ['stock', 'market']
            
            if len(aligned_data) >= 30:  # Need at least 30 days for meaningful beta
                try:
                    # Calculate beta 
                    stock_returns_array = aligned_data['stock'].to_numpy()
                    market_returns_array = aligned_data['market'].to_numpy()
                    
                    # Calculate covariance and variance
                    cov_matrix = np.cov(stock_returns_array, market_returns_array)
                    cov = cov_matrix[0, 1]
                    var = np.var(market_returns_array)
                    
                    # Check for valid values
                    if var != 0 and not np.isnan(cov) and not np.isnan(var):
                        beta = cov / var
                        # Check if beta is reasonable 
                        if abs(beta) > 10:  # Unrealistic beta values
                            beta = None
                except Exception as beta_error:
                    # If beta calculation fails set to None
                    beta = None
        
        results.append({
            "ticker": ticker,
            "Latest_Price": round(latest_price, 4),
            "Beta": round(beta, 4) if beta is not None else None,
            "6mo_Performance": round(perf_6mo, 4)
        })
        successful_tickers += 1
        
        # sleep to avoid rate limiting
        time.sleep(0.1)
        
    except Exception as e:
        error_msg = str(e)
        if "No data downloaded" in error_msg:
            print(f"❌ {ticker}: No data available (possibly delisted, SPAC, or invalid ticker)")
            no_data_tickers += 1
        elif "Insufficient data" in error_msg:
            print(f"⚠️ {ticker}: {error_msg}")
            insufficient_data_tickers += 1
        else:
            print(f"❌ {ticker}: {error_msg}")
            other_errors += 1
        
        results.append({
            "ticker": ticker,
            "Latest_Price": None,
            "Beta": None,
            "6mo_Performance": None
        })
        failed_tickers += 1

print(f"\n✅ Processing complete:")
print(f"   ✅ Successful: {successful_tickers}")
print(f"   ❌ No data available: {no_data_tickers}")
print(f"   ⚠️ Insufficient data: {insufficient_data_tickers}")
print(f"   ❌ Other errors: {other_errors}")
print(f"   📊 Total failed: {failed_tickers}")

# Step 5: concatenate
price_df = pd.DataFrame(results)
merged = pd.merge(df, price_df, on="ticker", how="inner")

# Step 6: missing value analysis
print("\n📊 Data Quality Analysis:")
missing_counts = merged[['ticker', 'environment_score', 'social_score', 'governance_score', 'Latest_Price', 'Beta', '6mo_Performance']].isna().sum()
print("Missing values per column:")
print(missing_counts)

# Count delisted companies
delisted_companies = merged[
    merged[['Latest_Price', 'Beta', '6mo_Performance']].isna().all(axis=1)
]
print(f"\n🗑️ Delisted companies (all financial data null): {len(delisted_companies)}")
if len(delisted_companies) > 0:
    print("Delisted tickers:", delisted_companies['ticker'].tolist())

# Count companies with partial data
partial_data = merged[
    merged[['Latest_Price', 'Beta', '6mo_Performance']].isna().any(axis=1) & 
    ~merged[['Latest_Price', 'Beta', '6mo_Performance']].isna().all(axis=1)
]
print(f"\n⚠️ Companies with partial data: {len(partial_data)}")
if len(partial_data) > 0:
    print("Partial data tickers:", partial_data['ticker'].tolist())

# Count valid companies
complete_data = merged.dropna(subset=['Latest_Price', 'Beta', '6mo_Performance'])
print(f"\n✅ Companies with complete financial data: {len(complete_data)}")

# Drop companies with missing value
print(f"\n🧹 Cleaning data...")
print(f"Original dataset size: {len(merged)}")
cleaned_df = merged.dropna(subset=['Latest_Price', 'Beta', '6mo_Performance'])
print(f"After dropping missing financial data: {len(cleaned_df)}")
print(f"Removed {len(merged) - len(cleaned_df)} companies with missing financial data")

# Calculate composite score 
print(f"\n📈 Calculating composite scores (as new total score)...")
valid = cleaned_df.dropna(subset=["environment_score", "social_score", "governance_score"]).copy()

if not valid.empty:
    print(f"Calculating composite scores for {len(valid)} companies with complete data...")
    # Weighted sum
    valid["composite_score"] = (
        ENV_WEIGHT * valid["environment_score"] +
        SOC_WEIGHT * valid["social_score"] +
        GOV_WEIGHT * valid["governance_score"]
    )
    # Merge back to cleaned table
    final_df = pd.merge(cleaned_df, valid[["ticker", "composite_score"]], on="ticker", how="left")
    print(f"✅ Composite scores calculated for {len(valid)} companies")
else:
    print("⚠️ No data available for composite score calculation")
    final_df = cleaned_df.copy()
    final_df["composite_score"] = None

# Step 9: Final output and summary
print(f"\n📊 Final Dataset Summary:")
print(f"Total companies in final dataset: {len(final_df)}")
print(f"Companies with composite scores: {len(final_df.dropna(subset=['composite_score']))}")

print(f"\n📈 Data Quality Summary:")
print(f"✅ Complete data (all fields): {len(final_df.dropna())}")
print(f"⚠️ Missing composite_score: {final_df['composite_score'].isna().sum()}")

print(f"\n📊 Final preview (first 10 companies):")
print(final_df[['ticker', 'name', 'industry', 'environment_score', 'social_score', 'governance_score', 'Latest_Price', 'Beta', '6mo_Performance', 'composite_score']].head(10))

# Save the cleaned dataset
output_filename = "cleaned_esg_stock_data.csv"
final_df.to_csv(output_filename, index=False)
print(f"\n💾 Cleaned dataset saved as: {output_filename}")
