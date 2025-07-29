from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load data
cluster_data = pd.read_csv("cleaned_esg_stock_data.csv")
print(f"Dataset loaded: {len(cluster_data)} companies")

# K-Means clustering with k=5
features = ["Beta", "1yr_Performance"]
scaler_rr = StandardScaler()
filtered = cluster_data.copy()
scaled = scaler_rr.fit_transform(filtered[features])
k_final = 5
kmeans_final = KMeans(n_clusters=k_final, random_state=42, n_init='auto')
labels_final = kmeans_final.fit_predict(scaled)
final_cluster_col = f'final_cluster_{k_final}'
filtered[final_cluster_col] = labels_final
final_k = k_final

# Visualization
plt.figure(figsize=(10,7))
sns.scatterplot(
    x=filtered['Beta'],
    y=filtered['1yr_Performance'],
    hue=filtered[final_cluster_col],
    palette='Set2',
    alpha=0.7,
    s=60
)
plt.title(f'K-Means Clustering (k={final_k}): Risk vs Return (Cleaned Data)', fontsize=15)
plt.xlabel('Beta (Risk)', fontsize=12)
plt.ylabel('1-Year Performance (Return)', fontsize=12)
plt.legend(title='Cluster', fontsize=10)
plt.grid(True, alpha=0.3)
plt.show()

# Cluster explanation
print(f"\nCLUSTER FEATURE EXPLANATION (k={final_k}):")
summary_rows = []
for cluster_id in sorted(pd.unique(filtered[final_cluster_col])):
    cluster_companies = filtered[filtered[final_cluster_col] == cluster_id]
    print(f"\nCluster {cluster_id} ({len(cluster_companies)} companies):")
    for feature in ["Beta", "1yr_Performance"]:
        mean = cluster_companies[feature].mean()
        std = cluster_companies[feature].std()
        print(f"  {feature}: mean={mean:.3f}, std={std:.3f}")
    beta_mean = cluster_companies['Beta'].mean()
    perf_mean = cluster_companies['1yr_Performance'].mean()
    industry_counts = cluster_companies['industry'].value_counts().head(3)
    print(f"  Top Industries: {', '.join([f'{ind} ({count})' for ind, count in industry_counts.items()])}")
    if beta_mean > 1.5:
        risk_level = 'Very High Risk'
    elif beta_mean > 1.2:
        risk_level = 'High Risk'
    elif beta_mean > 0.8:
        risk_level = 'Moderate Risk'
    else:
        risk_level = 'Low Risk'
    if perf_mean > 0.2:
        return_level = 'Exceptional Return'
    elif perf_mean > 0.1:
        return_level = 'Strong Return'
    elif perf_mean > 0.05:
        return_level = 'Good Return'
    elif perf_mean > -0.05:
        return_level = 'Neutral Return'
    else:
        return_level = 'Weak Return'
    summary_rows.append({
        'K': final_k,
        'Cluster': cluster_id,
        'N': len(cluster_companies),
        'Risk': risk_level,
        'Return': return_level
    })
# Print summary
print(f"\nSummary Table for K={final_k}:")
print(f"{'K':<3} {'Cluster':<7} {'N':<5} {'Risk':<18} {'Return':<20}")
for row in summary_rows:
    print(f"{row['K']:<3} {row['Cluster']:<7} {row['N']:<5} {row['Risk']:<18} {row['Return']:<20}")

# Save
output_filename = "clustered_esg_stock_data.csv"
filtered.to_csv(output_filename, index=False)
print(f"\nClustered dataset saved as: {output_filename}")

def recommendation(user_beta, user_performance, e_weight, s_weight, g_weight, num_recommendations=10):
    """
    Parameters:
    user_beta: User's risk tolerance (-4 - 4)
    user_performance: User's expected return (-2 - 8)
    e_weight: Weight for environment_score (0 - 1)
    s_weight: Weight for social_score (0 - 1)
    g_weight: Weight for governance_score (0 - 1)
    num_recommendations: Number of stocks to recommend (default: 15)
    """
    
    weight_dict = {
        'environment_score': e_weight,
        'social_score': s_weight,
        'governance_score': g_weight
    }
    # Sort 
    sorted_fields = sorted(weight_dict, key=weight_dict.get, reverse=True)
    # Check for ties 
    max_weight = max(weight_dict.values())
    max_fields = [field for field, w in weight_dict.items() if w == max_weight]
    if len(max_fields) > 1:
        if len(max_fields) == 3:
            print("All ESG weights are equal. Sorting by 'environment_score' only.")
            sorted_fields = ['environment_score']
        else:
            print(f"Multiple highest weights detected. Sorting by '{max_fields[0]}' only.")
            sorted_fields = [max_fields[0]]
    else:
        print(f"Automatic sorting order: {sorted_fields}")
    # Standardize 
    user_features = scaler_rr.transform([[user_beta, user_performance]])
    # Predict
    user_cluster = kmeans_final.predict(user_features)[0]
    print(f"User assigned to cluster {user_cluster}")
    user_cluster_stocks = filtered[filtered[final_cluster_col] == user_cluster].copy()
    # Sort 
    user_cluster_stocks = user_cluster_stocks.sort_values(by=sorted_fields, ascending=[False]*len(sorted_fields))
    # Recommend
    top_recommend = user_cluster_stocks.head(num_recommendations)
    print(f"Top {num_recommendations} recommended stocks:")
    print(top_recommend[['ticker', 'name', 'Beta', '1yr_Performance', 'environment_score', 'social_score', 'governance_score']])
    # save
    top_recommend.to_csv('user_recommended_stocks.csv', index=False)
    print(f"Top {num_recommendations} recommended stocks have been saved to 'user_recommended_stocks.csv'.")
    return top_recommend

if __name__ == "__main__":
    user_beta = 1.1
    user_performance = 0.15
    e_weight = 0.3
    s_weight = 0.5
    g_weight = 0.2
    num_recommendations = 15  # Customizable number of recommendations
    recommendation(user_beta, user_performance, e_weight, s_weight, g_weight, num_recommendations)
