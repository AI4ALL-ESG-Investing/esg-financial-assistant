from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names, but StandardScaler was fitted with feature names")

# Load
cluster_data = pd.read_csv("cleaned_esg_stock_data.csv")
print(f"Dataset loaded: {len(cluster_data)} companies")

# K-Means Clustering on Beta and return
print("\n=== K-Means Clustering on Beta and 1yr_Performance ===")

# Select features
risk_return_features = cluster_data[["Beta", "1yr_Performance"]].copy()
scaler_rr = StandardScaler()
risk_return_scaled = scaler_rr.fit_transform(risk_return_features)

# Elbow method and silhouette scores
inertias_rr = []
silhouette_scores = []
k_range = range(2, 8)
for k in k_range:
    kmeans_rr = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels_rr = kmeans_rr.fit_predict(risk_return_scaled)
    inertias_rr.append(kmeans_rr.inertia_)
    silhouette_scores.append(silhouette_score(risk_return_scaled, labels_rr))

# Plot Elbow Method
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(list(k_range), inertias_rr, marker='o')
plt.title('Elbow Method for K (Beta & 1yr_Performance)')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia')
plt.grid(True, alpha=0.3)

# Plot Silhouette Scores
plt.subplot(1,2,2)
plt.plot(list(k_range), silhouette_scores, marker='o', color='orange')
plt.title('Silhouette Score for K (Beta & 1yr_Performance)')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Silhouette Score')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# use k=5 for clustering
k_rr = 5
print(f"\n{'='*30}\nUsing {k_rr} clusters for risk-return analysis\n{'='*30}")
kmeans_rr = KMeans(n_clusters=k_rr, random_state=42, n_init='auto')
labels_rr = kmeans_rr.fit_predict(risk_return_scaled)
cluster_data[f'risk_return_cluster_{k_rr}'] = labels_rr

# Remove outlier clusters 
cluster_counts = cluster_data[f'risk_return_cluster_{k_rr}'].value_counts()
outlier_clusters = cluster_counts[cluster_counts < 3].index.tolist()
if outlier_clusters:
    print(f"Removing outlier clusters: {outlier_clusters} (less than 3 members)")
    filtered = cluster_data[~cluster_data[f'risk_return_cluster_{k_rr}'].isin(outlier_clusters)].reset_index(drop=True)
    risk_return_features_f = filtered[["Beta", "1yr_Performance"]].copy()
    risk_return_scaled_f = scaler_rr.fit_transform(risk_return_features_f)
    kmeans_rr = KMeans(n_clusters=k_rr, random_state=42, n_init='auto')
    labels_rr = kmeans_rr.fit_predict(risk_return_scaled_f)
    filtered[f'risk_return_cluster_{k_rr}'] = labels_rr
else:
    filtered = cluster_data.copy()

# visualization
plt.figure(figsize=(10,7))
sns.scatterplot(
    x=filtered['Beta'],
    y=filtered['1yr_Performance'],
    hue=filtered[f'risk_return_cluster_{k_rr}'],
    palette='Set2',
    alpha=0.7,
    s=60
)
plt.title(f'K-Means Clusters (k={k_rr}): Risk (Beta) vs Return (1yr_Performance)', fontsize=15)
plt.xlabel('Beta (Risk)', fontsize=12)
plt.ylabel('1-Year Performance (Return)', fontsize=12)
plt.legend(title='Cluster', fontsize=10)
plt.grid(True, alpha=0.3)
plt.show()

# Cluster explanation
print(f"\nCLUSTER FEATURE EXPLANATION (k={k_rr}):")
summary_rows = []
for cluster_id in sorted(pd.unique(filtered[f'risk_return_cluster_{k_rr}'])):
    cluster_companies = filtered[filtered[f'risk_return_cluster_{k_rr}'] == cluster_id]
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
        'K': k_rr,
        'Cluster': cluster_id,
        'N': len(cluster_companies),
        'Risk': risk_level,
        'Return': return_level
    })
# Print summary
print(f"\nSummary Table for K={k_rr}:")
print(f"{'K':<3} {'Cluster':<7} {'N':<5} {'Risk':<18} {'Return':<20}")
for row in summary_rows:
    print(f"{row['K']:<3} {row['Cluster']:<7} {row['N']:<5} {row['Risk']:<18} {row['Return']:<20}")

# Save
output_filename = "clustered_esg_stock_data.csv"
cluster_data.to_csv(output_filename, index=False)
print(f"\nClustered dataset saved as: {output_filename}")

def recommendation(user_beta, user_performance, e_weight, s_weight, g_weight):
    """
    Assign user to a cluster and recommend stocks sorted by ESG fields in order of user weight (descending).
    If two or more weights are equal and the highest, only the first highest field is used for sorting.
    If all three weights are equal, sort by 'environment_score' only.
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
    # tandardize 
    user_features = scaler_rr.transform([[user_beta, user_performance]])
    # Predict
    user_cluster = kmeans_rr.predict(user_features)[0]
    print(f"User assigned to cluster {user_cluster}")
    user_cluster_stocks = filtered[filtered[f'risk_return_cluster_{k_rr}'] == user_cluster].copy()
    # Sort 
    user_cluster_stocks = user_cluster_stocks.sort_values(by=sorted_fields, ascending=[False]*len(sorted_fields))
    # Recommend
    top_recommend = user_cluster_stocks.head(10)
    print("Top 10 recommended stocks:")
    print(top_recommend[['ticker', 'name', 'Beta', '1yr_Performance', 'environment_score', 'social_score', 'governance_score']])
    # save
    top_recommend.to_csv('user_recommended_stocks.csv', index=False)
    print("Top 10 recommended stocks have been saved to 'user_recommended_stocks.csv'.")
    return top_recommend

if __name__ == "__main__":
    user_beta = 1.1
    user_performance = 0.15
    e_weight = 0.3
    s_weight = 0.5
    g_weight = 0.2
    recommendation(user_beta, user_performance, e_weight, s_weight, g_weight)
