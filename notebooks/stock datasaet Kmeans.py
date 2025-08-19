from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load data
cluster_data = pd.read_csv("cleaned_esg_stock_data.csv")
print(f"Dataset loaded: {len(cluster_data)} companies")

# Filter out extreme outliers
print("Filtering extreme outliers using mean ± 2 standard deviations:")

# mean and standard deviation for Beta
beta_mean = cluster_data['Beta'].mean()
beta_std = cluster_data['Beta'].std()
beta_lower = beta_mean - 2 * beta_std
beta_upper = beta_mean + 2 * beta_std

# mean and standard deviation for 1yr_Performance
perf_mean = cluster_data['1yr_Performance'].mean()
perf_std = cluster_data['1yr_Performance'].std()
perf_lower = perf_mean - 2 * perf_std
perf_upper = perf_mean + 2 * perf_std

print(f"Beta: mean={beta_mean:.3f}, std={beta_std:.3f}")
print(f"Beta range (mean ± 2std): {beta_lower:.3f} to {beta_upper:.3f}")
print(f"Performance: mean={perf_mean:.3f}, std={perf_std:.3f}")
print(f"Performance range (mean ± 2std): {perf_lower:.3f} to {perf_upper:.3f}")

# Filter data for clustering 
filtered = cluster_data[
    (cluster_data['Beta'] >= beta_lower) & 
    (cluster_data['Beta'] <= beta_upper) &
    (cluster_data['1yr_Performance'] >= perf_lower) &
    (cluster_data['1yr_Performance'] <= perf_upper)
].copy()

print(f"After outlier filtering: {len(filtered)} companies")
print(f"Removed {len(cluster_data) - len(filtered)} extreme outliers")

# Count outliers
beta_outliers_high = len(cluster_data[cluster_data['Beta'] > beta_upper])
beta_outliers_low = len(cluster_data[cluster_data['Beta'] < beta_lower])
perf_outliers_high = len(cluster_data[cluster_data['1yr_Performance'] > perf_upper])
perf_outliers_low = len(cluster_data[cluster_data['1yr_Performance'] < perf_lower])

print(f"Beta outliers (> {beta_upper:.2f}): {beta_outliers_high} companies")
print(f"Beta outliers (< {beta_lower:.2f}): {beta_outliers_low} companies")
print(f"Performance outliers (> {perf_upper:.2f}): {perf_outliers_high} companies")
print(f"Performance outliers (< {perf_lower:.2f}): {perf_outliers_low} companies")

# Prepare data
features = ["Beta", "1yr_Performance"]
scaler_rr = StandardScaler()
scaled = scaler_rr.fit_transform(filtered[features])

def find_optimal_k_bic(scaled_data):
    """
    Find optimal number of clusters using BIC for k=3, 4, 5
    """
    bic_scores = []
    k_range = [3, 4, 5]
    
    print("Calculating BIC scores for k=3, 4, 5...")
    for k in k_range:
        gmm = GaussianMixture(n_components=k, random_state=42, n_init=10)
        gmm.fit(scaled_data)
        bic_scores.append(gmm.bic(scaled_data))
        print(f"k={k}: BIC={bic_scores[-1]:.2f}")
    
    return k_range, bic_scores

k_range, bic_scores = find_optimal_k_bic(scaled)

# Use k=5
n_components_final = 5
gmm_final = GaussianMixture(n_components=n_components_final, random_state=42)
labels_final = gmm_final.fit_predict(scaled)
final_cluster_col = f'final_cluster_{n_components_final}'
filtered[final_cluster_col] = labels_final
final_n = n_components_final

# Create improved visualization
plt.style.use('seaborn-v0_8')  # Use a modern style
fig, ax = plt.subplots(figsize=(14, 10))

# Use a more attractive color palette
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
palette = sns.color_palette(colors)

# Main scatter plot with enhanced styling
scatter = sns.scatterplot(
    x=filtered['Beta'],
    y=filtered['1yr_Performance'],
    hue=filtered[final_cluster_col],
    palette=palette,
    alpha=0.8,
    s=80,
    edgecolors='white',
    linewidth=0.5
)

# Add cluster centers with enhanced styling
centers = gmm_final.means_
centers_unscaled = scaler_rr.inverse_transform(centers)
ax.scatter(centers_unscaled[:, 0], centers_unscaled[:, 1], 
           c='red', marker='*', s=300, linewidths=2, 
           edgecolors='black', label='Cluster Centers', zorder=5)

# Add cluster annotations
for i, (x, y) in enumerate(centers_unscaled):
    cluster_size = len(filtered[filtered[final_cluster_col] == i])
    ax.annotate(f'Cluster {i}\n({cluster_size} stocks)', 
                xy=(x, y), xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'),
                fontsize=10, fontweight='bold', ha='left', va='bottom')

# Enhanced title and labels
plt.title(f'Gaussian Mixture Model Clustering: Risk vs Return Analysis\n{n_components_final} Clusters • {len(filtered)} Companies', 
          fontsize=18, fontweight='bold', pad=30, color='#2C3E50')
plt.xlabel('Beta (Systematic Risk)', fontsize=14, fontweight='bold', color='#34495E')
plt.ylabel('1-Year Performance (Return)', fontsize=14, fontweight='bold', color='#34495E')

# Enhanced legend
legend = plt.legend(title='Investment Clusters', fontsize=12, 
                   title_fontsize=13, bbox_to_anchor=(1.02, 1), loc='upper left')
legend.get_title().set_fontweight('bold')

# Enhanced grid
plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Add risk/return quadrants
ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=1)
ax.axvline(x=1, color='gray', linestyle='-', alpha=0.5, linewidth=1)

# Add quadrant labels
ax.text(0.02, 0.98, 'Low Risk\nLow Return', transform=ax.transAxes, 
        fontsize=10, verticalalignment='top', 
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
ax.text(0.98, 0.98, 'High Risk\nLow Return', transform=ax.transAxes, 
        fontsize=10, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))
ax.text(0.02, 0.02, 'Low Risk\nHigh Return', transform=ax.transAxes, 
        fontsize=10, verticalalignment='bottom',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
ax.text(0.98, 0.02, 'High Risk\nHigh Return', transform=ax.transAxes, 
        fontsize=10, verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))

# Enhanced outlier info box
outlier_text = f"Outliers Removed:\n• Beta > {beta_upper:.2f}: {beta_outliers_high} companies\n• Beta < {beta_lower:.2f}: {beta_outliers_low} companies\n• Performance > {perf_upper:.2f}: {perf_outliers_high} companies\n• Performance < {perf_lower:.2f}: {perf_outliers_low} companies\n\nTotal: {len(cluster_data) - len(filtered)} companies removed"
ax.text(0.02, 0.15, outlier_text, transform=ax.transAxes, 
        verticalalignment='top', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9FA', alpha=0.9, 
                 edgecolor='#DEE2E6', linewidth=1))

# Add data source and method info
ax.text(0.02, 0.02, f"Method: GMM Clustering (k={n_components_final})\nOutlier Filter: Mean ± 2σ", 
        transform=ax.transAxes, fontsize=8, style='italic', color='gray',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Set axis limits with some padding
x_min, x_max = filtered['Beta'].min(), filtered['Beta'].max()
y_min, y_max = filtered['1yr_Performance'].min(), filtered['1yr_Performance'].max()
x_padding = (x_max - x_min) * 0.05
y_padding = (y_max - y_min) * 0.05
plt.xlim(x_min - x_padding, x_max + x_padding)
plt.ylim(y_min - y_padding, y_max + y_padding)

# Adjust layout to prevent text cutoff
plt.tight_layout()
plt.subplots_adjust(right=0.85)  # Make room for legend

# Add a subtle background color
ax.set_facecolor('#F8F9FA')

plt.show()

# Cluster explanation
print(f"\nCLUSTER FEATURE EXPLANATION (n_components={final_n}):")
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
        'N_Components': final_n,
        'Cluster': cluster_id,
        'N': len(cluster_companies),
        'Risk': risk_level,
        'Return': return_level
    })

print(f"\nSummary Table for n_components={final_n}:")
print(f"{'N_Comp':<7} {'Cluster':<7} {'N':<5} {'Risk':<18} {'Return':<20}")
for row in summary_rows:
    print(f"{row['N_Components']:<7} {row['Cluster']:<7} {row['N']:<5} {row['Risk']:<18} {row['Return']:<20}")

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
        print(f"Sorting order: {sorted_fields}")
    # Standardize 
    user_features_df = pd.DataFrame([[user_beta, user_performance]], columns=features)
    user_features = scaler_rr.transform(user_features_df)
    # Predict
    user_cluster = gmm_final.predict(user_features)[0]
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
