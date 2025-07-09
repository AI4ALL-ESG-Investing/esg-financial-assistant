from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler

# features
cluster_data = stock_final.dropna(subset=["composite_score"]).copy()
features = cluster_data[["composite_score"]]

# standardise
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# assume 3 clusters: can be adjusted
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
cluster_labels = kmeans.fit_predict(features_scaled)

# label
cluster_data["cluster"] = cluster_labels

# visualisation
plt.figure(figsize=(10,6))
sns.scatterplot(data=cluster_data, x="ESG Score", y="6mo_Performance", hue="cluster", palette="Set2")
plt.title("K-Means Clustering of Stocks")
plt.xlabel("ESG Score")
plt.ylabel("6-Month Performance")
plt.legend(title="Cluster")
plt.show()

# show cluster center
centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=features.columns)
print("Cluster centers:", centers)
