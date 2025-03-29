import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
file_path = '/kaggle/input/steam-store-games/steam.csv'
df = pd.read_csv(file_path)

# Select relevant numerical columns
numerical_features = [
    "required_age", "achievements", "positive_ratings", "negative_ratings",
    "average_playtime", "median_playtime", "price"
]
df_numeric = df[numerical_features].copy()  # Create an explicit copy

# Handle missing values
df_numeric.fillna(0, inplace=True)

# Scale the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_numeric)

# Exploratory Data Analysis
plt.figure(figsize=(10, 6))
sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# K-Means Clustering
inertia = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(df_scaled, kmeans.labels_))

# Elbow Method Plot
plt.figure(figsize=(8, 4))
plt.plot(K_range, inertia, marker='o', linestyle='-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.show()

# Silhouette Score Plot
plt.figure(figsize=(8, 4))
plt.plot(K_range, silhouette_scores, marker='s', linestyle='-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores for Different K')
plt.show()

# Choose optimal K and apply KMeans
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(df_scaled)

# Hierarchical Clustering
linked = linkage(df_scaled, method='ward')
plt.figure(figsize=(10, 5))
dendrogram(linked, truncate_mode='lastp', p=10, leaf_rotation=45., leaf_font_size=12.)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()

# Cosine Similarity for Recommendations
user_game_matrix = df_numeric.values
cos_sim_matrix = cosine_similarity(user_game_matrix)

# Recommend top 5 similar games for a sample game (index 0)
target_game_index = 0
similar_games = np.argsort(cos_sim_matrix[target_game_index])[::-1][1:6]
print("Top 5 Similar Games:")
print(df.iloc[similar_games][['appid', 'name']])

# Conclusion
print("\nConclusion: Clustering helped group games based on numerical attributes, and cosine similarity allowed for content-based recommendations.")
