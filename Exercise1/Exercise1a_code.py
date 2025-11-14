from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Load and preprocess the data
data, meta = arff.loadarff('dataset.arff')
df = pd.DataFrame(data)
df = df.drop(columns=['casual', 'registered'])
df = df.map(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)  # FIX
df = pd.get_dummies(df, drop_first=True)

# Extract features and target variable
target_col = 'count'
X = df.drop(columns=[target_col])
y = df[target_col].astype(float)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# PCA cumulative explained variance 
pca_full = PCA().fit(X_train_scaled)
plt.figure(figsize=(6,4))
plt.plot(np.cumsum(pca_full.explained_variance_ratio_), marker='o')
plt.xlabel('Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Cumulative Explained Variance')
plt.grid(True)
plt.show()

# PCA biplot (2D)
pca2 = PCA(n_components=2)
X_pca_scores = pca2.fit_transform(X_train_scaled)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.scatter(X_pca_scores[:,0], X_pca_scores[:,1], c=y_train, cmap='viridis', s=20)
loadings = pca2.components_.T * np.sqrt(pca2.explained_variance_)
for i, feature in enumerate(X.columns):
    plt.arrow(0, 0, loadings[i,0]*3, loadings[i,1]*3,
              color='red', alpha=0.6, head_width=0.05, length_includes_head=True)
    plt.text(loadings[i,0]*3.3, loadings[i,1]*3.3, feature,
             color='red', ha='center', va='center', fontsize=8)
plt.xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)")
plt.title("PCA Biplot")
plt.grid(True)

# t-SNE (2D)
tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
X_tsne = tsne.fit_transform(X_train_scaled)

plt.subplot(1,2,2)
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y_train, cmap='viridis', s=20)
plt.title('t-SNE (2D)')
plt.xlabel('t-SNE1'); plt.ylabel('t-SNE2')

plt.tight_layout()
plt.show()

def score(name, y_true, y_pred):
    print(f"{name:7s}  MSE={mean_squared_error(y_true,y_pred):.4f}  R²={r2_score(y_true,y_pred):.4f}")

# MLP on raw scaled data 
mlp_raw = MLPRegressor(hidden_layer_sizes=(50,30), max_iter=100, random_state=42)
mlp_raw.fit(X_train_scaled, y_train)
score("RAW data", y_test, mlp_raw.predict(X_test_scaled))

# MLP with PCA features 
k = 6  
pca_k = PCA(n_components=k)
Xtr_pca = pca_k.fit_transform(X_train_scaled)
Xte_pca = pca_k.transform(X_test_scaled)

mlp_pca = MLPRegressor(hidden_layer_sizes=(50,30), max_iter=100, random_state=42)
mlp_pca.fit(Xtr_pca, y_train)
score(f"PCA({k})", y_test, mlp_pca.predict(Xte_pca))

# MLP with t-SNE features
X_all = np.vstack([X_train_scaled, X_test_scaled])
Z_all = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto').fit_transform(X_all)
Ztr, Zte = Z_all[:len(X_train_scaled)], Z_all[len(X_train_scaled):]

mlp_tsne = MLPRegressor(hidden_layer_sizes=(50,30), max_iter=100, random_state=42)
mlp_tsne.fit(Ztr, y_train)
score("t-SNE", y_test, mlp_tsne.predict(Zte))


