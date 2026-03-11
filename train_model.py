import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib
data=pd.read_excel("dataset/creditcard_customers.xlsx")
data = data.drop(columns=["CUST_ID"])
data=data[["BALANCE","PURCHASES","CASH_ADVANCE","CREDIT_LIMIT","PAYMENTS"]]
data=data.fillna(data.mean())
scaler=StandardScaler()
X_scaled=scaler.fit_transform(data)
pca=PCA(n_components=2)
X_pca=pca.fit_transform(X_scaled)
kmeans=KMeans(n_clusters=4,random_state=42)
kmeans.fit(X_pca)
joblib.dump((scaler,pca,kmeans),"credit_model.pkl")
print("Model Trained and saved")