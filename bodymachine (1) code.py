# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
df = pd.read_csv(r'D:\New folder (2)\fraudTest.csv')
print(df.shape)

# %%
df.head()

# %%
df.sample(5)

# %%
df.shape

# %%
df.info()

# %%
df.columns

# %%
df.describe().T

# %%
df.nunique()

# %%
df['is_fraud'].value_counts()

# %%
df['is_fraud'].value_counts(normalize=True)*100

# %%
df.isna().sum()

# %%
df.dropna(inplace=True)
df.isna().sum()

# %%
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
SimpleImputer(strategy="constant", fill_value=0)

# %%
print(f' duplicated = {df.duplicated().sum()}')
df[df.duplicated()]

# %%
df = df.drop_duplicates()
print(f' duplicated = {df.duplicated().sum()}')
df[df.duplicated()]

# %%
df.columns = df.columns.str.strip().str.lower()

# %%
df.dtypes

# %%
# Remove negative amounts
df = df[df['amt'] >= 0]

# Remove invalid city population
df = df[df['city_pop'] > 0]

# Fix latitude & longitude
df = df[(df['lat'].between(-90, 90))]
df = df[(df['long'].between(-180, 180))]
df = df[(df['merch_lat'].between(-90, 90))]
df = df[(df['merch_long'].between(-180, 180))]

print("Shape after removing invalid values:", df.shape)

# %%
cat_cols = df.select_dtypes(include='object').columns

for col in cat_cols:
    df[col] = df[col].str.strip().str.lower()

# %%
# Fix column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Categorical cleaning
cat_cols = df.select_dtypes(include='object').columns

for col in cat_cols:
    df[col] = df[col].astype(str)
    df[col] = df[col].str.strip()
    df[col] = df[col].str.lower()

# %%
# Convert unix_time to datetime
df['transaction_time'] = pd.to_datetime(df['unix_time'], unit='s')

# %%
num_cols = df.select_dtypes(include=['number']).columns

Q1 = df[num_cols].quantile(0.25)
Q3 = df[num_cols].quantile(0.75)
IQR = Q3 - Q1

outliers = ((df[num_cols] < (Q1 - 1.5 * IQR)) | 
            (df[num_cols] > (Q3 + 1.5 * IQR)))

df_outliers = df[outliers.any(axis=1)]

print("Number of outliers:", df_outliers.shape[0])

# %%
from numpy.random.mtrand import f
num_cols = df.select_dtypes(include=["number"]).columns.tolist()
plt.figure(figsize=(15, 10))
for i, col in enumerate(num_cols):
    plt.subplot(7, 2, i+1)
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()

# %%
import matplotlib.pyplot as plt
import seaborn as sns

num_cols = df.select_dtypes(include=['number']).columns

for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

# %%
cat_cols = df.select_dtypes(include='object').columns

cat_cols_filtered = [col for col in cat_cols if df[col].nunique() < 10]
for col in cat_cols_filtered:
    plt.figure(figsize=(6,4))
    sns.countplot(x=df[col])
    plt.title(f'Count Plot of {col}')
    plt.xticks(rotation=45)
    plt.show()

# %%
print("Skewness:\n", df[num_cols].skew())
print("\nKurtosis:\n", df[num_cols].kurt())

# %%
import numpy as np

skewed_cols = df[num_cols].skew()
skewed_cols = skewed_cols[skewed_cols > 1].index

for col in skewed_cols:
    df[col] = np.log1p(df[col])

# %%
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# %%
num_cols = df.select_dtypes(include=['number']).columns

for i in range(len(num_cols)):
    for j in range(i+1, len(num_cols)):
        plt.figure(figsize=(5,4))
        sns.scatterplot(x=df[num_cols[i]], y=df[num_cols[j]])
        plt.title(f"{num_cols[i]} vs {num_cols[j]}")
        plt.show()

# %%
for col in cat_cols_filtered:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col], y=df['amt'])
    plt.title(f'{col} vs amt')
    plt.xticks(rotation=45)
    plt.show()

# %%
for col in cat_cols_filtered:
    plt.figure(figsize=(6,4))
    sns.barplot(x=df[col], y=df['is_fraud'])
    plt.title(f'{col} vs Fraud')
    plt.xticks(rotation=45)
    plt.show()

# %%
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

x = df['amt']
y = df['is_fraud']
z = df['lat']

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x, y, z)

ax.set_xlabel('amt')
ax.set_ylabel('is_fraud')
ax.set_zlabel('lat')

plt.title("3D Scatter Plot")
plt.show()

# %%
import sys
!{sys.executable} -m pip install statsmodels

# %%
from statsmodels.stats.outliers_influence import variance_inflation_factor

# %%
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

num_cols = df.select_dtypes(include=['number']).columns

X_vif = df[num_cols].dropna()

vif_data = pd.DataFrame()
vif_data["feature"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) 
                   for i in range(len(X_vif.columns))]

print(vif_data)

# %%
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[num_cols])

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Explained Variance:", pca.explained_variance_ratio_)

# %%
plt.scatter(X_pca[:,0], X_pca[:,1])
plt.title("PCA Plot")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# %%
plt.figure(figsize=(15, 10))
for i, col in enumerate(num_cols):
    plt.subplot(5, 3, i+1)
    sns.histplot(df[col], kde=True, bins=30, color="skyblue")
    plt.title(f'distribution of {col}')
plt.tight_layout()
plt.show()

# %%
from scipy import stats
num_cols = df.select_dtypes(include=["number"]).columns.tolist()
plt.figure(figsize=(15, 10))
for i, col in enumerate(num_cols):
    plt.subplot(5, 3, i+1)
    z_scores = stats.zscore(df[col])
    sns.scatterplot(x=np.arange(len(z_scores)), y=z_scores)
    plt.axhline(y=3, color='r', linestyle='--')
    plt.axhline(y=-3, color='r', linestyle='--')
    plt.title(f'Z-score of {col}')
    plt.xlabel('Index')
    plt.ylabel('Z-score')
plt.tight_layout()
plt.show()

# %%
continuous_cols = df.select_dtypes(include=["number"]).columns.tolist()
df_no_outliers = df.copy()
for col in continuous_cols:
  Q1 = df_no_outliers[col].quantile(0.25)
  Q3 = df_no_outliers[col].quantile(0.75)
  IQR = Q3 - Q1
  lower = Q1 - 1.5 * IQR
  upper = Q3 + 1.5 * IQR
  df_no_outliers = df_no_outliers[(df_no_outliers[col] >= lower) & (df_no_outliers[col] <= upper)]
  plt.figure(figsize=(15, 10))
  for i, col in enumerate(continuous_cols):
    plt.subplot(len(continuous_cols), 2, 2*i+1)
    sns.boxplot(x=df[col])
    plt.title(f' before: {col}')
    plt.subplot(len(continuous_cols), 2, 2*i+2)
    sns.boxplot(x=df_no_outliers[col])
    plt.title(f' after: {col}')
  plt.tight_layout()
  plt.show()
  print("original data shape:",df.shape)
  print("data shape after removing outliers:",df_no_outliers.shape)


# %%
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
sns.countplot(x='is_fraud', data=df)
plt.title("Distribution of Fraud vs Normal Transactions")
plt.show()


# %%
plt.figure(figsize=(12,8))
numeric_df = df.select_dtypes(include=['number'])
sns.heatmap(numeric_df.corr(), cmap='coolwarm', linewidths=0.1)
plt.title("Correlation Heatmap")
plt.show()

# %%
df.dtypes
df.nunique().sort_values(ascending=False)

# %%
drop_cols = []

for col in df.columns:
    if df[col].dtype == 'object' and df[col].nunique() > 50:
        print("Dropping:", col)
        drop_cols.append(col)

df.drop(columns=drop_cols, inplace=True)

# %%
id_cols = ['transaction_id', 'name', 'merchant']

df.drop(columns=[col for col in id_cols if col in df.columns], inplace=True)

# %%
df = pd.get_dummies(df, columns=[
    'category',
    'gender',
    'state',
], drop_first=True)

# %%
X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]


# %%
df = df.drop(columns=[
    'unnamed:_0',
    'trans_num',
    'street',
    'first',
    'last',
    'dob',
    'trans_date_trans_time',
    'transaction_time'
])

# %%
drop_cols = [
    'trans_date_trans_time',
    'first',
    'last',
    'street'
]

X = X.drop(columns=drop_cols)

# %%
# Outlier Treatment باستخدام IQR Clipping

num_cols = df.select_dtypes(include=['number']).columns

df_clipped = df.copy()

for col in num_cols:
    Q1 = df_clipped[col].quantile(0.25)
    Q3 = df_clipped[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df_clipped[col] = df_clipped[col].clip(lower, upper)

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# %%
X_train.dtypes

# %%
X_train.select_dtypes(include='object').columns

# %%
X_train = X_train.drop(columns=['dob', 'trans_num'])
X_test = X_test.drop(columns=['dob', 'trans_num'])

# %%
print(X_train.shape)

# %%
num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
bool_cols = X_train.select_dtypes(include=['bool']).columns

# %%
num_cols = [col for col in num_cols if col not in bool_cols]

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# %%
X_train = X_train.select_dtypes(include=['number', 'bool'])
X_test = X_test.select_dtypes(include=['number', 'bool'])

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %%
from sklearn.preprocessing import StandardScaler

# Select only numerical columns from X for scaling
X_numeric = X.select_dtypes(include=['number'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

print("Shape of scaled numerical features:", X_scaled.shape)

# %%
from sklearn.model_selection import train_test_split
import numpy as np
nan_mask = y.isna()
X_scaled_filtered = X_scaled[~nan_mask.values]
y_filtered = y[~nan_mask]
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_filtered, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered
)


# %%
y = df['is_fraud']

# %%
y_train = y_train.apply(lambda x: 1 if x > 0 else 0)
y_test = y_test.apply(lambda x: 1 if x > 0 else 0)

# %%
y_train.value_counts()

# %%
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(X_train, y_train)

# %%
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(class_weight='balanced', max_iter=1000)

# %%
X_clustering = df.drop(columns=['is_fraud'])

# %%
X_clustering = pd.get_dummies(X_clustering, drop_first=True)

# %%
import pandas as pd

X_clustering = X_clustering.apply(pd.to_numeric, errors='coerce')

# %%
X_clustering = X_clustering.fillna(X_clustering.mean())

# %%
print(X_clustering.dtypes)

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clustering)

# %%
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.show()

# %%
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# %%
import numpy as np

sample_size = 2000  # أو 1000 لو تقيل
idx = np.random.choice(len(X_scaled), sample_size, replace=False)

sil_score = silhouette_score(X_scaled[idx], clusters[idx])
print("Silhouette Score:", sil_score)

# %%
from sklearn.metrics import davies_bouldin_score

dbi = davies_bouldin_score(X_scaled, clusters)
print("Davies-Bouldin Index:", dbi)

# %%
from sklearn.metrics import pairwise_distances_argmin_min
X['cluster_id'] = clusters


_, distances = pairwise_distances_argmin_min(
    X_scaled,
    kmeans.cluster_centers_
)


X['distance_to_centroid'] = distances

print(X.head())

# %%
import seaborn as sns

df['Cluster'] = clusters

sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=clusters)
plt.title("Clusters Visualization")
plt.show()

# %%
X_train = X_train.drop(columns=X_train.select_dtypes(include=['datetime', 'datetime64']).columns, errors='ignore')
X_test = X_test.drop(columns=X_test.select_dtypes(include=['datetime', 'datetime64']).columns, errors='ignore')

# %%
X_train = X_train.select_dtypes(include=['number', 'bool'])
X_test = X_test.select_dtypes(include=['number', 'bool'])

# %%
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# %%
import numpy as np

X_train = X_train.to_numpy(dtype=np.float32)
X_test = X_test.to_numpy(dtype=np.float32)

# %%
model.fit(X_train, y_train)

# %%
print(X_train.shape, X_test.shape)

# %%
from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# %%
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print(classification_report(y_test, y_pred_rf))


# %%
from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42
)

xgb.fit(X_train, y_train)


# %%
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

y_pred = xgb.predict(X_test)
y_prob = xgb.predict_proba(X_test)[:,1]

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))


# %%
import numpy as np

threshold = 0.3
y_custom = (y_prob >= threshold).astype(int)

print(confusion_matrix(y_test, y_custom))
print(classification_report(y_test, y_custom))


# %%
import numpy as np
from sklearn.metrics import classification_report

y_probs = xgb.predict_proba(X_test)[:, 1]

for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
    y_pred_custom = (y_probs >= t).astype(int)
    print(f"\nThreshold = {t}")
    print(classification_report(y_test, y_pred_custom))


# %%
from xgboost import XGBClassifier

xgb_weighted = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight= (len(y_train) - sum(y_train)) / sum(y_train),
    eval_metric='logloss',
    random_state=42
)

xgb_weighted.fit(X_train, y_train)


# %%
y_pred = xgb_weighted.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# %%
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


# %%
import joblib
import os

os.makedirs("model", exist_ok=True)

joblib.dump(xgb_weighted, "model/model.pkl")
joblib.dump(scaler, "model/scaler.pkl")



