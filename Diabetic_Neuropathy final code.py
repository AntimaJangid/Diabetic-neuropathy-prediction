#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F


from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from tab_transformer_pytorch import TabTransformer

import sys
sys.path.append(r"C:\Users\pavan\anaconda3v\node")

from node_lib.arch import NeuralObliviousDecisionEnsembleModel as NODE

print("✅ NODE imported successfully!")
model = NODE(input_dim=10)
print(model)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


# In[2]:


import os
base = r"C:\Users\pavan\anaconda3v\node\node_lib"
for root, _, files in os.walk(base):
    for f in files:
        if f.endswith(".py"):
            path = os.path.join(root, f)
            with open(path, "r", encoding="utf-8", errors="ignore") as fp:
                text = fp.read()
                if "class NeuralObliviousDecisionEnsembleModel" in text:
                    print("✅ Found in:", path)


# In[3]:


sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

#Loading Dataset
file_name = r'C:\Users\pavan\Downloads\diabetic nueropathy prediction\KeytoDatasetNeuropathy.xlsx'

try:
    df = pd.read_excel(file_name, header=1)
    print(f"Data Loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
except Exception as e:
    raise SystemExit(f"Error loading file: {e}")


# In[4]:


#Preprocessing
df.columns = df.columns.str.replace(r'[^\w\s]', '', regex=True).str.replace(r'\s+', '_', regex=True)

# Target: Neuropathy_Present = 1 if NDS > 0 else 0
df['Neuropathy_Present'] = (df['NDS'] > 0).astype(int)

# Feature selection
features = ['Age_baseline', 'BMI_baseline', 'HbA1c_baseline',
            'IENFD_baseline', 'amp_sur', 'cv_sur', 'cv_per']
target = 'Neuropathy_Present'

X = df[features].copy()
y = df[target]

# Drop missing values
X = X.dropna()
y = y.loc[X.index]

print(f"\nFinal Data Shape: {X.shape[0]} samples")
print("Target Class Distribution:\n", y.value_counts().rename({0: 'No Neuro', 1: 'Neuro'}))


# In[5]:


#feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)


# In[6]:


from sklearn.base import BaseEstimator, ClassifierMixin
import torch
import torch.nn as nn
import numpy as np

class SklearnNODE(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, epochs=20, lr=1e-3, device='cpu'):
        self.input_dim = input_dim
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.model = NODE(input_dim).to(self.device)

    def fit(self, X, y):
        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()

        self.model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            output = self.model(X_tensor)
            loss = loss_fn(output, y_tensor)
            loss.backward()
            optimizer.step()
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X.values, dtype=torch.float32).to(self.device)
            out = self.model(X_tensor)
            return (torch.sigmoid(out).cpu().numpy() > 0.5).astype(int)

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X.values, dtype=torch.float32).to(self.device)
            out = torch.sigmoid(self.model(X_tensor)).cpu().numpy()
            return np.hstack([1 - out, out])


# In[8]:


from sklearn.model_selection import train_test_split
# --- Split Data ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df, y, test_size=0.2, random_state=42, stratify=y
)

# --- Apply SMOTE only on training data ---
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("Before SMOTE:", y_train.value_counts().to_dict())
print("After SMOTE:", y_train_res.value_counts().to_dict())


# In[9]:


node_sklearn = SklearnNODE(input_dim=X.shape[1], epochs=50, lr=1e-3)
from lightgbm import LGBMClassifier

lgb_params = dict(
    n_estimators=100,
    learning_rate=0.05,
    num_leaves=15,
    min_data_in_leaf=3,
    min_gain_to_split=0.0,
    verbosity=-1,              # <--- suppress LightGBM internal logs
    random_state=42)
models = {
    "Logistic Regression (LR)": LogisticRegression(random_state=42),
    "K-Nearest Neighbor (KNN)": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree (DT)": DecisionTreeClassifier(random_state=42),
    "Naive Bayes (NB)": GaussianNB(),
    "LightGBM": LGBMClassifier(**lgb_params),
    "Random Forest (RF)": RandomForestClassifier(n_estimators=100, random_state=42),
    "Extreme Gradient Boosting (XGB)": XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1])
    ),
    "NODE": node_sklearn
}


# In[10]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df, y, test_size=0.2, random_state=42, stratify=y
)

# Check data sanity
print("\n🔹 Feature Uniqueness in Training Set:")
print(X_train.nunique())

print("\n🔹 Class Distribution in Training Set:")
print(y_train.value_counts())


# In[17]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# --- Train & Evaluate All Models ---
results = {}

for name, model in models.items():
    print(f"\n{'='*70}")
    print(f"🔹 Training and Evaluating: {name}")
    print(f"{'='*70}")

    try:
        # ✅ Train
        model.fit(X_train_res, y_train_res)

        # ✅ Predict
        y_pred = model.predict(X_test)

        # Handle probabilistic outputs for NODE if needed
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1)

        # ✅ Metrics
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred, target_names=['No Neuropathy', 'Neuropathy'], output_dict=True)

        # Store results
        results[name] = {
            'Accuracy': acc,
            'F1_No_Neuropathy': cr['No Neuropathy']['f1-score'],
            'F1_Neuropathy': cr['Neuropathy']['f1-score']
        }

        # Print
        print(f"✅ Accuracy: {acc:.4f}")
        print("Confusion Matrix:\n", cm)
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Neuropathy', 'Neuropathy']))

    except Exception as e:
        print(f"⚠️ {name} encountered an error: {e}")

# --- Final Summary ---
print("\n" + "="*70)
print("📊 Final Model Comparison Summary")
print("="*70)
for model_name, metrics in results.items():
    print(f"{model_name:30s} | Accuracy: {metrics['Accuracy']:.4f} | F1 (NoNeuro): {metrics['F1_No_Neuropathy']:.3f} | F1 (Neuro): {metrics['F1_Neuropathy']:.3f}")


# In[18]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Convert results dictionary → DataFrame
results_df = pd.DataFrame(results).T  # transpose so models are rows
results_df = results_df.sort_values(by='Accuracy', ascending=False)
print("\n✅ Results DataFrame:")
print(results_df)


# In[19]:


plt.figure(figsize=(10, 5))
sns.barplot(
    x=results_df.index,
    y='Accuracy',
    data=results_df,
    palette='crest'
)
plt.title('Model Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0, 1.0)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[20]:


plt.figure(figsize=(10, 6))
sns.barplot(
    data=results_df.reset_index().melt(
        id_vars='index',
        value_vars=['F1_No_Neuropathy', 'F1_Neuropathy'],
        var_name='Class', value_name='F1-score'
    ),
    x='F1-score', y='index', hue='Class', palette='coolwarm'
)
plt.title('F1-Score Comparison (Both Classes)')
plt.xlabel('F1-score')
plt.ylabel('Model')
plt.xlim(0, 1.0)
plt.legend(title='Class')
plt.tight_layout()
plt.show()


# In[21]:


fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.barplot(data=results_df, x='Accuracy', y=results_df.index, ax=axes[0], palette='viridis')
axes[0].set_title('Accuracy')

sns.barplot(data=results_df, x='F1_No_Neuropathy', y=results_df.index, ax=axes[1], palette='Blues')
axes[1].set_title('F1 (No Neuropathy)')

sns.barplot(data=results_df, x='F1_Neuropathy', y=results_df.index, ax=axes[2], palette='Reds')
axes[2].set_title('F1 (Neuropathy)')

for ax in axes:
    ax.set_xlim(0, 1.0)
    ax.set_xlabel('Score')
    ax.set_ylabel('Model')

plt.tight_layout()
plt.show()


# # Neuropathy Progression Modeling (Using ΔIENFD)

# In[25]:


# --- Step 1. Imports ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


# In[26]:


print(df.columns.tolist())


# In[27]:


# --- Step 2. Prepare Data ---
# Assuming df is your loaded DataFrame

predictors = [
    'NDS', 'NSS', 'amp_sur', 'cv_sur', 'cv_per',
    'heat_right', 'heat_left', 'cold_right', 'cold_left'
]

# Target (dependent variable)
target = 'delta_IENFD_followup_baseline'

# Define X and y for survival / regression modeling
X_ienfd = df[predictors]
y_ienfd = df[target]


# In[28]:


print("X shape:", X_ienfd.shape)
print("y shape:", y_ienfd.shape)
print("\nPreview of X:")
print(X_ienfd.head())

print("\nPreview of target variable:")
print(y_ienfd.head())


# In[29]:


# --- Step 3. Split Data ---
X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
    X_ienfd, y_ienfd, test_size=0.2, random_state=42
)


# In[32]:


# --- Step 4. Train Regression Model ---
# Combine X and y temporarily to align drops
train_data = pd.concat([X_train_i, y_train_i], axis=1)
train_data = train_data.dropna(subset=[y_train_i.name])  # drop rows where target is NaN

# Separate back
y_train_i = train_data[y_train_i.name]
X_train_i = train_data.drop(columns=[y_train_i.name])

# Same for test set
test_data = pd.concat([X_test_i, y_test_i], axis=1)
test_data = test_data.dropna(subset=[y_test_i.name])

y_test_i = test_data[y_test_i.name]
X_test_i = test_data.drop(columns=[y_test_i.name])

# Train model again
rf_reg = RandomForestRegressor(n_estimators=200, random_state=42)
rf_reg.fit(X_train_i, y_train_i)


# In[33]:


# --- Step 5. Evaluate Model ---
y_pred_i = rf_reg.predict(X_test_i)

mae = mean_absolute_error(y_test_i, y_pred_i)
r2 = r2_score(y_test_i, y_pred_i)

print(f"Mean Absolute Error: {mae:.3f}")
print(f"R² Score: {r2:.3f}")


# In[35]:


# --- Step 6. Feature Importance ---
importances = pd.Series(rf_reg.feature_importances_, index=X_ienfd.columns)
importances = importances.sort_values(ascending=True)

plt.figure(figsize=(5,3))
importances.plot(kind='barh', color='teal')
plt.title("Feature Importance for Nerve Fiber Loss (ΔIENFD)")
plt.xlabel("Importance Score")
plt.ylabel("Clinical Feature")
plt.show()


# # waste code down one

# In[ ]:





# In[13]:


#5-Fold Stratified Cross-Validation with SMOTE
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results_cv = {}

print("\n" + "="*70)
print("  5-FOLD STRATIFIED CROSS-VALIDATION RESULTS (With SMOTE + Class Weights)")
print("="*70)

for name, model in models.items():
    # Pipeline: SMOTE applied inside each CV fold
    pipe = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('model', model)
    ])
    y_pred_cv = cross_val_predict(pipe, X_scaled_df, y, cv=skf)
    report = classification_report(y, y_pred_cv, target_names=['No Neuropathy', 'Neuropathy'], output_dict=True)

    f1_0 = report['No Neuropathy']['f1-score']
    f1_1 = report['Neuropathy']['f1-score']
    # Convert NODE output to 1D labels
if y_pred_cv.ndim > 1 and y_pred_cv.shape[1] > 1:
    y_pred_labels = np.argmax(y_pred_cv, axis=1)
else:
    y_pred_labels = (y_pred_cv > 0.5).astype(int).flatten()

acc = (y.values == y_pred_labels).mean()

results_cv[name] = {
        'Accuracy': acc,
        'F1_Class_0 (No Neuro)': f1_0,
        'F1_Class_1 (Neuropathy)': f1_1
}

print(f"\n--- {name} ---")
print(classification_report(y, y_pred_cv, target_names=['No Neuropathy', 'Neuropathy']))


# In[15]:


results_cv_df = pd.DataFrame(results_cv).T
results_cv_df = results_cv_df[['Accuracy', 'F1_Class_0 (No Neuro)', 'F1_Class_1 (Neuropathy)']]

print("\n" + "="*70)
print("  SUMMARY OF MODEL PERFORMANCE (5-Fold CV)")
print("="*70)
print(results_cv_df.sort_values(by='F1_Class_1 (Neuropathy)', ascending=False).to_string(float_format='%.4f'))
print("="*70)


# In[16]:


plt.figure(figsize=(10, 6))
sns.barplot(
    y=results_cv_df.index,
    x='F1_Class_1 (Neuropathy)',
    data=results_cv_df.sort_values(by='F1_Class_1 (Neuropathy)', ascending=False),
    palette='Spectral'
)
plt.title('5-Fold CV F1-Score Comparison (Neuropathy Class)')
plt.xlabel('F1-score (Neuropathy)')
plt.ylabel('Model')
plt.xlim(0, 1.0)
plt.show()


# In[26]:


best_model_name = results_cv_df['F1_Class_1 (Neuropathy)'].idxmax()
best_f1_score = results_cv_df.loc[best_model_name, 'F1_Class_1 (Neuropathy)']

print(f"\nBest Performing Model (based on Neuropathy F1): {best_model_name}")
print(f"Best F1-score for Neuropathy (Class 1): {best_f1_score:.4f}")


# In[ ]:




