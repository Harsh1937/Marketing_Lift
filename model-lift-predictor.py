#!/usr/bin/env python
# coding: utf-8

# In[15]:


# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklift.models import ClassTransformation
from sklift.metrics import uplift_at_k, qini_auc_score
from sklift.metrics import uplift_at_k, qini_auc_score


# Load Data
df = pd.read_csv("/Users/harshshah/Desktop/Automation/GIT/Online_Retail.csv", encoding="ISO-8859-1")
# Drop rows with missing CustomerID or Description
df.dropna(subset=['CustomerID', 'Description'], inplace=True)

# Feature Engineering
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
df['Hour'] = df['InvoiceDate'].dt.hour

# Simulate treatment assignment for uplift modeling
df['treatment'] = np.random.binomial(1, 0.5, size=len(df))  # Random 0 or 1

# Simulate response: 1 if TotalPrice > threshold (e.g., 20), else 0
df['target'] = (df['TotalPrice'] > 20).astype(int)

# Aggregate to customer level
agg_df = df.groupby('CustomerID').agg({
    'treatment': 'max',
    'target': 'max',
    'TotalPrice': 'sum',
    'Quantity': 'sum',
    'InvoiceNo': 'nunique',
    'Hour': 'mean'
}).reset_index()

# Rename columns
agg_df.rename(columns={'InvoiceNo': 'NumPurchases', 'Hour': 'AvgHour'}, inplace=True)

# Define X, y, treatment
y = agg_df['target']
treatment = agg_df['treatment']
X = agg_df.drop(columns=['CustomerID', 'target', 'treatment'])

# Split Data
X_train, X_test, y_train, y_test, treat_train, treat_test = train_test_split(
    X, y, treatment, test_size=0.3, random_state=42, stratify=y
)

# Fit Model
ct = ClassTransformation(RandomForestClassifier(n_estimators=100, random_state=42))
ct.fit(X_train, y_train, treat_train)

# Predict uplift
uplift_preds = ct.predict(X_test)
auc_score = roc_auc_score(y_test, uplift_preds)

print(f"AUC Score: {auc_score:.2f}")

# Evaluate uplift metrics
qini_score = qini_auc_score(y_true=y_test, uplift=uplift_preds, treatment=treat_test)
print(f"Qini AUC Score: {qini_score:.2f}")


# In[ ]:




