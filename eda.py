# ================================
# MALL DATA ANALYSIS PROJECT
# ================================

import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
from sklearn.cluster import KMeans

# ================================
# STEP 1: LOAD DATA
# ================================
df = pd.read_csv("mall_customers.csv")

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Dataset Info ---")
print(df.info())

print("\n--- Summary Statistics ---")
print(df.describe())

# ================================
# STEP 2: DATA CLEANING
# ================================
# Rename columns for easier SQL usage
df.columns = df.columns.str.replace(" ", "_").str.replace("(", "").str.replace(")", "")

print("\n--- Updated Columns ---")
print(df.columns)

# ================================
# STEP 3: UNIVARIATE ANALYSIS
# ================================

# Age Distribution
plt.figure()
plt.hist(df['Age'])
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# Gender Distribution
plt.figure()
df['Gender'].value_counts().plot(kind='bar')
plt.title("Gender Distribution")
plt.show()

# Income Distribution
plt.figure()
plt.hist(df['Annual_Income_k$'])
plt.title("Annual Income Distribution")
plt.show()

# ================================
# STEP 4: MULTIVARIATE ANALYSIS
# ================================

# Scatter Plot: Income vs Spending
plt.figure()
plt.scatter(df['Annual_Income_k$'], df['Spending_Score_1-100'])
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Income vs Spending")
plt.show()

# Correlation
print("\n--- Correlation Matrix ---")
print(df.corr(numeric_only=True))

# ================================
# STEP 5: SQL ANALYSIS
# ================================

conn = sqlite3.connect("mall.db")
df.to_sql("customers", conn, if_exists="replace", index=False)

cursor = conn.cursor()

print("\n--- Top 5 Customers by Spending ---")
query1 = """
SELECT CustomerID, Spending_Score_1-100
FROM customers
ORDER BY Spending_Score_1-100 DESC
LIMIT 5;
"""
print(pd.read_sql(query1, conn))

print("\n--- Average Spending by Gender ---")
query2 = """
SELECT Gender, AVG(Spending_Score_1-100) as avg_spending
FROM customers
GROUP BY Gender;
"""
print(pd.read_sql(query2, conn))

print("\n--- Income vs Spending Trend ---")
query3 = """
SELECT Annual_Income_k$, AVG(Spending_Score_1-100)
FROM customers
GROUP BY Annual_Income_k$
ORDER BY Annual_Income_k$;
"""
print(pd.read_sql(query3, conn))

print("\n--- Age Group Analysis ---")
query4 = """
SELECT 
CASE 
    WHEN Age < 25 THEN 'Young'
    WHEN Age BETWEEN 25 AND 40 THEN 'Adult'
    ELSE 'Senior'
END AS Age_Group,
AVG(Spending_Score_1-100)
FROM customers
GROUP BY Age_Group;
"""
print(pd.read_sql(query4, conn))

print("\n--- High Income but Low Spending ---")
query5 = """
SELECT *
FROM customers
WHERE Annual_Income_k$ > 70 AND Spending_Score_1-100 < 40;
"""
print(pd.read_sql(query5, conn))

# ================================
# STEP 6: CUSTOMER SEGMENTATION
# ================================

X = df[['Annual_Income_k$', 'Spending_Score_1-100']]

kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

plt.figure()
plt.scatter(df['Annual_Income_k$'], df['Spending_Score_1-100'], c=df['Cluster'])
plt.xlabel("Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation")
plt.show()

# ================================
# STEP 7: SAVE OUTPUT
# ================================

df.to_csv("processed_mall.csv", index=False)

print("\n✅ Analysis Completed Successfully!")
