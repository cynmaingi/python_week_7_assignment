import matplotlib
matplotlib.use('TkAgg')         # ensure interactive mode
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# … your existing code …

# when you call plt.show(), a window will appear


# 1️⃣ Load
try:
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    print("✅ Dataset loaded successfully.\n")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")

# 2️⃣ Inspect
print("🔍 First 5 rows:")
print(df.head())
print("\n📋 Data Types:")
print(df.dtypes)
print("\n🧼 Missing Values:")
print(df.isnull().sum())

df.fillna(df.mean(numeric_only=True), inplace=True)

# 3️⃣ Analysis
print("\n📈 Summary Statistics:")
print(df.describe())
grouped = df.groupby('species')['petal length (cm)'].mean()
print("\n🌼 Average Petal Length per Species:")
print(grouped)

# 4️⃣ Plotting
sns.set(style="whitegrid")

# Line chart (blocking to keep window open in .py scripts)
plt.figure(figsize=(8, 5))
df['index'] = range(len(df))
plt.plot(df['index'], df['sepal length (cm)'], label='Sepal Length')
plt.title("Sepal Length Over Sample Index")
plt.xlabel("Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.show(block=True)    # <--- ensures window stays open in scripts

# Bar chart
plt.figure(figsize=(6, 4))
grouped.plot(kind='bar', color='skyblue')
plt.title("Avg Petal Length per Species")
plt.ylabel("Petal Length (cm)")
plt.xticks(rotation=0)
plt.show()

# Histogram
plt.figure(figsize=(6, 4))
plt.hist(df['sepal width (cm)'], bins=20, color='salmon', edgecolor='black')
plt.title("Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

# Scatter plot
plt.figure(figsize=(6, 6))
sns.scatterplot(
    data=df,
    x='sepal length (cm)',
    y='petal length (cm)',
    hue='species',
    palette='deep'
)
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title='Species')
plt.show()
