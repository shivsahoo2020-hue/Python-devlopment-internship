import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the CSV file
# Replace 'data.csv' with the actual path of your CSV file
df = pd.read_csv(r"C:\Users\Lenovo\Desktop\Python\Internship-1\data.csv")


# Show first 5 rows
print("Data Preview:")
print(df.head())

# 2. Basic Data Analysis
print("\nSummary Statistics:")
print(df.describe())

# Example: calculate average of a selected column (say 'Sales')
if 'Sales' in df.columns:
    avg_sales = df['Sales'].mean()
    print(f"\nAverage Sales: {avg_sales:.2f}")

# 3. Visualizations

# Bar chart (e.g., average sales per category)
if 'Category' in df.columns and 'Sales' in df.columns:
    category_avg = df.groupby('Category')['Sales'].mean()
    category_avg.plot(kind='bar', figsize=(6,4))
    plt.title("Average Sales per Category")
    plt.ylabel("Average Sales")
    plt.show()

# Scatter plot (e.g., Sales vs Profit)
if 'Sales' in df.columns and 'Profit' in df.columns:
    plt.figure(figsize=(6,4))
    plt.scatter(df['Sales'], df['Profit'], alpha=0.5)
    plt.title("Sales vs Profit")
    plt.xlabel("Sales")
    plt.ylabel("Profit")
    plt.show()

# Heatmap (correlation between numeric columns)
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
