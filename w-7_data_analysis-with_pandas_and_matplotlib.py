import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import os

# Load dataset
iris_data = load_iris(as_frame=True)
df = iris_data.frame
df["species"] = df["target"].map(dict(enumerate(iris_data.target_names)))

# Create folder for plots
os.makedirs("plots", exist_ok=True)

sns.set_style("whitegrid")

# Helper function: show + save + auto-close after 3s
def show_and_save(fig, filename):
    fig.savefig(f"plots/{filename}")
    plt.show(block=False)
    plt.pause(3)   # keep window open for 3 seconds
    plt.close(fig)

# 1. Line chart
fig = plt.figure(figsize=(8, 5))
plt.plot(df.index, df["sepal length (cm)"], label="Sepal Length")
plt.plot(df.index, df["petal length (cm)"], label="Petal Length")
plt.title("Line Chart: Sepal vs Petal Length over Index")
plt.xlabel("Index")
plt.ylabel("Length (cm)")
plt.legend()
show_and_save(fig, "line_chart.png")

# 2. Bar chart
fig = plt.figure(figsize=(6, 5))
sns.barplot(x="species", y="petal length (cm)", data=df, estimator="mean", palette="viridis")
plt.title("Bar Chart: Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
show_and_save(fig, "bar_chart.png")

# 3. Histogram
fig = plt.figure(figsize=(6, 5))
plt.hist(df["sepal length (cm)"], bins=15, color="skyblue", edgecolor="black")
plt.title("Histogram: Sepal Length Distribution")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
show_and_save(fig, "histogram.png")

# 4. Scatter plot
fig = plt.figure(figsize=(6, 5))
sns.scatterplot(
    x="sepal length (cm)",
    y="petal length (cm)",
    hue="species",
    data=df,
    palette="Set1"
)
plt.title("Scatter Plot: Sepal vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
show_and_save(fig, "scatter_plot.png")

print("✅ All plots displayed for 3 seconds and saved in 'plots/' folder")
