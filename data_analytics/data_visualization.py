import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np

# Load the data from a CSV file
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Visualize the data using Matplotlib
def visualize_data_matplotlib(data, x_axis, y_axis):
    plt.figure(figsize=(10, 6))
    plt.plot(data[x_axis], data[y_axis])
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(f"{x_axis} vs {y_axis}")
    plt.show()

# Visualize the data using Seaborn
def visualize_data_seaborn(data, x_axis, y_axis):
    sns.set(style="whitegrid")
    sns.lmplot(x=x_axis, y=y_axis, data=data)
    plt.show()

# Visualize the data using Plotly
def visualize_data_plotly(data, x_axis, y_axis):
    fig = px.scatter(data, x=x_axis, y=y_axis)
    fig.show()

# Example usage
if __name__ == "__main__":
    data = load_data("data.csv")
    visualize_data_matplotlib(data, "column1", "column2")
    visualize_data_seaborn(data, "column1", "column2")
    visualize_data_plotly(data, "column1", "column2")
