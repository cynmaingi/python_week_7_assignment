# Python Week 7 Assignment: Iris Dataset Analysis

## 📋 Overview  
This project demonstrates a complete data-analysis workflow using the classic Iris dataset. The single script (`task_1.py`) covers:  
1. Loading and exploring data with **pandas**  
2. Cleaning and summarizing numeric features  
3. Grouped statistics by species  
4. Four customized visualizations via **matplotlib** and **seaborn**

---

## 🚀 Features  
- Reads the Iris dataset from `sklearn.datasets`  
- Prints first five rows, data types, missing-value counts  
- Computes summary statistics (`.describe()`)  
- Groups by the `species` column and reports mean petal length  
- Renders:  
  - Line chart of sepal length over sample index  
  - Bar chart of average petal length per species  
  - Histogram of sepal-width distribution  
  - Scatter plot of sepal length vs. petal length, colored by species  

---

## 🛠️ Prerequisites  
- Python 3.7 or newer  
- A terminal or IDE that can display plots (Jupyter Notebook, VSCode, or a desktop GUI)  

---

## 📦 Installation  

1. **Clone this repo**  
   ```bash
   git clone https://github.com/your-username/python_week_7_assignment.git
   cd python_week_7_assignment
