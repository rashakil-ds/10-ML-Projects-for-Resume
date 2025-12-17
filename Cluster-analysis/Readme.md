# Cluster Analysis with K-Means (Interactive Streamlit App)

---

### Click -> [APP Link](https://cluster-analyst.streamlit.app/) <br>

---

This project is an interactive implementation of **K-Means clustering** built using **Python and Streamlit**.  It demonstrates how a traditional machine learning notebook can be transformed into a **usable, real-world analytics application**. The app is designed for exploratory data analysis and customer segmentation tasks, allowing users to experiment with clustering parameters and visually understand clustering behavior.

---

Most clustering examples remain limited to Jupyter notebooks. This project focuses on converting clustering logic into a **user-friendly application** that can be used by analysts, students, and non-technical users without modifying code.
The emphasis is on:
- Interactivity
- Explainability
- Practical usability

---

## What This Application Does

The application allows users to perform K-Means clustering with the following capabilities:

- Load a dataset or use a built-in sample dataset
- Select numeric features dynamically
- Explore different values of the number of clusters
- Automatically detect the optimal number of clusters
- Visualize clusters and centroids
- Download clustered results for further analysis

---

## Key Features

### Dataset Handling
- Upload any CSV file
- Use a sample dataset resembling customer income and spending behavior
- Preview the dataset directly in the interface
- Automatic detection of numeric columns

### Feature Selection
- Select any `two numeric features` for clustering
- No hardcoded column names
- Works with different datasets without modification

### Model Configuration
- Interactive slider to control the number of clusters
- Optional feature standardization using [`StandardScaler`](https://scikit-learn.org/0.22/modules/generated/sklearn.preprocessing.StandardScaler.html)
- Configurable random state for reproducibility

### Elbow and Knee Analysis
The application provides two complementary visual tools to select the number of clusters:

- **Elbow Curve**
  - Displays inertia versus the number of clusters
  - Helps identify diminishing returns as clusters increase

- **Knee Plot**
  - Automatically detects the optimal number of clusters using the knee (elbow) detection algorithm
  - Displays the recommended cluster count
  - Allows applying the optimal value directly to the model

### Cluster Visualization
- Scatter plot of clustered data
- Clear separation of clusters
- Centroids displayed explicitly
- Dynamic updates when parameters change

### Cluster Insights
- Table showing centroid values for each cluster
- Table showing the number of samples per cluster

### Export Functionality
- Download the final clustered dataset as a CSV file
- Cluster labels included for downstream analysis

---

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- kneed

---

## Project Structure
. <br>
├── app.py <br>
├── requirements.txt <br>
├── mall customers.csv <br>
├── Market Basket Analysis using K-Means Cluster Algorithm.ipynb <br>
├── README.md <br>
└── .gitignore

---

## How to Run the Application

### Clone the Repository
git clone https://github.com/rashakil-ds/cluster-analysis-with-kmeans.git <br>
cd cluster-analysis-with-kmeans

### Install Dependencies
pip install -r requirements.txt

### Run the App
streamlit run app.py

---

## Design Decisions

- The K-Means model is trained dynamically inside the application
- The model is not saved intentionally, as clustering depends on user-selected parameters and data
- Streamlit session state is used to manage interactive behavior correctly
- The application prioritizes clarity and explainability over complexity

---

## Possible Extensions

- Add silhouette score or Davies–Bouldin index for cluster evaluation
- Support clustering with more than two features
- Deploy the application on Streamlit Cloud
- Add authentication for multi-user environments

---

## Developed By

Rashedul Alam  
[LinkedIn](https://www.linkedin.com/in/kmrashedulalam/)
[GitHub](https://github.com/rashakil-ds)

