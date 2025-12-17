# @Cluster Analysis Project
import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator

st.set_page_config(page_title="Cluster Analysis", layout="wide")

# small helper
def set_k(val: int):
    st.session_state["k_value"] = int(val)

st.title("Cluster Analysis")
st.caption("Clustering on two features (e.g., income & score) with interactive clusters(k) and visualization.")

# data section
st.sidebar.header("1) Data")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
use_sample = st.sidebar.checkbox("Use sample dataset (Mall Customers-like)", value=(uploaded is None))

@st.cache_data
def load_sample():
    rng = np.random.default_rng(42)
    c1 = rng.normal([20, 30], [5, 8], size=(60, 2))
    c2 = rng.normal([80, 80], [8, 10], size=(60, 2))
    c3 = rng.normal([50, 50], [6, 6], size=(80, 2))
    c4 = rng.normal([25, 75], [6, 8], size=(50, 2))
    c5 = rng.normal([85, 20], [7, 7], size=(50, 2))
    X = np.vstack([c1, c2, c3, c4, c5])
    df = pd.DataFrame(X, columns=["income", "score"])
    df["age"] = rng.integers(18, 60, size=len(df))
    return df

if uploaded is not None and not use_sample:
    df = pd.read_csv(uploaded)
else:
    df = load_sample()

# preview section
show_preview = st.sidebar.checkbox("Show dataset preview", value=True)
if show_preview:
    with st.expander("Preview", expanded=True):
        st.dataframe(df.head(30), use_container_width=True, height=320)

# feature selection
st.sidebar.header("2) Features")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_cols) < 2:
    st.error("Need at least 2 numeric columns to run KMeans.")
    st.stop()

default_x = "score" if "score" in numeric_cols else numeric_cols[0]
default_y = "income" if "income" in numeric_cols else numeric_cols[1]

x_col = st.sidebar.selectbox("X-axis feature", numeric_cols, index=numeric_cols.index(default_x))
y_options = [c for c in numeric_cols if c != x_col]
y_col = st.sidebar.selectbox(
    "Y-axis feature",
    y_options,
    index=y_options.index(default_y) if default_y in y_options else 0
)

# model settings
st.sidebar.header("3) Model Settings")
show_elbow = st.sidebar.checkbox("Show elbow (inertia) chart", value=True)
scale = st.sidebar.checkbox("Standardize features (recommended)", value=False)
random_state = st.sidebar.number_input("random_state", value=42, step=1)

# slider state (dynamic)
if "k_value" not in st.session_state:
    st.session_state["k_value"] = 2  # small default

k = st.sidebar.slider(
    "Number of clusters (k)",
    min_value=2,
    max_value=10,
    step=1,
    key="k_value"
)

# prepare features
X = df[[x_col, y_col]].copy()
if scale:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
else:
    X_scaled = X.values

# elbow + knee section
optimal_k = None
if show_elbow:
    st.subheader("Elbow Chart (Inertia vs Clusters)")

    col_left, col_right = st.columns([2, 1])

    ks = list(range(2, min(11, len(df))))
    inertias = []
    for kk in ks:
        km_tmp = KMeans(n_clusters=kk, random_state=random_state, n_init="auto")
        km_tmp.fit(X_scaled)
        inertias.append(km_tmp.inertia_)

    knee = KneeLocator(ks, inertias, curve="convex", direction="decreasing")
    optimal_k = knee.knee

    # left plot (normal elbow)
    with col_left:
        fig1, ax1 = plt.subplots(figsize=(7, 4), dpi=120)
        ax1.plot(ks, inertias, marker="o")
        ax1.set_xlabel("k")
        ax1.set_ylabel("Inertia")
        ax1.set_title("Elbow Curve")
        st.pyplot(fig1, use_container_width=True)

    # right plot (knee style) + auto set button
    with col_right:
        st.markdown("### Knee Point")

        fig2, ax2 = plt.subplots(figsize=(4, 4), dpi=120)
        ax2.plot(ks, inertias, label="data")
        ax2.set_xlabel("Clusters (k)")
        ax2.set_ylabel("Inertia")
        ax2.set_title("Knee Plot")

        if optimal_k is not None:
            ax2.axvline(optimal_k, linestyle="--", label="knee/elbow")
            ax2.legend(loc="best")
            st.metric("Optimal k", int(optimal_k))

            st.button(
                "Use optimal k",
                on_click=set_k,
                args=(int(optimal_k),),
                key="use_optimal_k_btn"
            )
        else:
            ax2.legend(loc="best")
            st.warning("No clear elbow found.")

        st.pyplot(fig2, use_container_width=True)

# fit final model
km = KMeans(n_clusters=int(st.session_state["k_value"]), random_state=random_state, n_init="auto")
labels = km.fit_predict(X_scaled)

df_out = df.copy()
df_out["km_cluster"] = labels

# centroids in original scale
centers = km.cluster_centers_
centers_orig = scaler.inverse_transform(centers) if scale else centers

# cluster results
st.subheader("Cluster Result")
c1, c2 = st.columns([2, 1])

with c1:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    for cl in sorted(df_out["km_cluster"].unique()):
        subset = df_out[df_out["km_cluster"] == cl]
        ax.scatter(subset[x_col], subset[y_col], s=30, label=f"cluster_{cl+1}")

    ax.scatter(centers_orig[:, 0], centers_orig[:, 1], marker="*", s=250, label="centroid")
    ax.set_xlabel(x_col.upper())
    ax.set_ylabel(y_col.upper())
    ax.legend()
    st.pyplot(fig, use_container_width=True)

with c2:
    st.markdown("### Centroids")
    cent_df = pd.DataFrame(centers_orig, columns=[x_col, y_col])
    cent_df.index = [f"cluster_{i+1}" for i in range(int(st.session_state["k_value"]))]
    st.dataframe(cent_df, use_container_width=True, height=220)

    st.markdown("### Cluster Counts")
    counts = df_out["km_cluster"].value_counts().sort_index().rename_axis("cluster").to_frame("count")
    st.dataframe(counts, use_container_width=True, height=220)

# download section
st.subheader("Download clustered data")
csv = df_out.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv, "clustered_output.csv", "text/csv")

# footer
st.markdown("---")
st.caption("Developed by @Rashedul Alam")
