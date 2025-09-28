import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import DBSCAN
from optimizer import assign_crews_ilp
from data import simulate_outages, haversine_km

st.set_page_config(page_title="Storm Restoration Optimization Assistant",
                   layout="wide")

st.title("⚡ Storm Restoration Optimization Assistant")
st.caption("MVP demo: outage clustering + crew assignment optimization (cloud-friendly build)")

# ----- Sidebar controls
with st.sidebar:
    st.header("Simulation Controls")
    n_outages = st.number_input("Number of outages", 10, 1000, 120, step=10)
    n_crews = st.number_input("Number of available crews", 1, 50, 6, step=1)
    avg_fix_minutes = st.slider("Avg. fix time per outage (minutes)", 15, 180, 60, step=5)
    eps_meters = st.slider("DBSCAN ε (meters)", 50, 5000, 1200, step=50)
    min_samples = st.slider("DBSCAN min_samples", 2, 15, 4, step=1)
    seed = st.number_input("Random seed", 0, 9999, 42)

st.info(
    "This demo uses simulated locations within a bounding box (e.g., greater Springfield/Boston). "
    "Clusters = likely outage pockets; ILP assigns crews to clusters to minimize travel & total restoration time."
)

# ----- Simulate outages
# Bounding box roughly around Massachusetts (lat 41.2..42.9, lon -73.7..-69.9) – tweak as desired
bbox = dict(lat_min=41.2, lat_max=42.9, lon_min=-73.7, lon_max=-69.9)
df = simulate_outages(n_outages, bbox, avg_fix_minutes, random_state=seed)

# ----- Cluster outages with DBSCAN on meters (haversine)
# Transform lat/lon to a "meters space" by building a distance matrix and feeding to DBSCAN(metric='precomputed')
coords = df[["lat", "lon"]].to_numpy()
n = len(coords)
dist = np.zeros((n, n), dtype=float)
for i in range(n):
    for j in range(i + 1, n):
        d = haversine_km(coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1]) * 1000.0
        dist[i, j] = d
        dist[j, i] = d

db = DBSCAN(eps=eps_meters, min_samples=min_samples, metric="precomputed")
labels = db.fit_predict(dist)
df["cluster_id"] = labels

# ----- Compute cluster centroids and sizes
clusters = (
    df[df.cluster_id >= 0]
    .groupby("cluster_id")
    .agg(
        lat=("lat", "mean"),
        lon=("lon", "mean"),
        outages=("cluster_id", "size"),
        total_fix_min=("fix_minutes", "sum")
    )
    .reset_index()
    .sort_values("outages", ascending=False)
)

st.subheader("Cluster Summary")
if clusters.empty:
    st.warning("No clusters found with current ε/min_samples. Try increasing ε or decreasing min_samples.")
else:
    st.dataframe(clusters, use_container_width=True)

# ----- Map visualization
st.subheader("Geographic View")
fig = px.scatter_mapbox(
    df,
    lat="lat", lon="lon", color=df["cluster_id"].astype(str),
    hover_data=["fix_minutes", "cluster_id"],
    zoom=7, height=500
)
fig.update_layout(mapbox_style="carto-positron", margin=dict(l=0, r=0, t=0, b=0))
st.plotly_chart(fig, use_container_width=True)

# ----- Optimization: Assign crews to clusters (many-to-one)
st.subheader("Crew Assignment Optimization (ILP)")
if clusters.empty:
    st.info("Optimization will run once clusters exist.")
else:
    # Build a travel matrix from an ops center (choose cluster 0 centroid as placeholder) to each cluster centroid
    # For realism, you could add a depot coordinate or multiple yard locations
    yard_lat, yard_lon = float(df.lat.mean()), float(df.lon.mean())
    clusters["dist_km_from_yard"] = clusters.apply(
        lambda r: haversine_km(yard_lat, yard_lon, r["lat"], r["lon"]), axis=1
    )

    # Capacity (each crew can fix X minutes within a planning window)
    planning_window_min = st.slider("Planning window (minutes)", 60, 1440, 480, step=30)
    per_crew_capacity = planning_window_min

    assign_df, objective_val = assign_crews_ilp(
        clusters_df=clusters,
        n_crews=n_crews,
        per_crew_capacity_min=per_crew_capacity
    )

    st.write(f"**Objective (proxy)**: minimize travel + penalty for unassigned load → {objective_val:.2f}")

    st.markdown("### Assignments")
    st.dataframe(assign_df, use_container_width=True)

    # Simple KPI
    covered = assign_df[assign_df["assigned"] == 1]["outages"].sum()
    total = clusters["outages"].sum()
    st.metric("Estimated % of clustered outages covered this window",
              f"{(covered/total*100 if total>0 else 0):.1f}%")

st.caption("Note: This MVP uses simulated data & straight-line distances. Replace with real outage feeds & road network travel when integrating with enterprise systems.")
