import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="ObjectScope Dashboard", layout="wide")

st.title("🎯 ObjectScope: Real-Time Object Detection & Analytics Dashboard")

file_path = "detections.csv"

if not os.path.exists(file_path):
    st.warning("No detection data found. Run object_detection.py first.")
    st.stop()

# Load data
df = pd.read_csv(file_path)

# Convert Time column to datetime
df["Time"] = pd.to_datetime(df["Time"])

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Detections", len(df))
col2.metric("Unique Objects", df["Object"].nunique())
col3.metric("Average Confidence", f"{df['Confidence'].mean():.2f}")

st.divider()

# Object Count Chart
st.subheader("📊 Object Distribution")
fig1 = px.bar(df["Object"].value_counts().reset_index(),
              x="Object", y="count",
              labels={"count": "Count"})
st.plotly_chart(fig1, use_container_width=True)

# Timeline Chart
st.subheader("📈 Detection Timeline")
timeline = df.groupby(
    [df["Time"].dt.strftime("%H:%M:%S"), "Object"]
).size().reset_index(name="Count")

fig2 = px.line(timeline,
               x="Time",
               y="Count",
               color="Object",
               markers=True)
st.plotly_chart(fig2, use_container_width=True)

# Show data table
st.subheader("📋 Detection Data")
st.dataframe(df, use_container_width=True)