import streamlit as st
from fitparse import FitFile
import pandas as pd
import math


st.title("FIT Loop Detector (GPS-Based Only)")

uploaded_file = st.file_uploader("Upload a FIT file", type=["fit"])

def semicircles_to_degrees(semicircle):
    return semicircle * (180 / 2**31)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1-a)))


if uploaded_file is not None:
    fitfile = FitFile(uploaded_file)

    records = []
    for record in fitfile.get_messages("record"):
        row = {}
        for field in record:
            row[field.name] = field.value
        records.append(row)

    df = pd.DataFrame(records)

    if "position_lat" not in df or "position_long" not in df:
        st.error("No GPS data found in this FIT file.")
        st.stop()

    # Convert coordinates
    df["lat"] = df["position_lat"].apply(semicircles_to_degrees)
    df["lon"] = df["position_long"].apply(semicircles_to_degrees)

    # Drop rows without timestamps
    df = df.dropna(subset=["timestamp"])

    st.subheader("GPS Points Loaded")
    st.write(df[["timestamp", "lat", "lon"]])

    # Loop detection
    threshold_meters = 20  # adjust as needed

    start_lat = df["lat"].iloc[0]
    start_lon = df["lon"].iloc[0]

    loop_entries = []

    for i, row in df.iterrows():
        d = haversine(start_lat, start_lon, row["lat"], row["lon"])
        if d < threshold_meters:
            loop_entries.append((i, row["timestamp"]))

    # Remove duplicate consecutive points
    cleaned_loops = []
    for i in range(1, len(loop_entries)):
        prev_idx, _ = loop_entries[i-1]
        idx, time = loop_entries[i]
        if idx - prev_idx > 5:  # avoid same-second duplicates
            cleaned_loops.append((idx, time))

    st.subheader("Detected Loop Start/Finish Times")

    if len(cleaned_loops) < 2:
        st.warning("No complete loops detected.")
    else:
        for i in range(1, len(cleaned_loops)):
            loop_start = cleaned_loops[i-1][1]
            loop_end = cleaned_loops[i][1]
            st.write(f"**Loop {i}:** Start = {loop_start}, End = {loop_end}")