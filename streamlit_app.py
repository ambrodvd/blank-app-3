import streamlit as st
from fitparse import FitFile
import pandas as pd
import math

st.set_page_config(layout="wide")
st.title("FIT Loop Detector — GPS-based (Simple Map)")

# --- File uploader ---
uploaded_file = st.file_uploader("Upload a .fit file", type=["fit"])

# --- Helper functions ---
def semicircles_to_degrees(s):
    return s * (180.0 / 2**31)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1-a)))

def seconds_to_hhmm(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h:02d}:{m:02d}"

# --- Main processing ---
if uploaded_file:
    st.success("FIT file uploaded!")

    # --- Parameters ---
    st.subheader("Detection Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        base_radius = st.number_input("Base radius (meters)", value=20, min_value=1)
    with col2:
        percent_error = st.slider("GPS antenna error (%)", 0.0, 5.0, 1.0)/100
    with col3:
        min_samples_between_crossings = st.slider("Min samples between crossings", 1,50,5)

    # --- Read FIT file ---
    fitfile = FitFile(uploaded_file)
    records = []
    for rec in fitfile.get_messages("record"):
        row = {field.name: field.value for field in rec}
        records.append(row)
    df = pd.DataFrame(records)

    if "timestamp" not in df.columns or "position_lat" not in df.columns or "position_long" not in df.columns:
        st.error("Missing required GPS/timestamp data in FIT file.")
        st.stop()

    df = df.dropna(subset=["position_lat","position_long","timestamp"]).reset_index(drop=True)
    df["lat"] = df["position_lat"].apply(semicircles_to_degrees)
    df["lon"] = df["position_long"].apply(semicircles_to_degrees)

    # --- Compute distances from start ---
    start_lat = df.loc[0,"lat"]
    start_lon = df.loc[0,"lon"]
    start_time = pd.to_datetime(df.loc[0,"timestamp"])
    df["dist_from_start_m"] = df.apply(lambda r: haversine(start_lat,start_lon,r["lat"],r["lon"]), axis=1)
    threshold = base_radius + percent_error * df["dist_from_start_m"].max()

    # --- Loop detection ---
    crossings = []
    first_inside = df.loc[0,"dist_from_start_m"] <= threshold
    if first_inside:
        crossings.append((0, df.loc[0,"timestamp"]))

    inside = first_inside
    last_crossing_idx = 0 if first_inside else -1

    for idx in range(1,len(df)):
        d = df.loc[idx,"dist_from_start_m"]
        currently_inside = d <= threshold
        if (not inside) and currently_inside:
            if (idx - last_crossing_idx) > min_samples_between_crossings:
                crossings.append((idx, df.loc[idx,"timestamp"]))
                last_crossing_idx = idx
        inside = currently_inside

    # --- Build loops ---
    loops = []
    for i in range(1,len(crossings)):
        prev_idx,prev_time=crossings[i-1]
        idx,time=crossings[i]
        start_delta = pd.to_datetime(prev_time)-start_time
        end_delta = pd.to_datetime(time)-start_time
        duration = end_delta-start_delta
        loops.append({
            "loop_number": i,
            "start_time": seconds_to_hhmm(start_delta.total_seconds()),
            "end_time": seconds_to_hhmm(end_delta.total_seconds()),
            "duration": seconds_to_hhmm(duration.total_seconds()),
            "start_idx": prev_idx,
            "end_idx": idx
        })

    # --- Display loops table ---
    st.subheader("Detected Loops (relative times)")
    if loops:
        loops_df = pd.DataFrame(loops)
        st.dataframe(loops_df[["loop_number","start_time","end_time","duration"]])
    else:
        st.warning("No loops detected")

    # --- Map preview ---
    st.subheader("Map Preview (First Loop)")
    if loops:
        first = loops[0]
        window = df.loc[first["start_idx"]:first["end_idx"], ["lat","lon"]]
        st.map(window)
