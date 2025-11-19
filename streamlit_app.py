# app.py
import streamlit as st
from fitparse import FitFile
import pandas as pd
import math
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("FIT Loop Detector — GPS-based (error-aware)")

uploaded_file = st.file_uploader("Upload a .fit file", type=["fit"])

# Helpers
def semicircles_to_degrees(s):
    return s * (180.0 / 2**31)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1-a)))

if uploaded_file is None:
    st.info("Upload a FIT file to start.")
    st.stop()

# Parameters UI
st.sidebar.header("Detection parameters")
base_radius = st.sidebar.number_input("Base radius (meters)", value=20, min_value=1)
percent_error = st.sidebar.slider("GPS antenna error (percent of max distance)", 0.0, 5.0, 1.0) / 100.0
min_samples_between_crossings = st.sidebar.slider("Min samples between crossings (debounce)", 1, 50, 5)

# Read FIT records
fitfile = FitFile(uploaded_file)
records = []
for rec in fitfile.get_messages("record"):
    row = {}
    for field in rec:
        row[field.name] = field.value
    records.append(row)

if len(records) == 0:
    st.error("No record messages found in FIT file.")
    st.stop()

df = pd.DataFrame(records)

# Need timestamps and positions
if "timestamp" not in df.columns:
    st.error("No timestamps in FIT records.")
    st.stop()

if ("position_lat" not in df.columns) or ("position_long" not in df.columns):
    st.error("No GPS coordinates found in FIT records.")
    st.stop()

# Convert coordinates
df = df.dropna(subset=["position_lat", "position_long", "timestamp"]).reset_index(drop=True)
df["lat"] = df["position_lat"].apply(semicircles_to_degrees)
df["lon"] = df["position_long"].apply(semicircles_to_degrees)

# Optional accuracy field names that might exist in FITs
accuracy_fields = [c for c in df.columns if "accuracy" in c.lower() or "gps_accuracy" in c.lower()]
st.sidebar.write("Detected accuracy fields:", accuracy_fields if accuracy_fields else "None")

# Compute distances from start point
start_lat = df.loc[0, "lat"]
start_lon = df.loc[0, "lon"]
df["dist_from_start_m"] = df.apply(lambda r: haversine(start_lat, start_lon, r["lat"], r["lon"]), axis=1)

max_distance = df["dist_from_start_m"].max() if len(df) > 0 else 0.0

# Compute error buffer
# Prefer using accuracy values if available (mean of accuracy fields). Otherwise use percent_error * max_distance
error_buffer_from_accuracy = None
if accuracy_fields:
    try:
        # pick first accuracy-like column with numeric values
        acc_col = accuracy_fields[0]
        acc_vals = pd.to_numeric(df[acc_col], errors="coerce").dropna()
        if len(acc_vals) > 0:
            # Many devices report accuracy in meters; use mean + 2*std as conservative buffer
            error_buffer_from_accuracy = float(acc_vals.mean() + 2 * acc_vals.std())
    except Exception:
        error_buffer_from_accuracy = None

if error_buffer_from_accuracy is not None:
    error_buffer = error_buffer_from_accuracy
    st.sidebar.write(f"Using accuracy-derived buffer: {error_buffer:.1f} m (from {acc_col})")
else:
    error_buffer = percent_error * max_distance
    st.sidebar.write(f"Using percent-derived buffer: {error_buffer:.1f} m ({percent_error*100:.2f}% of max distance {max_distance:.1f} m)")

threshold = base_radius + error_buffer
st.sidebar.write(f"Final detection radius = base {base_radius} m + buffer {error_buffer:.1f} m -> {threshold:.1f} m")

# State machine for crossings: detect outside -> inside transitions
inside = None
last_crossing_idx = -9999
crossings = []  # list of (idx, timestamp, dist)

for idx, row in df.iterrows():
    d = row["dist_from_start_m"]
    currently_inside = d <= threshold

    if inside is None:
        # initialize state
        inside = currently_inside
        # if start already inside that's fine
        continue

    # detect a transition from outside -> inside (we only consider this as a valid "loop complete")
    if (not inside) and currently_inside:
        # debounce
        if (idx - last_crossing_idx) > min_samples_between_crossings:
            crossings.append((idx, row["timestamp"], d))
            last_crossing_idx = idx

    inside = currently_inside

# Build loops: between consecutive crossings we consider a loop
loops = []
for i in range(1, len(crossings)):
    prev_idx, prev_time, _ = crossings[i-1]
    idx, time, _ = crossings[i]
    # loop start is prev_time (the moment we entered the circle the previous time)
    # loop end is time (the moment we entered again)
    loops.append({
        "loop_number": i,
        "start_idx": prev_idx,
        "start_time": prev_time,
        "end_idx": idx,
        "end_time": time,
        "duration": (pd.to_datetime(time) - pd.to_datetime(prev_time)).total_seconds()
    })

# Display results
st.subheader("Summary")
st.write(f"Total GPS records: {len(df)}")
st.write(f"Max distance from start: {max_distance:.1f} m")
st.write(f"Detection radius (m): {threshold:.1f}")
st.write(f"Crossings detected (outside→inside): {len(crossings)}")
st.write(f"Complete loops (consecutive crossings): {len(loops)}")

st.subheader("Detected loops (from coordinates)")
if len(loops) == 0:
    st.warning("No complete loops detected with current parameters. Try increasing base radius or percent error, or reduce debounce.")
else:
    loops_df = pd.DataFrame(loops)
    loops_df["start_time"] = pd.to_datetime(loops_df["start_time"])
    loops_df["end_time"] = pd.to_datetime(loops_df["end_time"])
    loops_df["duration_s"] = loops_df["duration"].astype(float)
    st.dataframe(loops_df[["loop_number","start_time","end_time","duration_s","start_idx","end_idx"]])

# Optional: small map preview for the first loop
if len(loops) > 0:
    st.subheader("Map preview (first loop)")
    first = loops[0]
    window = df.loc[first["start_idx"]:first["end_idx"], ["lat","lon","timestamp"]]
    # streamlit map expects columns "lat" and "lon"
    st.map(window.rename(columns={"lat":"lat","lon":"lon"}))

# Allow user to export loops as CSV
if len(loops) > 0:
    export_df = pd.DataFrame(loops)
    export_csv = export_df.to_csv(index=False)
    st.download_button("Download loops CSV", data=export_csv, file_name="detected_loops.csv", mime="text/csv")
