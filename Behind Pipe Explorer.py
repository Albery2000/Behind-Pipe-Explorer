import streamlit as st
import pandas as pd
import numpy as np
import lasio
import matplotlib.pyplot as plt
from io import BytesIO, StringIO
import re
import seaborn as sns

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Well Net Pay & Perforation Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# HELPERS
# =========================================================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r'[^\x00-\x7F]+', '', text).strip()

def auto_fraction(series):
    """Convert % logs to fraction if needed"""
    if series.max() > 1.5:
        return series / 100.0
    return series

# =========================================================
# SESSION STATE
# =========================================================
for key, default in {
    "well_data": {},
    "apply_cutoffs": False,
    "vsh_cutoff": 50,
    "sw_cutoff": 50,
    "phit_cutoff": 10,
    "ait_cutoff": 0.0
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("Input Data")
    las_files = st.file_uploader("Upload LAS Files", type=["las"], accept_multiple_files=True)
    tops_file = st.file_uploader("Upload Well Tops (Excel / CSV)", type=["xlsx", "csv"])
    perf_file = st.file_uploader("Upload Perforations (Excel / CSV)", type=["xlsx", "csv"])

    st.divider()
    st.header("Cutoffs")
    st.session_state.apply_cutoffs = st.checkbox("Apply Cutoffs")

    if st.session_state.apply_cutoffs:
        st.session_state.vsh_cutoff = st.slider("VSH (%)", 0, 100, 50)
        st.session_state.sw_cutoff = st.slider("Sw (%)", 0, 100, 50)
        st.session_state.phit_cutoff = st.slider("PHIT (%)", 0, 30, 10)
        st.session_state.ait_cutoff = st.slider("AIT (m)", 0.0, 10.0, 0.0, 0.1)

# =========================================================
# LAS PROCESSING
# =========================================================
def process_las(files):
    for file in files:
        name = clean_text(file.name.split(".")[0])

        try:
            las = lasio.read(BytesIO(file.getvalue()))
        except:
            las = lasio.read(StringIO(file.getvalue().decode()))

        df = las.df().reset_index()
        df.columns = df.columns.str.upper().str.strip()

        # ---------- Curve Mapping (NORPETCO Safe) ----------
        alias_map = {
            # Porosity
            "PHIT_D": "PHIT", "PHIT_T": "PHIT", "PHI_T": "PHIT", "PHI_TOTAL": "PHIT",
            # Effective porosity
            "PHIE_D": "PHIE", "PHIE_T": "PHIE",
            # Water saturation
            "SW_AR": "SW", "SW_T": "SW", "SWT": "SW",
            # Shale volume
            "VSH_GR": "VSH", "VSHL": "VSH", "VCL": "VSH",
            # Net flags
            "NET_RES": "NET_RES", "NET_PAY": "NET_PAY"
        }

        for src, dst in alias_map.items():
            if src in df.columns and dst not in df.columns:
                df[dst] = df[src]

        # ---------- Auto-scale ----------
        for col in ["PHIT", "PHIE", "SW", "VSH"]:
            if col in df.columns:
                df[col] = auto_fraction(df[col])

        st.session_state.well_data[name] = {"data": df}

# =========================================================
# TOPS
# =========================================================
def process_tops(file):
    df = pd.read_excel(file) if file.name.endswith("xlsx") else pd.read_csv(file)
    df.columns = df.columns.str.upper().str.strip()
    df["WELL"] = df["WELL"].str.lower().str.strip()

    for well in st.session_state.well_data:
        mask = df["WELL"] == well.lower()
        if mask.any():
            st.session_state.well_data[well]["tops"] = df[mask].sort_values("DEPTH")

# =========================================================
# PERFORATIONS
# =========================================================
def process_perfs(file):
    df = pd.read_excel(file) if file.name.endswith("xlsx") else pd.read_csv(file)
    df.columns = ["WELL", "ZONE", "TOP", "BASE", "STATUS"]
    df["WELL"] = df["WELL"].str.lower().str.strip()

    df["PERF_VALUE"] = df["STATUS"].str.lower().map({
        "open": 1,
        "plugged": -1
    }).fillna(0)

    for well in st.session_state.well_data:
        mask = df["WELL"] == well.lower()
        if mask.any():
            st.session_state.well_data[well]["perfs"] = df[mask]

# =========================================================
# NET PAY LOGIC
# =========================================================
def compute_net_flags(df):
    if {"VSH", "SW", "PHIT"}.issubset(df.columns):
        vsh_cut = st.session_state.vsh_cutoff / 100
        sw_cut = st.session_state.sw_cutoff / 100
        phit_cut = st.session_state.phit_cutoff / 100

        df["NET_RESERVOIR"] = (df["VSH"] <= vsh_cut).astype(int)
        df["NET_PAY"] = (
            (df["NET_RESERVOIR"] == 1) &
            (df["SW"] <= sw_cut) &
            (df["PHIT"] >= phit_cut)
        ).astype(int)
    return df

# =========================================================
# RUN LOADERS
# =========================================================
if las_files:
    process_las(las_files)
if tops_file:
    process_tops(tops_file)
if perf_file:
    process_perfs(perf_file)

# =========================================================
# MAIN APP
# =========================================================
st.title("Well Net Pay & Unperforated Interval Analyzer")

if not st.session_state.well_data:
    st.info("Upload LAS files to start.")
    st.stop()

selected_well = st.selectbox("Select Well", list(st.session_state.well_data))
well = st.session_state.well_data[selected_well]
df = compute_net_flags(well["data"].copy())

# ---------- Perforations ----------
df["PERF"] = 0
if "perfs" in well:
    for _, r in well["perfs"].iterrows():
        df.loc[(df.DEPTH >= r.TOP) & (df.DEPTH <= r.BASE), "PERF"] = r.PERF_VALUE

df["UNPERF_NET_PAY"] = ((df["NET_PAY"] == 1) & (df["PERF"] == 0)).astype(int)

# =========================================================
# VISUALIZATION
# =========================================================
fig, ax = plt.subplots(figsize=(6, 12))
ax.invert_yaxis()
ax.plot(df["PHIT"] * 100, df["DEPTH"], label="PHIT")
ax.plot(df["SW"] * 100, df["DEPTH"], label="SW")
ax.fill_betweenx(df["DEPTH"], 0, df["NET_PAY"], color="green", alpha=0.3)
ax.set_xlabel("Value")
ax.set_ylabel("Depth")
ax.legend()
st.pyplot(fig)

# =========================================================
# UNPERFORATED INTERVALS TABLE
# =========================================================
st.subheader("Unperforated Net Pay Intervals")

step = df["DEPTH"].diff().median()
unperf = df[df["UNPERF_NET_PAY"] == 1].copy()
unperf["GROUP"] = (unperf["DEPTH"].diff() > step * 1.5).cumsum()

intervals = unperf.groupby("GROUP").agg(
    Top=("DEPTH", "min"),
    Base=("DEPTH", "max"),
    Thickness=("DEPTH", lambda x: x.max() - x.min()),
    Avg_PHIT=("PHIT", "mean"),
    Avg_SW=("SW", "mean")
).reset_index(drop=True)

intervals["Avg_PHIT"] *= 100
intervals["Avg_SW"] *= 100

if st.session_state.apply_cutoffs:
    intervals = intervals[intervals["Thickness"] >= st.session_state.ait_cutoff]

st.dataframe(intervals.round(3), use_container_width=True)
