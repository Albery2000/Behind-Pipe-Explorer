import streamlit as st
import pandas as pd
import numpy as np
import lasio
import matplotlib.pyplot as plt
from io import BytesIO, StringIO
import sys
import re
import seaborn as sns
from pathlib import Path

# Optional: Remove custom icon loading to avoid deployment issues
# Set page config with default icon
st.set_page_config(
    page_title="Well Data Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'well_data' not in st.session_state:
    st.session_state.well_data = {}
if 'apply_cutoffs' not in st.session_state:
    st.session_state.apply_cutoffs = False
if 'vsh_value' not in st.session_state:
    st.session_state.vsh_value = 50
if 'sw_value' not in st.session_state:
    st.session_state.sw_value = 50
if 'phit_value' not in st.session_state:
    st.session_state.phit_value = 10
if 'ait_cutoff' not in st.session_state:
    st.session_state.ait_cutoff = 0.0

# Helper to clean text
def clean_text(text):
    if not isinstance(text, str):
        return ""
    try:
        return text.encode('utf-8', 'replace').decode('utf-8')
    except:
        return re.sub(r'[^\x00-\x7F]', '', text)

# Title
st.title("Well Net Pay, Reservoir & Perforation Visualizer")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    uploaded_files = st.file_uploader("Upload LAS Files", type=['las', 'txt'], accept_multiple_files=True)
    tops_file = st.file_uploader("Upload Well Tops (CSV/Excel)", type=['csv', 'xlsx'])
    perf_file = st.file_uploader("Upload Perforation Data (CSV/Excel)", type=['csv', 'xlsx'])

    st.subheader("Analysis Parameters")
    st.session_state.apply_cutoffs = st.checkbox("Apply All Cutoffs", value=st.session_state.apply_cutoffs)

    if st.session_state.apply_cutoffs:
        st.session_state.vsh_value = st.slider("VSH Cutoff (%)", 0, 100, st.session_state.vsh_value)
        st.session_state.sw_value = st.slider("Sw Cutoff (%)", 0, 100, st.session_state.sw_value)
        st.session_state.phit_value = st.slider("Porosity Cutoff (%)", 0, 30, st.session_state.phit_value)
        st.session_state.ait_cutoff = st.slider("Unperf Pay Cutoff (m)", 0.0, 10.0, st.session_state.ait_cutoff, 0.1)

# Color Pickers
st.subheader("Track Colors")
cols = st.columns(4)
color_defaults = {
    "porosity": "#17becf",
    "saturation": "#9467bd",
    "net_reservoir": "#1f77b4",
    "net_pay": "#ff7f0e",
    "perforation": "#2ca02c",
    "unperf_net_pay": "#dc143c"
}
colors = {}
for (label, default), col in zip(color_defaults.items(), cols):
    colors[label] = col.color_picker(label.replace("_", " ").capitalize(), default)

# Placeholder processing functions
def process_las_files(files):
    for file in files:
        try:
            name = clean_text(file.name.split('.')[0].strip())
            content = file.getvalue()
            las = lasio.read(BytesIO(content))
            df = las.df().reset_index().rename(columns={las.df().index.name: 'DEPTH'})
            df.columns = df.columns.str.strip().str.upper()
            st.session_state.well_data[name] = {'data': df, 'las': las}
        except Exception as e:
            st.error(f"Error processing {file.name}: {e}")

# Process uploaded files
if uploaded_files:
    process_las_files(uploaded_files)

# Basic Viewer
if st.session_state.well_data:
    well_name = st.selectbox("Select Well", list(st.session_state.well_data.keys()))
    df = st.session_state.well_data[well_name]['data']
    st.write(df.head())

# Footer
st.markdown("""
---
**Well Log Visualizer** – Built with Streamlit for interactive LAS data exploration.
""")
