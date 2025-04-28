import streamlit as st
import pandas as pd
import numpy as np
import lasio
import matplotlib.pyplot as plt
from io import BytesIO, StringIO
import re

# Helper to clean text from surrogate characters
def clean_text(text):
    if isinstance(text, str):
        try:
            return text.encode('utf-8', 'replace').decode('utf-8')
        except:
            return re.sub(r'[^\x00-\x7F]', '', text)
    return text

# Streamlit page configuration

st.set_page_config(
    page_title="Well Data Explorer",
    layout="wide")


st.markdown("""
<style>
/* Use Google Inter font for better readability */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

/* ========== SIDEBAR ========== */
[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    padding: 2rem 1.5rem;
    border-right: 3px solid #4AA4D9;
}

/* Sidebar section headers */
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-size: 1.1rem;
    color: #1A3C6D;
    font-weight: 700;
    margin-top: 1.5rem;
    margin-bottom: 0.75rem;
}

/* Widget group spacing */
[data-testid="stSidebar"] > div > div {
    margin-bottom: 1rem;
}

/* ===== SLIDERS ===== */
.stSlider > div {
    padding: 0.4rem 0.5rem;
    
    border-radius: 12px;
    box-shadow: inset 0 1px 2px rgba(0,0,0,0.06);
}

.stSlider [role=slider] {
    border: 2px solid white;
    width: 16px;
    height: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}
.stSlider > div > div:first-child {
    
    height: 6px;
    border-radius: 3px;
}

/* Slider labels */
[data-testid="stSidebar"] label {
    font-size: 0.85rem;
    color: #333333;
    font-weight: 600;
    margin-bottom: 0.3rem;
    display: block;
}

/* ===== CHECKBOXES ===== */
.stCheckbox {
    background-color: #F7FAFC;
    border-radius: 10px;
    padding: 0.6rem;
    margin-bottom: 0.5rem;
    transition: background-color 0.3s ease;
}
.stCheckbox:hover {
    background-color: #E6F3F9;
}

.stCheckbox label {
    font-size: 0.88rem;
    color: #1A1A1A;
    font-weight: 600;
}

/* Disabled checkboxes styling */
.stCheckbox input:disabled + div label {
    color: #BBBBBB !important;
}

/* ========== FILE UPLOAD BOXES ========== */
.stFileUploader {
    border: 2px dashed #4AA4D9 !important;
    background-color: #F9FBFD;
    padding: 1rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
}
.stFileUploader:hover {
    background-color: #E1F0F9;
}

/* Spacing for layout */
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}

/* Smooth transitions */
.stButton > button,
.stCheckbox,
.stSlider > div {
    transition: all 0.3s ease-in-out;
}
</style>
""", unsafe_allow_html=True)


st.title("Well Net Pay, Reservoir & Perforation Visualization")

# Sidebar inputs
with st.sidebar:
    st.header("Configuration")
    uploaded_files = st.file_uploader("Upload LAS files", type=['las', 'txt'], accept_multiple_files=True)
    tops_file = st.file_uploader("Upload Well Tops (CSV or Excel)", type=['csv', 'xlsx'])
    perf_file = st.file_uploader("Upload Perforation Data (CSV or Excel)", type=['csv', 'xlsx'])

    st.subheader(" Analysis Parameters")
    vsh_cutoff = st.slider("Vsh Cutoff (%)", 0, 100, 50)
    sw_cutoff = st.slider("Sw Cutoff (%)", 0, 100, 50)
    phit_cutoff = st.slider("Porosity Cutoff (%)", 0, 30, 10)

    st.subheader("Display Options")
    show_net_reservoir =st.checkbox("Show Net Reservoir",True)
    show_net_pay =st.checkbox("Show Net Pay",True)
    show_saturation =st.checkbox("Show Saturation",True)
    show_porosity =st.checkbox("Show Porosity",True)
    show_tops_track =st.checkbox("Show Colored Tops Track",True)
    show_perf =st.checkbox("Show Perforations",True)

# Color pickers
st.subheader("Track Colors")
c1, c2, c3, c4, c5, c6 = st.columns(6)
porosity_color = c1.color_picker("Porosity", "#17becf")
saturation_color = c2.color_picker("Saturation", "#9467bd")
net_reservoir_color = c3.color_picker("Net Reservoir", "#1f77b4")
net_pay_color = c4.color_picker("Net Pay", "#ff7f0e")
perf_color = c5.color_picker("Perforation", "#2ca02c")
unperf_net_pay_color = c6.color_picker("Unperf Net Pay", "#dc143c")


if 'well_data' not in st.session_state:
    st.session_state.well_data = {}

# Read LAS files
if uploaded_files:
    for file in uploaded_files:
        try:
            file_content = file.getvalue()
            try:
                las = lasio.read(BytesIO(file_content))
            except:
                las = lasio.read(StringIO(file_content.decode('utf-8')))

            df = las.df().reset_index()
            df.rename(columns={df.columns[0]: 'DEPTH'}, inplace=True)

            mapping = {
                'PHIT_D': 'PHIT', 'PHIE_D': 'PHIE', 'SW_AR': 'SW',
                'SWT_NET': 'SW_NET', 'VSH': 'VSH', 'NET_PAY': 'NET_PAY', 'NET_RES': 'NET_RES'
            }
            for orig, std in mapping.items():
                if orig in df.columns and std not in df.columns:
                    df[std] = df[orig]

            well_name = clean_text(file.name.split('.')[0].strip())
            st.session_state.well_data[well_name] = {
                'data': df,
                'las': las,
                'header': clean_text(str(las.header))
            }
        except Exception as e:
            st.error(clean_text(f"Failed to process {file.name}: {str(e)}"))

# Read well tops
if tops_file:
    try:
        tops_df = pd.read_csv(tops_file) if tops_file.name.endswith('.csv') else pd.read_excel(tops_file, engine='openpyxl')
        tops_df.columns = ['WELL', 'TOP', 'DEPTH']
        tops_df['WELL'] = tops_df['WELL'].astype(str).apply(clean_text).str.strip().str.lower()
        tops_df['TOP'] = tops_df['TOP'].astype(str).apply(clean_text)

        las_keys = {k.lower().strip(): k for k in st.session_state.well_data.keys()}
        for well in tops_df['WELL'].unique():
            if well in las_keys:
                matched_key = las_keys[well]
                st.session_state.well_data[matched_key]['tops'] = tops_df[tops_df['WELL'] == well]
    except Exception as e:
        st.error(clean_text(f"Failed to read tops file: {str(e)}"))

# Read perforation data
if perf_file:
    try:
        perf_df = pd.read_csv(perf_file) if perf_file.name.endswith('.csv') else pd.read_excel(perf_file, engine='openpyxl')
        perf_df.columns = ['WELL', 'RESERVOIR', 'TOP', 'BASE', 'DATE']
        perf_df['WELL'] = perf_df['WELL'].astype(str).apply(clean_text).str.strip().str.lower()
        perf_df['RESERVOIR'] = perf_df['RESERVOIR'].astype(str).apply(clean_text)

        las_keys = {k.lower().strip(): k for k in st.session_state.well_data.keys()}
        for well in perf_df['WELL'].unique():
            if well in las_keys:
                matched_key = las_keys[well]
                st.session_state.well_data[matched_key]['perforations'] = perf_df[perf_df['WELL'] == well]
    except Exception as e:
        st.error(clean_text(f"Failed to read perforation file: {str(e)}"))

if st.session_state.well_data:
    selected = st.selectbox("Select Well", list(st.session_state.well_data.keys()))
    well = st.session_state.well_data[selected]
    df = well['data']

    if 'VSH' in df.columns and 'SW' in df.columns and 'PHIT' in df.columns:
        df['NET_RESERVOIR'] = np.where(df['VSH'] <= vsh_cutoff / 100, 1, 0)
        df['NET_PAY'] = np.where((df['NET_RESERVOIR'] == 1) & (df['SW'] <= sw_cutoff / 100) & (df['PHIT'] >= phit_cutoff / 100), 1, 0)
    else:
        df['NET_RESERVOIR'] = 0
        df['NET_PAY'] = 0

    df['PERF'] = 0
    if 'perforations' in well and show_perf:
        perf_data = well['perforations']
        for _, row in perf_data.iterrows():
            df.loc[(df['DEPTH'] >= row['TOP']) & (df['DEPTH'] <= row['BASE']), 'PERF'] = 1

    # Calculate unperforated net pay
    df['UNPERF_NET_PAY'] = np.where((df['NET_PAY'] == 1) & (df['PERF'] == 0), 1, 0)

    min_d, max_d = float(df['DEPTH'].min()), float(df['DEPTH'].max())
    depth_range = st.slider("Depth Range (m)", min_d, max_d, (min_d, max_d), 0.1)
    df = df[(df['DEPTH'] >= depth_range[0]) & (df['DEPTH'] <= depth_range[1])]

    st.markdown(f'<div class="well-name">{clean_text(selected)}</div>', unsafe_allow_html=True)

    # Plotting
    track_count = 4 + int(show_tops_track) + int(show_perf) + 1  # +1 for Unperf Net Pay
    fig, ax = plt.subplots(figsize=(14, 18), ncols=track_count)
    ax = ax if track_count > 1 else [ax]
    track_idx = 0

    if show_tops_track:
        ax_tops = ax[track_idx]
        ax_tops.set_title("Tops")
        ax_tops.invert_yaxis()
        ax_tops.set_xticks([])
        ax_tops.grid(False)
        if 'tops' in well:
            tops_to_plot = well['tops']
            tops_to_plot = tops_to_plot[(tops_to_plot['DEPTH'] >= depth_range[0]) & (tops_to_plot['DEPTH'] <= depth_range[1])].sort_values('DEPTH')
            colors = plt.cm.Pastel1.colors
            for i in range(len(tops_to_plot) - 1):
                top1, top2 = tops_to_plot.iloc[i], tops_to_plot.iloc[i + 1]
                ax_tops.axhline(top1['DEPTH'], color='black', linestyle='--', linewidth=1)
                ax_tops.fill_betweenx([top1['DEPTH'], top2['DEPTH']], 0, 1, color=colors[i % len(colors)], alpha=0.5)
                ax_tops.text(0.5, (top1['DEPTH'] + top2['DEPTH']) / 2, clean_text(top1['TOP']), ha='center', va='center', fontsize=8, backgroundcolor='white', transform=ax_tops.get_yaxis_transform())
            if len(tops_to_plot) > 0:
                last_top = tops_to_plot.iloc[-1]
                ax_tops.axhline(last_top['DEPTH'], color='black', linestyle='--', linewidth=1)
                ax_tops.text(0.5, last_top['DEPTH'], clean_text(last_top['TOP']), ha='center', va='center', fontsize=8, backgroundcolor='white', transform=ax_tops.get_yaxis_transform())
        track_idx += 1

    ax1 = ax[track_idx]
    ax1.set_title("Porosity (%)")
    ax1.invert_yaxis()
    ax1.grid(True)
    if 'PHIT' in df.columns and show_porosity:
        ax1.plot(df['PHIT'] * 100, df['DEPTH'], color=porosity_color, label='PHIT')
        if 'PHIE' in df.columns:
            ax1.plot(df['PHIE'] * 100, df['DEPTH'], linestyle='--', label='PHIE')
        ax1.axvline(phit_cutoff, color='red', linestyle=':')
        ax1.legend()
    track_idx += 1

    ax2 = ax[track_idx]
    ax2.set_title("Water Saturation (%)")
    ax2.invert_yaxis()
    ax2.grid(True)
    if 'SW' in df.columns and show_saturation:
        ax2.plot(df['SW'] * 100, df['DEPTH'], color=saturation_color)
        ax2.axvline(sw_cutoff, color='red', linestyle=':')
    track_idx += 1

    ax3 = ax[track_idx]
    ax3.set_title("Net Reservoir")
    ax3.invert_yaxis()
    ax3.grid(True)
    if show_net_reservoir:
        ax3.step(df['NET_RESERVOIR'], df['DEPTH'], color=net_reservoir_color, where='pre')
    track_idx += 1

    ax4 = ax[track_idx]
    ax4.set_title("Net Pay")
    ax4.invert_yaxis()
    ax4.grid(True)
    if show_net_pay:
        if "NET_PAY" in df.columns:
            ax4.step(df['NET_PAY'], df['DEPTH'], color=net_pay_color, where='pre', label="Net Pay")
        elif "SHPOR" in df.columns:
        # Normalize SHPOR to 0–1 for visualization if needed
            shp = df["SHPOR"]
            min_val, max_val = shp.min(), shp.max()
            if max_val > min_val:
                shp_scaled = (shp - min_val) / (max_val - min_val)
            else:
                shp_scaled = shp  # All values same

            ax4.plot(shp_scaled, df["DEPTH"], color=net_pay_color, label="SHPOR (Scaled)")
            ax4.set_xlim(0, 1)
        else:
            ax4.text(0.5, 0.5, "No NET_PAY or SHPOR found", transform=ax4.transAxes, ha="center", va="center")
        track_idx += 1


    if show_perf:
        ax5 = ax[track_idx]
        ax5.set_title("Perforations")
        ax5.invert_yaxis()
        ax5.grid(True)
        ax5.step(df['PERF'], df['DEPTH'], color=perf_color, where='pre')#

        ax4.text(0.5, 0.95, "(from SHPOR)", transform=ax4.transAxes, ha='center', va='top')
        track_idx += 1

    # New: Unperforated Net Pay Track
    ax6 = ax[track_idx]
    ax6.set_title("Unperf Net Pay")
    ax6.invert_yaxis()
    ax6.grid(True)
    ax6.step(df['UNPERF_NET_PAY'], df['DEPTH'], color="crimson", where='pre')
    track_idx += 1

    for a in ax:
        a.set_ylim(depth_range[1], depth_range[0])
        a.set_ylabel("Depth (m)")

    plt.tight_layout()
    st.pyplot(fig)

    # Tabs for extra analysis
    tabs = st.tabs(["Summary Table", "Unperforated Net Pay"])

    with tabs[0]:
        st.subheader("Well Log Summary Data")
        st.dataframe(df[['DEPTH', 'PHIT', 'SW', 'VSH', 'NET_RESERVOIR', 'NET_PAY', 'PERF']].round(3))

    with tabs[1]:
        st.subheader("Net Pay Intervals Not Yet Perforated")
        unperf_df = df[(df['NET_PAY'] == 1) & (df['PERF'] == 0)].copy()
        if unperf_df.empty:
            st.success("\U0001F389 All net pay zones have been perforated!")
        else:
            unperf_df['GROUP'] = (unperf_df['DEPTH'].diff() > 0.2).cumsum()
            grouped = unperf_df.groupby('GROUP').agg(Top=('DEPTH', 'min'), Base=('DEPTH', 'max')).reset_index(drop=True)
            grouped['Thickness (m)'] = grouped['Base'] - grouped['Top']
            st.dataframe(grouped[['Top', 'Base', 'Thickness (m)']].round(2))

else:
    st.info(" Upload LAS files to begin visualization")

st.markdown("""
---
**Well Log Visualizer** – Interactive Streamlit app for well log, tops, and perforation visualization
""")
