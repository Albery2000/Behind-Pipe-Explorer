import streamlit as st
import pandas as pd
import numpy as np
import lasio
import matplotlib.pyplot as plt
from io import BytesIO, StringIO
import sys
import re
import base64
from pathlib import Path
import seaborn as sns

# Set up paths
BASE_DIR = Path(r"C:\Users\Hassan.Gamal\Desktop\Bedhind Pipe Project")
ICON_PATH = Path(r"C:\Users\Hassan.Gamal\Desktop\hassan project\Images\logo icon.png")

# Function to encode icon as base64
def get_base64_icon(path: Path) -> str:
    try:
        with open(path, "rb") as f:
            data = f.read()
        return "data:image/png;base64," + base64.b64encode(data).decode()
    except FileNotFoundError:
        st.error(f"Icon file not found at {path}. Using default icon.")
        return ":material/oil:"

# Set page config
st.set_page_config(
    page_title="Well Data Explorer",
    page_icon=get_base64_icon(ICON_PATH),
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom path for css.py
sys.path.append(str(BASE_DIR))

# Import css.py
try:
    from css import CSS
except ModuleNotFoundError:
    st.error(f"Cannot find css.py in {BASE_DIR}. Please ensure css.py exists.")
    st.stop()

# Function to apply CSS
def apply_css(css: str) -> None:
    try:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error applying CSS: {str(e)}")

# Apply CSS
apply_css(CSS)

# Helper to clean text
def clean_text(text: str | None) -> str:
    if not isinstance(text, str):
        return ""
    try:
        return text.encode('utf-8', 'replace').decode('utf-8')
    except:
        return re.sub(r'[^\x00-\x7F]', '', text)

# Custom CSS for improved styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
[data-testid="stSidebar"] {
    background-color: #F8FAFC;
    padding: 1.5rem;
    border-right: 2px solid #4AA4D9;
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    font-size: 1.15rem;
    color: #1A3C6D;
    font-weight: 600;
}
.stSlider > div {
    background-color: #F0F4F8;
    padding: 0.5rem;
    border-radius: 8px;
}
.stFileUploader {
    border: 2px dashed #4AA4D9;
    background-color: #F9FBFD;
    border-radius: 10px;
    padding: 1rem;
}
.stFileUploader:hover {
    background-color: #E6F3F9;
}
.stButton > button {
    background-color: #4AA4D9;
    color: white;
    border-radius: 8px;
}
.block-container {
    padding: 1rem 2rem;
}
</style>
""", unsafe_allow_html=True)

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

# Main title
st.title("Well Net Pay, Reservoir & Perforation Visualizer")

# Sidebar inputs
with st.sidebar:
    st.header("Configuration")
    uploaded_files = st.file_uploader(
        "Upload LAS Files", type=['las', 'txt'], accept_multiple_files=True, key="las_uploader"
    )
    tops_file = st.file_uploader(
        "Upload Well Tops (CSV/Excel)", type=['csv', 'xlsx'], key="tops_uploader"
    )
    perf_file = st.file_uploader(
        "Upload Perforation Data (CSV/Excel)", type=['csv', 'xlsx'], key="perf_uploader"
    )

    st.subheader("Analysis Parameters")
    st.session_state.apply_cutoffs = st.checkbox("Apply All Cutoffs", value=st.session_state.apply_cutoffs, key="apply_cutoffs_checkbox")

    if st.session_state.apply_cutoffs:
        st.session_state.vsh_value = st.slider(
            "VSH Cutoff (%)", 0, 100, st.session_state.vsh_value, key="vsh_slider"
        )
        st.session_state.sw_value = st.slider(
            "Sw Cutoff (%)", 0, 100, st.session_state.sw_value, key="sw_slider"
        )
        st.session_state.phit_value = st.slider(
            "Porosity Cutoff (%)", 0, 30, st.session_state.phit_value, key="phit_slider"
        )
        st.session_state.ait_cutoff = st.slider(
            "Unperf Pay Cutoff (m)", 0.0, 10.0, st.session_state.ait_cutoff, 0.1, key="ait_slider"
        )

    st.subheader("Display Options")
    display_options = {
        "Show Net Reservoir": True,
        "Show Net Pay": True,
        "Show Saturation": True,
        "Show Porosity": True,
        "Show Colored Tops Track": True,
        "Show Perforations": True
    }
    for label, default in display_options.items():
        st.session_state[label.lower().replace(" ", "_")] = st.checkbox(label, default, key=label.lower())

# Color pickers
st.subheader("Track Colors")
cols = st.columns(8)
color_defaults = {
    "porosity": "#17becf",
    "saturation": "#9467bd",
    "net_reservoir": "#1f77b4",
    "net_pay": "#ff7f0e",
    "perforation": "#2ca02c",
    "unperf_net_pay": "#dc143c",
    "shpor": "#ff9896",
    "pornet": "#c5b0d5"
}
colors = {}
for (label, default), col in zip(color_defaults.items(), cols):
    colors[label] = col.color_picker(label.replace("_", " ").capitalize(), default, key=label)

# Process LAS files
def process_las_files(files: list) -> None:
    for file in files:
        try:
            well_name = clean_text(file.name.split('.')[0].strip())
            file_content = file.getvalue()
            try:
                las = lasio.read(BytesIO(file_content))
            except:
                las = lasio.read(StringIO(file_content.decode('utf-8')))

            df = las.df().reset_index().rename(columns={las.df().index.name: 'DEPTH'})
            df.columns = df.columns.str.strip().str.upper()

            # Standardize column names
            mapping = {
                'PHIT_D': 'PHIT', 'PHIE_D': 'PHIE', 'SW_AR': 'SW',
                'SWT_NET': 'SW_NET', 'VSH': 'VSH', 'NET_PAY': 'NET_PAY',
                'NET_RES': 'NET_RES', 'SH_POR': 'SHPOR', 'PORNET_D': 'PORNET'
            }
            for orig, std in mapping.items():
                if orig in df.columns and std not in df.columns:
                    df[std] = df[orig]

            # Fallback for alternative curve names
            fallbacks = {
                'SHPOR': ['SHPOR_12'],
                'PORNET': ['PORNET_12'],
                'SW_NET': ['SWNET', 'SWNET_12']
            }
            for std, candidates in fallbacks.items():
                if std not in df.columns:
                    for cand in candidates:
                        if cand in df.columns:
                            df[std] = df[cand]
                            break

            st.session_state.well_data[well_name] = {
                'data': df,
                'las': las,
                'header': clean_text(str(las.header))
            }
        except Exception as e:
            st.error(f"Failed to process {file.name}: {str(e)}")

# Process well tops
def process_tops(tops_file) -> None:
    try:
        tops_df = pd.read_csv(tops_file) if tops_file.name.endswith('.csv') else pd.read_excel(tops_file)
        tops_df.columns = tops_df.columns.str.strip().str.upper()
        expected_columns = ['WELL', 'TOP', 'DEPTH']
        if not all(col in tops_df.columns for col in expected_columns):
            tops_df.columns = expected_columns
        tops_df['WELL'] = tops_df['WELL'].astype(str).apply(clean_text).str.strip().str.lower()
        tops_df['TOP'] = tops_df['TOP'].astype(str).apply(clean_text)

        las_keys = {k.lower().strip(): k for k in st.session_state.well_data.keys()}
        for well in tops_df['WELL'].unique():
            if well in las_keys:
                st.session_state.well_data[las_keys[well]]['tops'] = tops_df[tops_df['WELL'] == well]
    except Exception as e:
        st.error(f"Failed to read tops file: {str(e)}")

# Process perforation data
def process_perforations(perf_file) -> None:
    try:
        perf_df = pd.read_csv(perf_file) if perf_file.name.endswith('.csv') else pd.read_excel(perf_file)
        perf_df.columns = perf_df.columns.str.strip().str.upper()

        required_columns = ['WELL', 'TOP', 'BASE']
        optional_columns = ['ZONE', 'RESERVOIR']
        available_columns = perf_df.columns.tolist()

        column_mapping = {}
        for col in required_columns:
            found = False
            for avail_col in available_columns:
                if col.lower() == avail_col.lower():
                    column_mapping[avail_col] = col
                    found = True
                    break
            if not found:
                st.error(f"Required column '{col}' not found in perforation file.")
                return

        reservoir_col = None
        for col in optional_columns:
            for avail_col in available_columns:
                if col.lower() == avail_col.lower():
                    column_mapping[avail_col] = 'RESERVOIR'
                    reservoir_col = 'RESERVOIR'
                    break
            if reservoir_col:
                break
        if not reservoir_col:
            perf_df['RESERVOIR'] = 'Unknown'
            column_mapping['RESERVOIR'] = 'RESERVOIR'

        perf_df = perf_df.rename(columns=column_mapping)

        perf_df['WELL'] = perf_df['WELL'].astype(str).apply(clean_text).str.strip().str.lower()
        perf_df['RESERVOIR'] = perf_df['RESERVOIR'].astype(str).apply(clean_text)
        perf_df['TOP'] = pd.to_numeric(perf_df['TOP'], errors='coerce')
        perf_df['BASE'] = pd.to_numeric(perf_df['BASE'], errors='coerce')

        perf_df = perf_df.dropna(subset=['TOP', 'BASE'])

        las_keys = {k.lower().strip(): k for k in st.session_state.well_data.keys()}
        for well in perf_df['WELL'].unique():
            if well in las_keys:
                st.session_state.well_data[las_keys[well]]['perforations'] = perf_df[perf_df['WELL'] == well]
    except Exception as e:
        st.error(f"Failed to read perforation file: {str(e)}")

# Function to get unperforated net pay intervals for all wells
def get_all_wells_unperf_intervals() -> pd.DataFrame:
    all_intervals = []
    for well_name, well in st.session_state.well_data.items():
        df = well['data'].copy()
        
        if st.session_state.apply_cutoffs:
            if all(col in df.columns for col in ['VSH', 'SW', 'PHIT']):
                if 'VSH' in df.columns:
                    df['NET_RESERVOIR'] = (df['VSH'] <= st.session_state.vsh_value / 100).astype(int)
                else:
                    df['NET_RESERVOIR'] = df.get('NET_RES', pd.Series(np.nan, index=df.index)).astype(float)
                
                conditions = []
                if 'NET_RESERVOIR' in df.columns and not df['NET_RESERVOIR'].isna().all():
                    conditions.append(df['NET_RESERVOIR'] == 1)
                if 'SW' in df.columns:
                    conditions.append(df['SW'] <= st.session_state.sw_value / 100)
                if 'PHIT' in df.columns:
                    conditions.append(df['PHIT'] >= st.session_state.phit_value / 100)
                
                if conditions:
                    df['NET_PAY'] = (np.all(conditions, axis=0)).astype(int)
                else:
                    df['NET_PAY'] = df.get('NET_PAY', pd.Series(np.nan, index=df.index)).astype(float)
            else:
                df['NET_RESERVOIR'] = df.get('NET_RES', pd.Series(np.nan, index=df.index)).astype(float)
                df['NET_PAY'] = df.get('NET_PAY', pd.Series(np.nan, index=df.index)).astype(float)
        else:
            df['NET_RESERVOIR'] = df.get('NET_RES', pd.Series(np.nan, index=df.index)).astype(float)
            df['NET_PAY'] = df.get('NET_PAY', pd.Series(np.nan, index=df.index)).astype(float)
        
        df['PERF'] = 0
        if 'perforations' in well and st.session_state.show_perforations:
            for _, row in well['perforations'].iterrows():
                df.loc[(df['DEPTH'] >= row['TOP']) & (df['DEPTH'] <= row['BASE']), 'PERF'] = 1
        df['UNPERF_NET_PAY'] = ((df['NET_PAY'] == 1) & (df['PERF'] == 0)).astype(int) if 'NET_PAY' in df.columns and not df['NET_PAY'].isna().all() else pd.Series(np.nan, index=df.index)
        
        unperf_df = df[(df['NET_PAY'] == 1) & (df['PERF'] == 0)].copy() if 'NET_PAY' in df.columns and not df['NET_PAY'].isna().all() else pd.DataFrame()
        if unperf_df.empty:
            continue
        
        unperf_df['GROUP'] = (unperf_df['DEPTH'].diff() > 0.2).cumsum()
        grouped = unperf_df.groupby('GROUP').agg(
            Top=('DEPTH', 'min'),
            Base=('DEPTH', 'max'),
            Avg_Porosity=('PHIT', 'mean') if 'PHIT' in unperf_df.columns else ('DEPTH', lambda x: np.nan),
            Avg_Sw=('SW', 'mean') if 'SW' in unperf_df.columns else ('DEPTH', lambda x: np.nan)
        ).reset_index(drop=True)
        
        grouped['Thickness (m)'] = (grouped['Base'] - grouped['Top']).round(2)
        if st.session_state.apply_cutoffs:
            grouped = grouped[grouped['Thickness (m)'] >= st.session_state.ait_cutoff]
        
        if grouped.empty:
            continue
        
        grouped['Well'] = well_name
        
        grouped['Zone'] = 'Unknown'
        if 'tops' in well:
            tops = well['tops'].sort_values('DEPTH')
            for i, row in grouped.iterrows():
                valid_tops = tops[tops['DEPTH'] <= row['Top']]
                if not valid_tops.empty:
                    grouped.at[i, 'Zone'] = clean_text(valid_tops.iloc[-1]['TOP'])
        
        grouped = grouped[['Well', 'Zone', 'Top', 'Base', 'Thickness (m)', 'Avg_Porosity', 'Avg_Sw']]
        grouped['Avg_Porosity'] = grouped['Avg_Porosity'].apply(lambda x: round(x * 100, 2) if pd.notna(x) else np.nan)
        grouped['Avg_Sw'] = grouped['Avg_Sw'].apply(lambda x: round(x * 100, 2) if pd.notna(x) else np.nan)
        
        all_intervals.append(grouped)
    
    if all_intervals:
        result = pd.concat(all_intervals, ignore_index=True)
    else:
        result = pd.DataFrame(columns=['Well', 'Zone', 'Top', 'Base', 'Thickness (m)', 'Avg_Porosity', 'Avg_Sw'])
    
    return result

# Process uploaded files
if uploaded_files:
    process_las_files(uploaded_files)
if tops_file:
    process_tops(tops_file)
if perf_file:
    process_perforations(perf_file)

# Main visualization logic
if st.session_state.well_data:
    selected_well = st.selectbox("Select Well", list(st.session_state.well_data.keys()), key="well_select")
    well = st.session_state.well_data[selected_well]
    df = well['data'].copy()

    # Calculate net reservoir and net pay for visualization
    if st.session_state.apply_cutoffs:
        if all(col in df.columns for col in ['VSH', 'SW', 'PHIT']):
            if 'VSH' in df.columns:
                df['NET_RESERVOIR'] = (df['VSH'] <= st.session_state.vsh_value / 100).astype(int)
            else:
                df['NET_RESERVOIR'] = df.get('NET_RES', pd.Series(np.nan, index=df.index)).astype(float)
            
            conditions = []
            if 'NET_RESERVOIR' in df.columns and not df['NET_RESERVOIR'].isna().all():
                conditions.append(df['NET_RESERVOIR'] == 1)
            if 'SW' in df.columns:
                conditions.append(df['SW'] <= st.session_state.sw_value / 100)
            if 'PHIT' in df.columns:
                conditions.append(df['PHIT'] >= st.session_state.phit_value / 100)
            
            if conditions:
                df['NET_PAY'] = (np.all(conditions, axis=0)).astype(int)
            else:
                df['NET_PAY'] = df.get('NET_PAY', pd.Series(np.nan, index=df.index)).astype(float)
        else:
            df['NET_RESERVOIR'] = df.get('NET_RES', pd.Series(np.nan, index=df.index)).astype(float)
            df['NET_PAY'] = df.get('NET_PAY', pd.Series(np.nan, index=df.index)).astype(float)
    else:
        df['NET_RESERVOIR'] = df.get('NET_RES', pd.Series(np.nan, index=df.index)).astype(float)
        df['NET_PAY'] = df.get('NET_PAY', pd.Series(np.nan, index=df.index)).astype(float)

    # Process perforations
    df['PERF'] = 0
    if 'perforations' in well and st.session_state.show_perforations:
        for _, row in well['perforations'].iterrows():
            df.loc[(df['DEPTH'] >= row['TOP']) & (df['DEPTH'] <= row['BASE']), 'PERF'] = 1
    df['UNPERF_NET_PAY'] = ((df['NET_PAY'] == 1) & (df['PERF'] == 0)).astype(int) if 'NET_PAY' in df.columns and not df['NET_PAY'].isna().all() else pd.Series(np.nan, index=df.index)

    # Depth filter
    min_d, max_d = float(df['DEPTH'].min()), float(df['DEPTH'].max())
    depth_range = st.slider("Depth Range (m)", min_d, max_d, (min_d, max_d), 0.1, key="depth_slider")
    df = df[(df['DEPTH'] >= depth_range[0]) & (df['DEPTH'] <= depth_range[1])].copy()

    st.markdown(f'<h3 style="color: #1A3C6D;">{clean_text(selected_well)}</h3>', unsafe_allow_html=True)

    # Create tabs for Standard and Customized Visualization
    standard_tab, custom_tab = st.tabs(["Standard Visualization", "Customized Visualization"])

    # Standard Visualization Tab (existing functionality)
    with standard_tab:
        # Determine available tracks
        available_tracks = []
        track_labels = {
            'tops': 'Tops',
            'phit': 'Porosity',
            'sw': 'Saturation',
            'net_reservoir': 'Net Reservoir',
            'net_pay': 'Net Pay',
            'shpor': 'SHPOR',
            'pornet': 'PORNET',
            'perf': 'Perforations',
            'unperf_pay': 'Unperf Net Pay'
        }
        if 'tops' in well and st.session_state.show_colored_tops_track:
            available_tracks.append('tops')
        if 'PHIT' in df.columns and st.session_state.show_porosity:
            available_tracks.append('phit')
        if 'SW' in df.columns and st.session_state.show_saturation:
            available_tracks.append('sw')
        if 'NET_RESERVOIR' in df.columns and not df['NET_RESERVOIR'].isna().all() and st.session_state.show_net_reservoir:
            available_tracks.append('net_reservoir')
        if 'NET_PAY' in df.columns and not df['NET_PAY'].isna().all() and st.session_state.show_net_pay:
            available_tracks.append('net_pay')
        if 'SHPOR' in df.columns and st.session_state.show_porosity:
            available_tracks.append('shpor')
        if 'PORNET' in df.columns and st.session_state.show_porosity:
            available_tracks.append('pornet')
        if 'PERF' in df.columns and st.session_state.show_perforations:
            available_tracks.append('perf')
        if 'UNPERF_NET_PAY' in df.columns and not df['UNPERF_NET_PAY'].isna().all():
            available_tracks.append('unperf_pay')

        # Track selection
        selected_tracks = st.multiselect(
            "Select Tracks to Display",
            options=available_tracks,
            default=available_tracks,
            format_func=lambda x: track_labels.get(x, x),
            key="track_select_standard"
        )

        # Plotting
        if not selected_tracks:
            st.warning("Please select at least one track to display.")
        else:
            fig, axes = plt.subplots(
                figsize=(3 * len(selected_tracks), 12),
                ncols=len(selected_tracks),
                sharey=True,
                gridspec_kw={'wspace': 0.05}
            )
            axes = [axes] if len(selected_tracks) == 1 else axes

            for i, track in enumerate(selected_tracks):
                ax = axes[i]
                ax.invert_yaxis()
                ax.grid(True, alpha=0.3)
                ax.set_ylabel("Depth (m)", fontsize=10)

                if track == 'tops':
                    ax.set_title("Tops", fontsize=10)
                    ax.set_xticks([])
                    if 'tops' in well:
                        tops = well['tops'][(well['tops']['DEPTH'] >= depth_range[0]) &
                                            (well['tops']['DEPTH'] <= depth_range[1])].sort_values('DEPTH')
                        top_colors = sns.color_palette("Pastel1", len(tops))
                        for j in range(len(tops) - 1):
                            top1, top2 = tops.iloc[j], tops.iloc[j + 1]
                            ax.axhline(top1['DEPTH'], color='black', ls='--', lw=0.8)
                            ax.fill_betweenx([top1['DEPTH'], top2['DEPTH']], 0, 1,
                                            color=top_colors[j], alpha=0.5)
                            ax.text(0.5, (top1['DEPTH'] + top2['DEPTH']) / 2, clean_text(top1['TOP']),
                                    ha='center', va='center', fontsize=8, bbox=dict(facecolor='white', alpha=0.8),
                                    transform=ax.get_yaxis_transform())
                        if len(tops) > 0:
                            ax.axhline(tops.iloc[-1]['DEPTH'], color='black', ls='--', lw=0.8)

                elif track == 'phit':
                    ax.set_title("Porosity (%)", fontsize=10)
                    ax.plot(df['PHIT'] * 100, df['DEPTH'], color=colors['porosity'], label='PHIT', lw=1.5)
                    if 'PHIE' in df.columns:
                        ax.plot(df['PHIE'] * 100, df['DEPTH'], color=colors['porosity'], ls='--', label='PHIE', lw=1)
                    if st.session_state.apply_cutoffs:
                        ax.axvline(st.session_state.phit_value, color='red', ls=':', lw=1)
                    ax.legend(fontsize=8)
                    ax.set_xlim(0, df['PHIT'].max() * 100 * 1.1 if 'PHIT' in df.columns else 100)

                elif track == 'sw':
                    ax.set_title("Water Saturation (%)", fontsize=10)
                    ax.plot(df['SW'] * 100, df['DEPTH'], color=colors['saturation'], lw=1.5)
                    if st.session_state.apply_cutoffs:
                        ax.axvline(st.session_state.sw_value, color='red', ls=':', lw=1)
                    ax.set_xlim(0, 100)

                elif track == 'net_reservoir':
                    ax.set_title("Net Reservoir", fontsize=10)
                    values = df['NET_RESERVOIR'].dropna()
                    if not values.empty:
                        ax.fill_betweenx(df.loc[values.index, 'DEPTH'], 0, values,
                                        color=colors['net_reservoir'], step='pre', alpha=0.7)
                        ax.set_xlim(0, 1)
                        ax.set_xticks([0, 1])
                    else:
                        st.warning("No valid Net Reservoir data to plot.")

                elif track == 'net_pay':
                    ax.set_title("Net Pay", fontsize=10)
                    values = df['NET_PAY'].dropna()
                    if not values.empty:
                        ax.fill_betweenx(df.loc[values.index, 'DEPTH'], 0, values,
                                        color=colors['net_pay'], step='pre', alpha=0.7)
                        ax.set_xlim(0, 1)
                        ax.set_xticks([0, 1])
                    else:
                        st.warning("No valid Net Pay data to plot.")

                elif track == 'perf':
                    ax.set_title("Perforations", fontsize=10)
                    values = df['PERF'].dropna()
                    if not values.empty:
                        ax.fill_betweenx(df.loc[values.index, 'DEPTH'], 0, values,
                                        color=colors['perforation'], step='pre', alpha=0.7)
                        ax.set_xlim(0, 1)
                        ax.set_xticks([0, 1])
                    else:
                        st.warning("No valid Perforations data to plot.")

                elif track == 'unperf_pay':
                    ax.set_title("Unperf Net Pay", fontsize=10)
                    values = df['UNPERF_NET_PAY'].dropna()
                    if not values.empty:
                        ax.fill_betweenx(df.loc[values.index, 'DEPTH'], 0, values,
                                        color=colors['unperf_net_pay'], step='pre', alpha=0.7)
                        ax.set_xlim(0, 1)
                        ax.set_xticks([0, 1])
                    else:
                        st.warning("No valid Unperf Net Pay data to plot.")

                elif track == 'shpor':
                    ax.set_title("SHPOR", fontsize=10)
                    if 'SHPOR' in df.columns:
                        shpor_values = df['SHPOR'].dropna()
                        if not shpor_values.empty:
                            ax.plot(shpor_values, df.loc[shpor_values.index, 'DEPTH'], 
                                    color=colors['shpor'], lw=1.5, label='SHPOR')
                            min_val = shpor_values.min() * 0.9 if shpor_values.min() > 0 else shpor_values.min() * 1.1 if shpor_values.min() < 0 else 0
                            max_val = shpor_values.max() * 1.1 if shpor_values.max() > 0 else shpor_values.max() * 0.9 if shpor_values.max() < 0 else 1
                            ax.set_xlim(min_val, max_val)
                            ax.legend(fontsize=8)
                        else:
                            st.warning("No valid SHPOR data to plot.")
                    else:
                        st.warning("SHPOR column not found in the data.")

                elif track == 'pornet':
                    ax.set_title("PORNET", fontsize=10)
                    if 'PORNET' in df.columns:
                        pornet_values = df['PORNET'].dropna()
                        if not pornet_values.empty:
                            ax.plot(pornet_values, df.loc[pornet_values.index, 'DEPTH'], 
                                    color=colors['pornet'], lw=1.5, label='PORNET')
                            min_val = pornet_values.min() * 0.9 if pornet_values.min() > 0 else pornet_values.min() * 1.1 if pornet_values.min() < 0 else 0
                            max_val = pornet_values.max() * 1.1 if pornet_values.max() > 0 else pornet_values.max() * 0.9 if pornet_values.max() < 0 else 1
                            ax.set_xlim(min_val, max_val)
                            ax.legend(fontsize=8)
                        else:
                            st.warning("No valid PORNET data to plot.")
                    else:
                        st.warning("PORNET column not found in the data.")

                ax.set_ylim(depth_range[1], depth_range[0])
                ax.tick_params(axis='both', labelsize=8)

            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

        # Summary and analysis tabs
        tab1, tab2 = st.tabs(["Summary Table", "Unperforated Net Pay"])

        with tab1:
            st.subheader("Well Log Summary")
            columns = ['DEPTH', 'PHIT', 'SW', 'VSH', 'NET_RESERVOIR', 'NET_PAY', 'PERF', 'SHPOR', 'PORNET']
            summary_df = df[[col for col in columns if col in df.columns and not df[col].isna().all()]].round(3)
            st.dataframe(summary_df, use_container_width=True)

        with tab2:
            st.subheader("Unperforated Net Pay Intervals")
            unperf_df = df[(df['NET_PAY'] == 1) & (df['PERF'] == 0)].copy() if 'NET_PAY' in df.columns and not df['NET_PAY'].isna().all() else pd.DataFrame()
            if unperf_df.empty:
                st.success("All net pay zones have been perforated! 🎉")
            else:
                unperf_df['GROUP'] = (unperf_df['DEPTH'].diff() > 0.2).cumsum()
                
                grouped = unperf_df.groupby('GROUP').agg(
                    Top=('DEPTH', 'min'),
                    Base=('DEPTH', 'max'),
                    Avg_Porosity=('PHIT', 'mean') if 'PHIT' in unperf_df.columns else ('DEPTH', lambda x: np.nan),
                    Avg_Sw=('SW', 'mean') if 'SW' in unperf_df.columns else ('DEPTH', lambda x: np.nan)
                ).reset_index(drop=True)
                
                grouped['Thickness (m)'] = (grouped['Base'] - grouped['Top']).round(2)
                if st.session_state.apply_cutoffs:
                    grouped = grouped[grouped['Thickness (m)'] >= st.session_state.ait_cutoff]
                
                if grouped.empty:
                    st.info(f"No unperforated net pay intervals meet the AIT cutoff of {st.session_state.ait_cutoff} meters.")
                else:
                    grouped['Well'] = selected_well
                    
                    grouped['Zone'] = 'Unknown'
                    if 'tops' in well:
                        tops = well['tops'].sort_values('DEPTH')
                        for i, row in grouped.iterrows():
                            valid_tops = tops[tops['DEPTH'] <= row['Top']]
                            if not valid_tops.empty:
                                grouped.at[i, 'Zone'] = clean_text(valid_tops.iloc[-1]['TOP'])
                    
                    grouped = grouped[['Well', 'Zone', 'Top', 'Base', 'Thickness (m)', 'Avg_Porosity', 'Avg_Sw']]
                    grouped['Avg_Porosity'] = grouped['Avg_Porosity'].apply(lambda x: round(x * 100, 2) if pd.notna(x) else np.nan)
                    grouped['Avg_Sw'] = grouped['Avg_Sw'].apply(lambda x: round(x * 100, 2) if pd.notna(x) else np.nan)
                    
                    st.dataframe(grouped, use_container_width=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        csv = grouped.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "Download Unperforated Intervals",
                            csv,
                            "unperforated_net_pay.csv",
                            "text/csv",
                            key="download_unperf_standard"
                        )
                    with col2:
                        all_wells_csv = get_all_wells_unperf_intervals().to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "Download All Wells Unperforated Intervals",
                            all_wells_csv,
                            "all_wells_unperforated_net_pay.csv",
                            "text/csv",
                            key="download_all_wells_unperf_standard"
                        )

    # Customized Visualization Tab
    with custom_tab:
        # Track Curve Selection
        st.subheader("Track Curve Selection")
        st.write("Select the curves to display in each track:")

        # Get all available curves from the DataFrame
        available_curves = list(df.columns)
        # Add a "None" option to allow deselecting a track
        available_curves.insert(0, "None")

        # Define track labels and default curves
        track_configs = {
            'depth': {'label': 'Depth Track', 'default': 'DEPTH', 'type': 'continuous'},
            'saturation': {'label': 'Saturation Track', 'default': 'SW', 'type': 'continuous'},
            'net_reservoir': {'label': 'Net Reservoir Track', 'default': 'NET_RESERVOIR', 'type': 'binary'},
            'porosity': {'label': 'Porosity Track', 'default': 'PHIT', 'type': 'continuous'},
            'vsh': {'label': 'VSH Track', 'default': 'VSH', 'type': 'continuous'},
            'net_pay': {'label': 'Net Pay Track', 'default': 'NET_PAY', 'type': 'binary'},
            'shpor': {'label': 'SHPOR Track', 'default': 'SHPOR', 'type': 'continuous'},
            'pornet': {'label': 'PORNET Track', 'default': 'PORNET', 'type': 'continuous'},
            'perforations': {'label': 'Perforations Track', 'default': 'PERF', 'type': 'binary'},
            'unperf_net_pay': {'label': 'Unperf Net Pay Track', 'default': 'UNPERF_NET_PAY', 'type': 'binary'}
        }

        # Check for missing curves and display a warning
        missing_curves = [config['default'] for config in track_configs.values() if config['default'] not in df.columns]
        if missing_curves:
            st.warning(f"The following logs are not available in the current dataset: {', '.join(missing_curves)}")

        # Initialize session state for selected curves
        for track in track_configs.keys():
            if f"custom_curve_{track}" not in st.session_state:
                st.session_state[f"custom_curve_{track}"] = track_configs[track]['default'] if track_configs[track]['default'] in df.columns else "None"

        # Ensure session state values are valid for the current available_curves
        for track in track_configs.keys():
            if st.session_state[f"custom_curve_{track}"] not in available_curves:
                st.session_state[f"custom_curve_{track}"] = "None"

        # Create dropdowns for each track
        cols = st.columns(2)
        for idx, (track, config) in enumerate(track_configs.items()):
            with cols[idx % 2]:
                st.session_state[f"custom_curve_{track}"] = st.selectbox(
                    config['label'],
                    options=available_curves,
                    index=available_curves.index(st.session_state[f"custom_curve_{track}"]),
                    key=f"custom_curve_select_{track}"
                )

        # Determine selected tracks (exclude "None" selections)
        selected_custom_tracks = [track for track in track_configs.keys() if st.session_state[f"custom_curve_{track}"] != "None"]

        # Plotting for Customized Visualization
        if not selected_custom_tracks:
            st.warning("Please select at least one track to display.")
        else:
            fig, axes = plt.subplots(
                figsize=(3 * len(selected_custom_tracks), 12),
                ncols=len(selected_custom_tracks),
                sharey=True,
                gridspec_kw={'wspace': 0.05}
            )
            axes = [axes] if len(selected_custom_tracks) == 1 else axes

            for i, track in enumerate(selected_custom_tracks):
                ax = axes[i]
                ax.invert_yaxis()
                ax.grid(True, alpha=0.3)
                ax.set_ylabel("Depth (m)", fontsize=10)

                selected_curve = st.session_state[f"custom_curve_{track}"]
                track_type = track_configs[track]['type']
                track_label = track_configs[track]['label'].replace(" Track", "")

                if selected_curve == "None" or selected_curve not in df.columns:
                    continue

                # Handle different track types
                if track == 'depth':
                    ax.set_title(track_label, fontsize=10)
                    ax.set_xticks([])
                elif track_type == 'continuous':
                    ax.set_title(f"{track_label} ({selected_curve})", fontsize=10)
                    values = df[selected_curve].dropna()
                    if not values.empty:
                        ax.plot(values, df.loc[values.index, 'DEPTH'], 
                                color=colors.get(track, '#000000'), lw=1.5, label=selected_curve)
                        min_val = values.min() * 0.9 if values.min() > 0 else values.min() * 1.1 if values.min() < 0 else 0
                        max_val = values.max() * 1.1 if values.max() > 0 else values.max() * 0.9 if values.max() < 0 else 1
                        ax.set_xlim(min_val, max_val)
                        ax.legend(fontsize=8)
                    else:
                        st.warning(f"No valid data for {selected_curve} to plot.")
                elif track_type == 'binary':
                    ax.set_title(f"{track_label} ({selected_curve})", fontsize=10)
                    values = df[selected_curve].dropna()
                    if not values.empty:
                        ax.fill_betweenx(df.loc[values.index, 'DEPTH'], 0, values,
                                        color=colors.get(track, '#000000'), step='pre', alpha=0.7)
                        ax.set_xlim(0, 1)
                        ax.set_xticks([0, 1])
                    else:
                        st.warning(f"No valid data for {selected_curve} to plot.")

                ax.set_ylim(depth_range[1], depth_range[0])
                ax.tick_params(axis='both', labelsize=8)

            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

        # Summary and analysis tabs for Customized Visualization
        tab1, tab2 = st.tabs(["Summary Table", "Unperforated Net Pay"])

        with tab1:
            st.subheader("Well Log Summary")
            summary_df = df.round(3)
            st.dataframe(summary_df, use_container_width=True)

        with tab2:
            st.subheader("Unperforated Net Pay Intervals")
            unperf_df = df[(df['NET_PAY'] == 1) & (df['PERF'] == 0)].copy() if 'NET_PAY' in df.columns and not df['NET_PAY'].isna().all() else pd.DataFrame()
            if unperf_df.empty:
                st.success("All net pay zones have been perforated! 🎉")
            else:
                unperf_df['GROUP'] = (unperf_df['DEPTH'].diff() > 0.2).cumsum()
                
                grouped = unperf_df.groupby('GROUP').agg(
                    Top=('DEPTH', 'min'),
                    Base=('DEPTH', 'max'),
                    Avg_Porosity=('PHIT', 'mean') if 'PHIT' in unperf_df.columns else ('DEPTH', lambda x: np.nan),
                    Avg_Sw=('SW', 'mean') if 'SW' in unperf_df.columns else ('DEPTH', lambda x: np.nan)
                ).reset_index(drop=True)
                
                grouped['Thickness (m)'] = (grouped['Base'] - grouped['Top']).round(2)
                if st.session_state.apply_cutoffs:
                    grouped = grouped[grouped['Thickness (m)'] >= st.session_state.ait_cutoff]
                
                if grouped.empty:
                    st.info(f"No unperforated net pay intervals meet the AIT cutoff of {st.session_state.ait_cutoff} meters.")
                else:
                    grouped['Well'] = selected_well
                    
                    grouped['Zone'] = 'Unknown'
                    if 'tops' in well:
                        tops = well['tops'].sort_values('DEPTH')
                        for i, row in grouped.iterrows():
                            valid_tops = tops[tops['DEPTH'] <= row['Top']]
                            if not valid_tops.empty:
                                grouped.at[i, 'Zone'] = clean_text(valid_tops.iloc[-1]['TOP'])
                    
                    grouped = grouped[['Well', 'Zone', 'Top', 'Base', 'Thickness (m)', 'Avg_Porosity', 'Avg_Sw']]
                    grouped['Avg_Porosity'] = grouped['Avg_Porosity'].apply(lambda x: round(x * 100, 2) if pd.notna(x) else np.nan)
                    grouped['Avg_Sw'] = grouped['Avg_Sw'].apply(lambda x: round(x * 100, 2) if pd.notna(x) else np.nan)
                    
                    st.dataframe(grouped, use_container_width=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        csv = grouped.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "Download Unperforated Intervals",
                            csv,
                            "unperforated_net_pay.csv",
                            "text/csv",
                            key="download_unperf_custom"
                        )
                    with col2:
                        all_wells_csv = get_all_wells_unperf_intervals().to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "Download All Wells Unperforated Intervals",
                            all_wells_csv,
                            "all_wells_unperforated_net_pay.csv",
                            "text/csv",
                            key="download_all_wells_unperf_custom"
                        )

else:
    st.info("Please upload LAS files to begin visualization.")

# Footer
st.markdown("""
---
**Well Log Visualizer** – Interactive Streamlit app for well log, tops, and perforation visualization.
Developed by Egypt Technical Team.
""", unsafe_allow_html=True)
