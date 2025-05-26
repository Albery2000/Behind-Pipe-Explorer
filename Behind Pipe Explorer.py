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
if 'apply_vsh' not in st.session_state:
    st.session_state.apply_vsh = False
if 'vsh_value' not in st.session_state:
    st.session_state.vsh_value = 50
if 'apply_sw' not in st.session_state:
    st.session_state.apply_sw = False
if 'sw_value' not in st.session_state:
    st.session_state.sw_value = 50
if 'apply_phit' not in st.session_state:
    st.session_state.apply_phit = False
if 'phit_value' not in st.session_state:
    st.session_state.phit_value = 10
if 'apply_ait' not in st.session_state:
    st.session_state.apply_ait = False
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
    # VSH Cutoff
    st.session_state.apply_vsh = st.checkbox("Apply VSH Cutoff", value=st.session_state.apply_vsh, key="vsh_checkbox")
    if st.session_state.apply_vsh:
        st.session_state.vsh_value = st.slider(
            "VSH Cutoff (%)", 0, 100, st.session_state.vsh_value, key="vsh_slider"
        )

    # Sw Cutoff
    st.session_state.apply_sw = st.checkbox("Apply Sw Cutoff", value=st.session_state.apply_sw, key="sw_checkbox")
    if st.session_state.apply_sw:
        st.session_state.sw_value = st.slider(
            "Sw Cutoff (%)", 0, 100, st.session_state.sw_value, key="sw_slider"
        )

    # Porosity Cutoff
    st.session_state.apply_phit = st.checkbox("Apply Porosity Cutoff", value=st.session_state.apply_phit, key="phit_checkbox")
    if st.session_state.apply_phit:
        st.session_state.phit_value = st.slider(
            "Porosity Cutoff (%)", 0, 30, st.session_state.phit_value, key="phit_slider"
        )

    # AIT Cutoff
    st.session_state.apply_ait = st.checkbox("Apply Unperf Pay Cutoff", value=st.session_state.apply_ait, key="ait_checkbox")
    if st.session_state.apply_ait:
        st.session_state.ait_cutoff = st.slider(
            "Unperf PayCutoff (m)", 0.0, 10.0, st.session_state.ait_cutoff, 0.1, key="ait_slider"
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
cols = st.columns(6)
color_defaults = {
    "Porosity": "#17becf",
    "Saturation": "#9467bd",
    "Net Reservoir": "#1f77b4",
    "Net Pay": "#ff7f0e",
    "Perforation": "#2ca02c",
    "Unperf Net Pay": "#dc143c"
}
colors = {}
for (label, default), col in zip(color_defaults.items(), cols):
    colors[label.lower().replace(" ", "_")] = col.color_picker(label, default, key=label.lower())

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
        expected_columns = ['WELL', 'RESERVOIR', 'TOP', 'BASE', 'DATE']
        if not all(col in perf_df.columns for col in expected_columns):
            perf_df.columns = expected_columns
        perf_df['WELL'] = perf_df['WELL'].astype(str).apply(clean_text).str.strip().str.lower()
        perf_df['RESERVOIR'] = perf_df['RESERVOIR'].astype(str).apply(clean_text)

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
        
        # Calculate net reservoir and net pay for table filtering
        if st.session_state.apply_vsh or st.session_state.apply_sw or st.session_state.apply_phit:
            if all(col in df.columns for col in ['VSH', 'SW', 'PHIT']):
                if st.session_state.apply_vsh and 'VSH' in df.columns:
                    df['NET_RESERVOIR'] = (df['VSH'] <= st.session_state.vsh_value / 100).astype(int)
                else:
                    df['NET_RESERVOIR'] = df.get('NET_RES', pd.Series(np.nan, index=df.index)).astype(float)
                
                conditions = []
                if st.session_state.apply_vsh and 'NET_RESERVOIR' in df.columns and not df['NET_RESERVOIR'].isna().all():
                    conditions.append(df['NET_RESERVOIR'] == 1)
                if st.session_state.apply_sw and 'SW' in df.columns:
                    conditions.append(df['SW'] <= st.session_state.sw_value / 100)
                if st.session_state.apply_phit and 'PHIT' in df.columns:
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
        
        # Get unperforated intervals
        unperf_df = df[(df['NET_PAY'] == 1) & (df['PERF'] == 0)].copy() if 'NET_PAY' in df.columns and not df['NET_PAY'].isna().all() else pd.DataFrame()
        if unperf_df.empty:
            continue
        
        # Group intervals
        unperf_df['GROUP'] = (unperf_df['DEPTH'].diff() > 0.2).cumsum()
        grouped = unperf_df.groupby('GROUP').agg(
            Top=('DEPTH', 'min'),
            Base=('DEPTH', 'max'),
            Avg_Porosity=('PHIT', 'mean'),
            Avg_Sw=('SW', 'mean')
        ).reset_index(drop=True)
        
        # Calculate thickness and apply AIT cutoff if enabled
        grouped['Thickness (m)'] = (grouped['Base'] - grouped['Top']).round(2)
        if st.session_state.apply_ait:
            grouped = grouped[grouped['Thickness (m)'] >= st.session_state.ait_cutoff]
        
        if grouped.empty:
            continue
        
        # Add well name
        grouped['Well'] = well_name
        
        # Assign zones
        grouped['Zone'] = 'Unknown'
        if 'tops' in well:
            tops = well['tops'].sort_values('DEPTH')
            for i, row in grouped.iterrows():
                valid_tops = tops[tops['DEPTH'] <= row['Top']]
                if not valid_tops.empty:
                    grouped.at[i, 'Zone'] = clean_text(valid_tops.iloc[-1]['TOP'])
        
        # Format columns
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
    if st.session_state.apply_vsh or st.session_state.apply_sw or st.session_state.apply_phit:
        if all(col in df.columns for col in ['VSH', 'SW', 'PHIT']):
            if st.session_state.apply_vsh and 'VSH' in df.columns:
                df['NET_RESERVOIR'] = (df['VSH'] <= st.session_state.vsh_value / 100).astype(int)
            else:
                df['NET_RESERVOIR'] = df.get('NET_RES', pd.Series(np.nan, index=df.index)).astype(float)
            
            conditions = []
            if st.session_state.apply_vsh and 'NET_RESERVOIR' in df.columns and not df['NET_RESERVOIR'].isna().all():
                conditions.append(df['NET_RESERVOIR'] == 1)
            if st.session_state.apply_sw and 'SW' in df.columns:
                conditions.append(df['SW'] <= st.session_state.sw_value / 100)
            if st.session_state.apply_phit and 'PHIT' in df.columns:
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

    # Determine available tracks
    available_tracks = []
    track_labels = {
        'tops': 'Tops',
        'phit': 'Porosity',
        'sw': 'Saturation',
        'net_reservoir': 'Net Reservoir',
        'net_pay': 'Net Pay',
        'shpor': 'SHPOR Flag',
        'pornet': 'PORNET Flag',
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
    if 'SHPOR' in df.columns:
        df['SHPOR_FLAG'] = (~df['SHPOR'].isna()).astype(int)
        available_tracks.append('shpor')
    if 'PORNET' in df.columns:
        df['PORNET_FLAG'] = (~df['PORNET'].isna()).astype(int)
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
        key="track_select"
    )

    if not selected_tracks:
        st.warning("Please select at least one track to display.")
    else:
        # Plotting
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
                if st.session_state.apply_phit:
                    ax.axvline(st.session_state.phit_value, color='red', ls=':', lw=1)
                ax.legend(fontsize=8)
                ax.set_xlim(0, df['PHIT'].max() * 100 * 1.1 if 'PHIT' in df.columns else 100)

            elif track == 'sw':
                ax.set_title("Water Saturation (%)", fontsize=10)
                ax.plot(df['SW'] * 100, df['DEPTH'], color=colors['saturation'], lw=1.5)
                if st.session_state.apply_sw:
                    ax.axvline(st.session_state.sw_value, color='red', ls=':', lw=1)
                ax.set_xlim(0, 100)

            elif track == 'net_reservoir':
                ax.set_title("Net Reservoir", fontsize=10)
                ax.fill_betweenx(df['DEPTH'], 0, df['NET_RESERVOIR'],
                                color=colors['net_reservoir'], step='pre', alpha=0.7)
                ax.set_xlim(0, 1)
                ax.set_xticks([0, 1])

            elif track == 'net_pay':
                ax.set_title("Net Pay", fontsize=10)
                ax.fill_betweenx(df['DEPTH'], 0, df['NET_PAY'],
                                color=colors['net_pay'], step='pre', alpha=0.7)
                ax.set_xlim(0, 1)
                ax.set_xticks([0, 1])

            elif track == 'perf':
                ax.set_title("Perforations", fontsize=10)
                ax.fill_betweenx(df['DEPTH'], 0, df['PERF'],
                                color=colors['perforation'], step='pre', alpha=0.7)
                ax.set_xlim(0, 1)
                ax.set_xticks([0, 1])

            elif track == 'unperf_pay':
                ax.set_title("Unperf Net Pay", fontsize=10)
                ax.fill_betweenx(df['DEPTH'], 0, df['UNPERF_NET_PAY'],
                                color=colors['unperf_net_pay'], step='pre', alpha=0.7)
                ax.set_xlim(0, 1)
                ax.set_xticks([0, 1])

            elif track == 'shpor':
                ax.set_title("SHPOR Flag", fontsize=10)
                ax.fill_betweenx(df['DEPTH'], 0, df['SHPOR_FLAG'],
                                color=colors['porosity'], step='pre', alpha=0.7)
                ax.set_xlim(0, 1)
                ax.set_xticks([0, 1])

            elif track == 'pornet':
                ax.set_title("PORNET Flag", fontsize=10)
                ax.fill_betweenx(df['DEPTH'], 0, df['PORNET_FLAG'],
                                color=colors['porosity'], step='pre', alpha=0.7)
                ax.set_xlim(0, 1)
                ax.set_xticks([0, 1])

            ax.set_ylim(depth_range[1], depth_range[0])
            ax.tick_params(axis='both', labelsize=8)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

        # Summary and analysis tabs
        tab1, tab2 = st.tabs(["Summary Table", "Unperforated Net Pay"])

        with tab1:
            st.subheader("Well Log Summary")
            columns = ['DEPTH', 'PHIT', 'SW', 'VSH', 'NET_RESERVOIR', 'NET_PAY', 'PERF']
            summary_df = df[[col for col in columns if col in df.columns and not df[col].isna().all()]].round(3)
            st.dataframe(summary_df, use_container_width=True)

        with tab2:
            st.subheader("Unperforated Net Pay Intervals")
            unperf_df = df[(df['NET_PAY'] == 1) & (df['PERF'] == 0)].copy() if 'NET_PAY' in df.columns and not df['NET_PAY'].isna().all() else pd.DataFrame()
            if unperf_df.empty:
                st.success("All net pay zones have been perforated! 🎉")
            else:
                # Group unperforated intervals
                unperf_df['GROUP'] = (unperf_df['DEPTH'].diff() > 0.2).cumsum()
                
                # Aggregate data for each group
                grouped = unperf_df.groupby('GROUP').agg(
                    Top=('DEPTH', 'min'),
                    Base=('DEPTH', 'max'),
                    Avg_Porosity=('PHIT', 'mean'),
                    Avg_Sw=('SW', 'mean')
                ).reset_index(drop=True)
                
                # Calculate thickness and apply AIT cutoff if enabled
                grouped['Thickness (m)'] = (grouped['Base'] - grouped['Top']).round(2)
                if st.session_state.apply_ait:
                    grouped = grouped[grouped['Thickness (m)'] >= st.session_state.ait_cutoff]
                
                if grouped.empty:
                    st.info(f"No unperforated net pay intervals meet the AIT cutoff of {st.session_state.ait_cutoff} meters.")
                else:
                    # Add well name
                    grouped['Well'] = selected_well
                    
                    # Assign zones based on tops data
                    grouped['Zone'] = 'Unknown'
                    if 'tops' in well:
                        tops = well['tops'].sort_values('DEPTH')
                        for i, row in grouped.iterrows():
                            valid_tops = tops[tops['DEPTH'] <= row['Top']]
                            if not valid_tops.empty:
                                grouped.at[i, 'Zone'] = clean_text(valid_tops.iloc[-1]['TOP'])
                    
                    # Reorder columns and format
                    grouped = grouped[['Well', 'Zone', 'Top', 'Base', 'Thickness (m)', 'Avg_Porosity', 'Avg_Sw']]
                    grouped['Avg_Porosity'] = grouped['Avg_Porosity'].apply(lambda x: round(x * 100, 2) if pd.notna(x) else np.nan)
                    grouped['Avg_Sw'] = grouped['Avg_Sw'].apply(lambda x: round(x * 100, 2) if pd.notna(x) else np.nan)
                    
                    # Display table
                    st.dataframe(grouped, use_container_width=True)
                    
                    # Download buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        csv = grouped.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "Download Unperforated Intervals",
                            csv,
                            "unperforated_net_pay.csv",
                            "text/csv",
                            key="download_unperf"
                        )
                    with col2:
                        all_wells_csv = get_all_wells_unperf_intervals().to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "Download All Wells Unperforated Intervals",
                            all_wells_csv,
                            "all_wells_unperforated_net_pay.csv",
                            "text/csv",
                            key="download_all_wells_unperf"
                        )

else:
    st.info("Please upload LAS files to begin visualization.")

# Footer
st.markdown("""
---
**Well Log Visualizer** – Interactive Streamlit app for well log, tops, and perforation visualization.
Developed by Egypt Technical Team.
""", unsafe_allow_html=True)