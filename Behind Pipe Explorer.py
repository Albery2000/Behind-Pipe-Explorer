import streamlit as st
import pandas as pd
import numpy as np
import lasio
import matplotlib.pyplot as plt
from io import BytesIO, StringIO
import sys
import re
from pathlib import Path
import seaborn as sns


# Set page config
st.set_page_config(
    page_title="Well Data Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Color customization
st.markdown("**Customize Track Colors**")
cols = st.columns(4)
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
for idx, (label, default) in enumerate(color_defaults.items()):
    with cols[idx % 4]:
        colors[label] = st.color_picker(
            label.replace("_", " ").title(),
            default,
            key=f"color_{label}",
            help=f"Choose a color for the {label.replace('_', ' ')} track."
        )

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
                'PHIT_D': 'PHIT', 'PHIE_T': 'PHIT','PHI_T': 'PHIT',
                'PHI_TOTAL': 'PHIT', 'SW_AR': 'SW','PHIE_D': 'PHIE',
                'PHIE_T': 'PHIE','SW_AR': 'SW', 'SW_T': 'SW',
                'SWT': 'SW', 'VSH_GR': 'VSH',
                'VSHL': 'VSH',
                'VCL': 'VSH',  'NET_PAY': 'NET_PAY','NET_RES': 'NET_RES',
                'SWT_NET': 'SW_NET', 'VSH': 'VSH', 'NET_PAY': 'NET_PAY',
                'NET_RES': 'NET_RES', 'SH_POR': 'SHPOR', 'PORNET_D': 'PORNET'
            }
            for orig, std in mapping.items():
                if orig in df.columns and std not in df.columns:
                    df[std] = df[orig]

            # Auto-scale percent curves to fraction
           for col in ['PHIT', 'PHIE', 'SW', 'VSH']:
                if col in df.columns:
                    if df[col].max() > 1.5:  # likely percentage
                       df[col] = df[col] / 100.0


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
        # Read the file based on extension
        perf_df = pd.read_csv(perf_file) if perf_file.name.endswith('.csv') else pd.read_excel(perf_file)
        
        # Check if the file has at least 5 columns
        if len(perf_df.columns) < 5:
            st.error(f"Perforation file must have at least 5 columns in order: Well, Zone, Top, Base, Status. Found {len(perf_df.columns)} columns.")
            return

        # Assign columns in the specified order
        perf_df.columns = ['WELL', 'ZONE', 'TOP', 'BASE', 'STATUS']

        # Data cleaning and processing
        perf_df['WELL'] = perf_df['WELL'].astype(str).apply(clean_text).str.strip().str.lower()
        perf_df['ZONE'] = perf_df['ZONE'].astype(str).apply(clean_text)
        perf_df['TOP'] = pd.to_numeric(perf_df['TOP'], errors='coerce')
        perf_df['BASE'] = pd.to_numeric(perf_df['BASE'], errors='coerce')
        perf_df['STATUS'] = perf_df['STATUS'].astype(str).apply(clean_text).str.strip().str.capitalize()

        # Map status to PERF values: Open -> 1, Pluggged -> -1, else (including empty) -> 0
        perf_df['PERF_VALUE'] = perf_df['STATUS'].apply(
            lambda x: 1 if x == 'Open' else -1 if x == 'Pluggged' else 0
        )

        # Drop rows with invalid Top or Base values
        perf_df = perf_df.dropna(subset=['TOP', 'BASE'])

        # Match wells with LAS data
        las_keys = {k.lower().strip(): k for k in st.session_state.well_data.keys()}
        for well in perf_df['WELL'].unique():
            if well in las_keys:
                st.session_state.well_data[las_keys[well]]['perforations'] = perf_df[perf_df['WELL'] == well]
            else:
                st.warning(f"Well '{well}' in perforation file does not match any LAS file.")
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
                df.loc[(df['DEPTH'] >= row['TOP']) & (df['DEPTH'] <= row['BASE']), 'PERF'] = row['PERF_VALUE']
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
    selected_well = st.selectbox("Select Well", list(st.session_state.well_data.keys()),help="Choose a well to visualize its log data.", key="well_select")
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
            df.loc[(df['DEPTH'] >= row['TOP']) & (df['DEPTH'] <= row['BASE']), 'PERF'] = row['PERF_VALUE']
    df['UNPERF_NET_PAY'] = ((df['NET_PAY'] == 1) & (df['PERF'] == 0)).astype(int) if 'NET_PAY' in df.columns and not df['NET_PAY'].isna().all() else pd.Series(np.nan, index=df.index)

    # Depth filter
    min_d, max_d = float(df['DEPTH'].min()), float(df['DEPTH'].max())
    depth_range = st.slider("Depth Range (m)", min_d, max_d, (min_d, max_d), 0.1, key="depth_slider")
    df = df[(df['DEPTH'] >= depth_range[0]) & (df['DEPTH'] <= depth_range[1])].copy()

    st.markdown(f'<h3 style="color: #1A3C6D;">{clean_text(selected_well)}</h3>', unsafe_allow_html=True)

    # Create tabs for Standard and Customized Visualization
    standard_tab, custom_tab, correlation_tab = st.tabs(["Standard Visualization", "Customized Visualization", "Correlation Analysis"])

    # Standard Visualization Tab
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
                figsize=(3 * len(selected_tracks), 18),  # Increased height for better visibility
                ncols=len(selected_tracks),
                gridspec_kw={'wspace': 0.1, 'hspace': 0.1}  # Added hspace for padding
            )
            axes = np.atleast_1d(axes)



            for i, track in enumerate(selected_tracks):
                ax = axes[i]
                ax.invert_yaxis()
                ax.grid(True, alpha=0.3)
                ax.set_ylabel("Depth (m)", fontsize=10)
                
                # Explicitly set and label ticks

                ax.set_ylim(depth_range[1], depth_range[0])
                ax.tick_params(axis='y', which='both', left=True, labelleft=True, labelsize=8)

                # Set y-axis label only on first track
                if i == 0:
                    ax.set_ylabel("Depth (m)", fontsize=10)
                else:
                    ax.set_ylabel("")

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
                    ax.plot(df['SW'] * 100, df['DEPTH'], color=colors['saturation'], label='Sw', lw=1.5)
                    if st.session_state.apply_cutoffs:
                        ax.axvline(st.session_state.sw_value, color='red', ls=':', lw=1)
                    ax.legend(fontsize=8)
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
                        st.warning("No valid data for Net Reservoir to plot.")

                elif track == 'net_pay':
                    ax.set_title("Net Pay", fontsize=10)
                    values = df['NET_PAY'].dropna()
                    if not values.empty:
                        ax.fill_betweenx(df.loc[values.index, 'DEPTH'], 0, values,
                                         color=colors['net_pay'], step='pre', alpha=0.7)
                        ax.set_xlim(0, 1)
                        ax.set_xticks([0, 1])
                    else:
                        st.warning("No valid data for Net Pay to plot.")

                elif track == 'perf':
                    ax.set_title("Perforations", fontsize=10)
                    values = df['PERF'].dropna()
                    if not values.empty:
                        ax.step(values, df.loc[values.index, 'DEPTH'], where='mid', color=colors['perforation'], lw=1.5)
                        ax.set_xlim(-1.5, 1.5)
                        ax.text(-1, depth_range[0], 'pluged perf', ha='right', va='bottom', fontsize=8)
                        ax.text(1, depth_range[0], 'open perf', ha='left', va='bottom', fontsize=8)
                        ax.set_xticks([-1, 0, 1])
                    else:
                        st.warning("No valid data for Perforations to plot.")

                elif track == 'unperf_pay':
                    ax.set_title("Unperf Net Pay", fontsize=10)
                    values = df['UNPERF_NET_PAY'].dropna()
                    if not values.empty:
                        ax.fill_betweenx(df.loc[values.index, 'DEPTH'], 0, values,
                                         color=colors['unperf_net_pay'], step='pre', alpha=0.7)
                        ax.set_xlim(0, 1)
                        ax.set_xticks([0, 1])
                    else:
                        st.warning("No valid data for Unperforated Net Pay to plot.")

                elif track == 'shpor':
                    ax.set_title("SHPOR", fontsize=10)
                    if 'SHPOR' in df.columns:
                        shpor_values = df['SHPOR']
                        if not shpor_values.isna().all():
                            mask = ~shpor_values.isna()
                            segments = np.split(np.where(mask)[0], np.where(np.diff(np.where(mask)[0]) > 1)[0] + 1)
                            for seg in segments:
                                if len(seg) > 1:
                                    ax.plot(shpor_values.iloc[seg], df['DEPTH'].iloc[seg], color=colors['shpor'], lw=1.5)
                            min_val = shpor_values.min() * 0.9 if shpor_values.min() > 0 else shpor_values.min() * 1.1 if shpor_values.min() < 0 else 0
                            max_val = shpor_values.max() * 1.1 if shpor_values.max() > 0 else shpor_values.max() * 0.9 if shpor_values.max() < 0 else 1
                            ax.set_xlim(min_val, max_val)
                            ax.legend([f'SHPOR'], fontsize=8)
                        else:
                            st.warning("No valid data for SHPOR to plot.")
                    else:
                        st.warning("SHPOR column not found in data.")

                elif track == 'pornet':
                    ax.set_title("PORNET", fontsize=10)
                    if 'PORNET' in df.columns:
                        pornet_values = df['PORNET']
                        if not pornet_values.isna().all():
                            mask = ~pornet_values.isna()
                            segments = np.split(np.where(mask)[0], np.where(np.diff(np.where(mask)[0]) > 1)[0] + 1)
                            for seg in segments:
                                if len(seg) > 1:
                                    ax.plot(pornet_values.iloc[seg], df['DEPTH'].iloc[seg], color=colors['pornet'], lw=1.5)
                            min_val = pornet_values.min() * 0.9 if pornet_values.min() > 0 else pornet_values.min() * 1.1 if pornet_values.min() < 0 else 0
                            max_val = pornet_values.max() * 1.1 if pornet_values.max() > 0 else pornet_values.max() * 0.9 if pornet_values.max() < 0 else 1
                            ax.set_xlim(min_val, max_val)
                            ax.legend([f'PORNET'], fontsize=8)
                        else:
                            st.warning("No valid data for PORNET to plot.")
                    else:
                        st.warning("PORNET column not found in data.")

            plt.tight_layout(pad=2.0, h_pad=1.0)  # Increased padding and h_pad
            st.pyplot(fig, use_container_width=True)
            # Uncomment the line below for local debugging (run script outside Streamlit)
            # plt.show()

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
        st.subheader("Track Curve Customization")
        st.write("Select curves to display in each track:")

        # Get all available curves from the DataFrame
        available_curves = list(df.columns)
        available_curves.insert(0, "None")

        # Define track labels and default curves
        track_configs = {
            'depth': {'label': 'Depth Track', 'default': 'DEPTH', 'type': 'depth'},
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
        missing_curves = [config['label'] for track, config in track_configs.items() if config['default'] not in df.columns]
        if missing_curves:
            st.warning(f"Following logs not available: {', '.join(missing_curves)}")

        # Initialize session state for selected curves
        for track in track_configs.keys():
            if f"custom_curve_{track}" not in st.session_state:
                st.session_state[f"custom_curve_{track}"] = track_configs[track]['default'] if track_configs[track]['default'] in df.columns else "None"

        # Ensure session state values are valid
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

        # Determine selected tracks
        selected_custom_tracks = [track for track in track_configs.keys() if st.session_state[f"custom_curve_{track}"] != "None"]

        # Plotting for customized visualization
        if not selected_custom_tracks:
            st.warning("Select at least one track to display.")
        else:
            fig, axes = plt.subplots(
                figsize=(3 * len(selected_custom_tracks), 18),  # Increased height for better visibility
                ncols=len(selected_custom_tracks),
                gridspec_kw={'wspace': 0.1, 'hspace': 0.1}  # Added hspace for padding
            )
            axes = np.atleast_1d(axes)

            # Calculate and debug tick intervals
            depth_min, depth_max = depth_range[1], depth_range[0]
            tick_interval = max(10, (depth_max - depth_min) / 10)
            major_ticks = np.arange(
                np.floor(depth_min / tick_interval) * tick_interval,
                np.ceil(depth_max / tick_interval) * tick_interval + tick_interval,
                tick_interval
            )
            st.write(f"Debug: Major ticks = {major_ticks}")  # Debug output

            for i, track in enumerate(selected_custom_tracks):
                ax = axes[i]
                ax.invert_yaxis()
                ax.grid(True, alpha=0.3)

                # Explicitly set and label ticks
                ax.set_yticks(major_ticks)
                ax.set_yticklabels([f"{tick:.0f}" for tick in major_ticks], fontsize=8)  # Force tick labels
                ax.set_ylim(depth_range[1], depth_range[0])
                ax.tick_params(axis='y', which='both', left=True, labelleft=True, labelsize=8)

                # Set y-axis label only on first track
                if i == 0:
                    ax.set_ylabel("Depth (m)", fontsize=10)
                else:
                    ax.set_ylabel("")

                selected_curve = st.session_state[f"custom_curve_{track}"]
                track_type = track_configs[track]['type']
                track_label = track_configs[track]['label'].replace(" Track", "")

                if selected_curve == "None" or selected_curve not in df.columns:
                    continue

                # Handle different track types
                if track == 'depth':
                    ax.set_title(track_label, fontsize=10)
                    ax.plot(df['DEPTH'], df['DEPTH'], color='black', lw=0)  # Invisible line to set depth axis
                    ax.set_xlim(depth_range[0], depth_range[1])
                    ax.set_xticks([])

                elif track_type == 'continuous':
                    ax.set_title(f"{track_label} ({selected_curve})", fontsize=10)
                    values = df[selected_curve]
                    if not values.isna().all():
                        if track in ['shpor', 'pornet']:
                            mask = ~values.isna()
                            segments = np.split(np.where(mask)[0], np.where(np.diff(np.where(mask)[0]) > 1)[0] + 1)
                            for seg in segments:
                                if len(seg) > 1:
                                    ax.plot(values.iloc[seg], df['DEPTH'].iloc[seg], color=colors.get(track, '#000000'), lw=1.5)
                            min_val = values.min() * 0.9 if values.min() > 0 else values.min() * 1.1 if values.min() < 0 else 0
                            max_val = values.max() * 1.1 if values.max() > 0 else values.max() * 0.9 if values.max() < 0 else 1
                            ax.set_xlim(min_val, max_val)
                        else:
                            ax.plot(values, df['DEPTH'], color=colors.get(track, '#000000'), lw=1.5, label=selected_curve)
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
                        if selected_curve == 'PERF':
                            ax.step(values, df.loc[values.index, 'DEPTH'], where='mid', color=colors.get(track, '#000000'), lw=1.5)
                            ax.set_xlim(-1.5, 1.5)
                            ax.text(-1, depth_range[0], 'pluged perf', ha='right', va='bottom', fontsize=8)
                            ax.text(1, depth_range[0], 'open perf', ha='left', va='bottom', fontsize=8)
                            ax.set_xticks([-1, 0, 1])
                        else:
                            ax.fill_betweenx(
                                df.loc[values.index, 'DEPTH'],
                                0, values,
                                color=colors.get(track, '#000000'),
                                step='pre',
                                alpha=0.7
                            )
                            ax.set_xlim(0, 1)
                            ax.set_xticks([0, 1])
                    else:
                        st.warning(f"No valid data for {selected_curve} to plot.")

            plt.tight_layout(pad=2.0, h_pad=1.0)  # Increased padding and h_pad
            st.pyplot(fig, use_container_width=True)
    with correlation_tab:
          st.subheader("Well Log Parameter Correlation Analysis")
    
    # Select wells for correlation analysis
    selected_wells = st.multiselect(
        "Select Wells for Correlation",
        options=list(st.session_state.well_data.keys()),
        default=list(st.session_state.well_data.keys())[:1] if st.session_state.well_data else [],
        key="corr_well_select"
    )
    
    if selected_wells:
        # Get common parameters across selected wells
        common_params = set()
        for well in selected_wells:
            df = st.session_state.well_data[well]['data']
            common_params.update(df.columns)
        
        # Remove non-numeric parameters
        numeric_params = []
        for param in common_params:
            for well in selected_wells:
                df = st.session_state.well_data[well]['data']
                if param in df.columns and pd.api.types.is_numeric_dtype(df[param]):
                    numeric_params.append(param)
                    break
        
        if not numeric_params:
            st.warning("No numeric parameters available for correlation analysis.")
        else:
            # Parameter selection
            param1 = st.selectbox("Select First Parameter", numeric_params, key="corr_param1")
            param2 = st.selectbox("Select Second Parameter", numeric_params, key="corr_param2")
            
            # Depth range selection
            min_depth = min([st.session_state.well_data[well]['data']['DEPTH'].min() for well in selected_wells])
            max_depth = max([st.session_state.well_data[well]['data']['DEPTH'].max() for well in selected_wells])
            depth_range_corr = st.slider(
                "Depth Range for Correlation (m)",
                min_depth, max_depth, (min_depth, max_depth), 0.1,
                key="depth_slider_corr"
            )
            
            # Collect data from all selected wells
            all_data = []
            for well in selected_wells:
                df = st.session_state.well_data[well]['data']
                mask = (df['DEPTH'] >= depth_range_corr[0]) & (df['DEPTH'] <= depth_range_corr[1])
                filtered = df[mask].copy()
                filtered['WELL'] = well
                all_data.append(filtered[[param1, param2, 'WELL']])
            
            if not all_data:
                st.warning("No data available in the selected depth range.")
            else:
                combined = pd.concat(all_data).dropna()
                
                if combined.empty:
                    st.warning("No common data points available for correlation.")
                else:
                    # Calculate correlation matrix
                    corr_matrix = combined.groupby('WELL')[[param1, param2]].corr().unstack().iloc[:, 1]
                    corr_matrix = corr_matrix.to_frame(name='Correlation').reset_index()
                    
                    # Display correlation results
                    st.subheader("Correlation Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.dataframe(corr_matrix.style.format({'Correlation': '{:.3f}'}), use_container_width=True)
                    
                    with col2:
                        st.metric(
                            "Overall Correlation",
                            f"{combined[[param1, param2]].corr().iloc[0,1]:.3f}",
                            help="Pearson correlation coefficient for all selected wells combined"
                        )
                    
                    # Plot scatter plot with regression line
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.regplot(
                        data=combined,
                        x=param1,
                        y=param2,
                        scatter_kws={'alpha': 0.5},
                        line_kws={'color': 'red'},
                        ax=ax
                    )
                    
                    # Add well names to legend if multiple wells
                    if len(selected_wells) > 1:
                        for well in selected_wells:
                            well_data = combined[combined['WELL'] == well]
                            ax.scatter(
                                well_data[param1],
                                well_data[param2],
                                alpha=0.5,
                                label=well
                            )
                        ax.legend()
                    
                    ax.set_xlabel(param1)
                    ax.set_ylabel(param2)
                    ax.set_title(f"{param1} vs {param2} Correlation")
                    st.pyplot(fig)
                    
                    # Pairplot for multiple parameters
                    if len(numeric_params) > 2:
                        st.subheader("Multi-Parameter Correlation Analysis")
                        selected_multi_params = st.multiselect(
                            "Select Parameters for Pairplot",
                            numeric_params,
                            default=numeric_params[:3],
                            key="multi_param_select"
                        )
                        
                        if len(selected_multi_params) >= 2:
                            pairplot_data = combined[selected_multi_params + ['WELL']].dropna()
                            if not pairplot_data.empty:
                                fig = sns.pairplot(
                                    pairplot_data,
                                    hue='WELL' if len(selected_wells) > 1 else None,
                                    diag_kind='kde',
                                    plot_kws={'alpha': 0.5}
                                )
                                st.pyplot(fig)
                            else:
                                st.warning("No common data points available for selected parameters.")
        # Summary and analysis tabs
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
    st.info("Upload LAS files to begin visualization.")

# Footer
st.markdown('''
---
**Streamlit App** – Interactive well log, tops, and perforation visualization.  
Developed by Egypt Technical Team.
''', unsafe_allow_html=True)


