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
        "Show Perforations": True,
        "Show ResFlag": True,
        "Show PayFlag": True
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
    "pornet": "#c5b0d5",
    "vsh": "#8c564b",
    "resflag": "#3498db",
    "payflag": "#e74c3c"
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

# Function to calculate unperforated pay (using PAYFLAG from CPI)
def calculate_unperf_pay(df):
    """Calculate unperforated net pay using PAYFLAG from CPI"""
    df = df.copy()
    
    # Create PERF column if not exists
    if 'PERF' not in df.columns:
        df['PERF'] = 0
    
    # Calculate unperforated net pay using PAYFLAG
    if 'PAYFLAG' in df.columns:
        # Ensure PAYFLAG is binary (0 or 1)
        df['PAYFLAG'] = df['PAYFLAG'].fillna(0).clip(0, 1).astype(int)
        df['UNPERF_NET_PAY'] = ((df['PAYFLAG'] == 1) & (df['PERF'] == 0)).astype(int)
    else:
        df['UNPERF_NET_PAY'] = 0
    
    return df

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

            # Standardize column names with focus on CPI-derived flags
            mapping = {
                'PHIT_D': 'PHIT', 'PHIE_T': 'PHIT','PHI_T': 'PHIT',
                'PHI_TOTAL': 'PHIT', 'SW_AR': 'SW','PHIE_D': 'PHIE',
                'PHIE_T': 'PHIE','SW_AR': 'SW', 'SW_T': 'SW',
                'SWT': 'SW', 'VSH_GR': 'VSH',
                'VSHL': 'VSH',
                'VCL': 'VSH', 
                'SWT_NET': 'SW_NET', 'VSH': 'VSH',
                'SH_POR': 'SHPOR', 'PORNET_D': 'PORNET',
                'RESFLAG': 'RESFLAG', 'PAYFLAG': 'PAYFLAG',
                # Common CPI flag names
                'PAY_FLAG': 'PAYFLAG', 'PAY': 'PAYFLAG', 'PAYFLG': 'PAYFLAG',
                'RES_FLAG': 'RESFLAG', 'RES': 'RESFLAG', 'RESFLG': 'RESFLAG',
                'RESERVOIR_FLAG': 'RESFLAG'
            }
            for orig, std in mapping.items():
                if orig in df.columns and std not in df.columns:
                    df[std] = df[orig]
            
            # --- AUTO-DETECT & NORMALIZE PERCENT LOGS ---
            percent_logs = ['PHIT', 'PHIE', 'SW', 'VSH', 'VGLAU', 'VILITE', 'VLIME', 'VOIL', 'VSAND', 'VSILT', 'VWATER']
            for col in percent_logs:
                if col in df.columns:
                    if df[col].max() > 1.5:   # percent data
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
            
            # Ensure RESFLAG and PAYFLAG are properly formatted
            for flag in ['RESFLAG', 'PAYFLAG']:
                if flag in df.columns:
                    # Convert to binary (0 or 1)
                    df[flag] = df[flag].fillna(0)
                    # Handle various formats
                    if df[flag].dtype == object:
                        # String values like 'YES', 'NO', '1', '0'
                        df[flag] = df[flag].astype(str).str.upper()
                        df[flag] = df[flag].map({
                            'YES': 1, 'Y': 1, '1': 1, 'TRUE': 1, 'T': 1,
                            'NO': 0, 'N': 0, '0': 0, 'FALSE': 0, 'F': 0
                        }).fillna(0)
                    # Ensure integer and clip to 0-1
                    df[flag] = df[flag].astype(float).fillna(0).clip(0, 1).astype(int)
            
            # Calculate unperforated net pay
            df = calculate_unperf_pay(df)
            
            st.session_state.well_data[well_name] = {
                'data': df,
                'las': las,
                'header': clean_text(str(las.header)),
                'available_curves': list(df.columns)
            }
            
            # Show available curves for debugging
            with st.sidebar.expander(f"Curves in {well_name}"):
                st.write(f"Available curves: {', '.join(df.columns)}")
                if 'RESFLAG' in df.columns:
                    st.write(f"RESFLAG values: {df['RESFLAG'].unique()}")
                if 'PAYFLAG' in df.columns:
                    st.write(f"PAYFLAG values: {df['PAYFLAG'].unique()}")
                
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

# Function to assign formation based on depth and tops
def assign_formation(depth, tops_df):
    """Assign formation name based on depth and tops data"""
    if tops_df is None or tops_df.empty:
        return 'Unknown'
    
    # Sort tops by depth
    tops_sorted = tops_df.sort_values('DEPTH')
    
    # Find the formation for this depth
    formation = 'Unknown'
    for i in range(len(tops_sorted) - 1):
        if depth >= tops_sorted.iloc[i]['DEPTH'] and depth < tops_sorted.iloc[i + 1]['DEPTH']:
            formation = tops_sorted.iloc[i]['TOP']
            break
    else:
        # If depth is below the last top or above the first top
        if depth >= tops_sorted.iloc[-1]['DEPTH']:
            formation = tops_sorted.iloc[-1]['TOP'] + " (Base)"
        elif depth < tops_sorted.iloc[0]['DEPTH']:
            formation = "Above " + tops_sorted.iloc[0]['TOP']
    
    return formation

# Function to get unperforated net pay intervals for all wells using PAYFLAG
def get_all_wells_unperf_intervals() -> pd.DataFrame:
    all_intervals = []
    for well_name, well in st.session_state.well_data.items():
        df = well['data'].copy()
        
        # Apply cutoffs if selected (only affects calculations, not PAYFLAG)
        if st.session_state.apply_cutoffs:
            # Start with all rows and columns
            filtered_df = df.copy()
            conditions = []
            
            # Porosity condition
            if 'PHIT' in filtered_df.columns:
                phit_series = filtered_df['PHIT'].copy()
                if phit_series.max() > 1.5:
                    phit_series = phit_series / 100
                conditions.append(phit_series >= st.session_state.phit_value / 100)
            
            # Sw condition
            if 'SW' in filtered_df.columns:
                sw_series = filtered_df['SW'].copy()
                if sw_series.max() > 1.5:
                    sw_series = sw_series / 100
                conditions.append(sw_series <= st.session_state.sw_value / 100)
            
            # VSH condition for reservoir quality
            if 'VSH' in filtered_df.columns:
                vsh_series = filtered_df['VSH'].copy()
                if vsh_series.max() > 1.5:
                    vsh_series = vsh_series / 100
                conditions.append(vsh_series <= st.session_state.vsh_value / 100)
            
            # Apply conditions if we have them
            if conditions:
                filter_mask = np.all(conditions, axis=0)
                # Filter rows but keep all columns (including DEPTH)
                filtered_df = filtered_df[filter_mask].copy()
            
            # Use PAYFLAG from CPI
            if 'PAYFLAG' in filtered_df.columns:
                pay_mask = filtered_df['PAYFLAG'] == 1
                filtered_df = filtered_df[pay_mask].copy()
        else:
            # Simply use PAYFLAG from CPI
            if 'PAYFLAG' in df.columns:
                filtered_df = df[df['PAYFLAG'] == 1].copy()
            else:
                filtered_df = pd.DataFrame()
        
        if filtered_df.empty:
            continue
            
        # Process perforations if available
        if 'PERF' not in filtered_df.columns:
            filtered_df['PERF'] = 0
        
        if 'perforations' in well and st.session_state.show_perforations:
            # Ensure DEPTH column exists before processing perforations
            if 'DEPTH' in filtered_df.columns:
                for _, row in well['perforations'].iterrows():
                    filtered_df.loc[(filtered_df['DEPTH'] >= row['TOP']) & 
                                   (filtered_df['DEPTH'] <= row['BASE']), 'PERF'] = row['PERF_VALUE']
            else:
                st.warning(f"No DEPTH column found in filtered data for {well_name}. Skipping perforation processing.")
        
        # Calculate unperforated net pay
        filtered_df['UNPERF_NET_PAY'] = (filtered_df['PERF'] == 0).astype(int)
        
        # Filter for unperforated intervals only
        unperf_df = filtered_df[filtered_df['UNPERF_NET_PAY'] == 1].copy()
        if unperf_df.empty:
            continue
        
        # Sort by depth and create continuous intervals
        unperf_df = unperf_df.sort_values('DEPTH')
        unperf_df['GROUP'] = 0
        group_num = 0
        
        if len(unperf_df) > 0:
            unperf_df.iloc[0, unperf_df.columns.get_loc('GROUP')] = group_num
            
            for i in range(1, len(unperf_df)):
                current_depth = unperf_df.iloc[i]['DEPTH']
                prev_depth = unperf_df.iloc[i-1]['DEPTH']
                
                # If depths are not consecutive (more than 0.6 m apart), start new group
                if current_depth - prev_depth > 0.6:
                    group_num += 1
                
                unperf_df.iloc[i, unperf_df.columns.get_loc('GROUP')] = group_num
        
        # Group by the GROUP column
        grouped = unperf_df.groupby('GROUP').agg(
            Top=('DEPTH', 'min'),
            Base=('DEPTH', 'max'),
            Avg_Porosity=('PHIT', 'mean') if 'PHIT' in unperf_df.columns else ('DEPTH', lambda x: np.nan),
            Avg_Sw=('SW', 'mean') if 'SW' in unperf_df.columns else ('DEPTH', lambda x: np.nan),
            Avg_VSH=('VSH', 'mean') if 'VSH' in unperf_df.columns else ('DEPTH', lambda x: np.nan),
            Data_Points=('DEPTH', 'count')
        ).reset_index(drop=True)
        
        # Calculate thickness
        grouped['Thickness (m)'] = (grouped['Base'] - grouped['Top']).round(2)
        
        # Estimate thickness for single-point intervals
        for idx, row in grouped.iterrows():
            if row['Data_Points'] == 1:
                grouped.at[idx, 'Thickness (m)'] = 0.1524
                grouped.at[idx, 'Base'] = row['Top'] + 0.1524
        
        # Apply AIT cutoff
        if st.session_state.apply_cutoffs:
            grouped = grouped[grouped['Thickness (m)'] >= st.session_state.ait_cutoff]
        
        if grouped.empty:
            continue
        
        grouped['Well'] = well_name
        
        # Determine zone from tops - using the improved assign_formation function
        grouped['Zone'] = 'Unknown'
        if 'tops' in well:
            tops = well['tops']
            if not tops.empty and 'DEPTH' in tops.columns and 'TOP' in tops.columns:
                try:
                    tops = tops.sort_values('DEPTH')
                    for i, row in grouped.iterrows():
                        # For each interval, assign the formation based on the midpoint depth
                        mid_depth = (row['Top'] + row['Base']) / 2
                        grouped.at[i, 'Zone'] = assign_formation(mid_depth, tops)
                except Exception as e:
                    st.warning(f"Error processing tops for {well_name}: {str(e)}")
        
        # Prepare final columns
        final_columns = ['Well', 'Zone', 'Top', 'Base', 'Thickness (m)', 'Avg_Porosity', 'Avg_Sw', 'Avg_VSH']
        grouped = grouped[final_columns]
        
        # Convert to percentages for display
        for col in ['Avg_Porosity', 'Avg_Sw', 'Avg_VSH']:
            if col in grouped.columns:
                grouped[col] = grouped[col].apply(lambda x: round(x * 100, 2) if pd.notna(x) else np.nan)
        
        all_intervals.append(grouped)
    
    if all_intervals:
        result = pd.concat(all_intervals, ignore_index=True)
    else:
        result = pd.DataFrame(columns=['Well', 'Zone', 'Top', 'Base', 'Thickness (m)', 'Avg_Porosity', 'Avg_Sw', 'Avg_VSH'])
    
    return result

# Function to create formation breakdown table similar to the image
def create_formation_breakdown(all_intervals_df):
    """Create a formation breakdown table similar to the image provided"""
    if all_intervals_df.empty:
        return pd.DataFrame()
    
    # Pivot the data to create a matrix of Well x Formation with Thickness
    formation_breakdown = all_intervals_df.pivot_table(
        index='Well', 
        columns='Zone', 
        values='Thickness (m)',
        aggfunc='sum',
        fill_value=0
    ).round(2)
    
    # Add a Total column
    formation_breakdown['Total'] = formation_breakdown.sum(axis=1).round(2)
    
    # Sort wells by total thickness
    formation_breakdown = formation_breakdown.sort_values('Total', ascending=False)
    
    # Reset index to make Well a column
    formation_breakdown = formation_breakdown.reset_index()
    
    return formation_breakdown

# Function to create a styled formation breakdown table
def style_formation_table(df):
    """Apply styling to the formation breakdown table"""
    if df.empty:
        return df
    
    # Create a copy for styling
    styled_df = df.copy()
    
    # Add color coding based on thickness values
    def color_thickness(val):
        if pd.isna(val) or val == 0:
            return 'background-color: #f0f0f0'
        elif val >= 5:
            return 'background-color: #90EE90'  # Light green for thick
        elif val >= 2:
            return 'background-color: #FFD700'  # Gold for medium
        else:
            return 'background-color: #FFB6C1'  # Light pink for thin
    
    return styled_df

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
    
    # Process perforations
    df['PERF'] = 0
    if 'perforations' in well and st.session_state.show_perforations:
        for _, row in well['perforations'].iterrows():
            df.loc[(df['DEPTH'] >= row['TOP']) & (df['DEPTH'] <= row['BASE']), 'PERF'] = row['PERF_VALUE']
    
    # Calculate unperforated net pay using PAYFLAG
    df = calculate_unperf_pay(df)
    
    # Apply cutoffs if selected (only affects filtering, not recalculating flags)
    if st.session_state.apply_cutoffs:
        filtered_mask = pd.Series(True, index=df.index)
        
        # Apply porosity cutoff
        if 'PHIT' in df.columns:
            phit_series = df['PHIT'].copy()
            if phit_series.max() > 1.5:
                phit_series = phit_series / 100
            filtered_mask &= (phit_series >= st.session_state.phit_value / 100)
        
        # Apply Sw cutoff
        if 'SW' in df.columns:
            sw_series = df['SW'].copy()
            if sw_series.max() > 1.5:
                sw_series = sw_series / 100
            filtered_mask &= (sw_series <= st.session_state.sw_value / 100)
        
        # Apply VSH cutoff
        if 'VSH' in df.columns:
            vsh_series = df['VSH'].copy()
            if vsh_series.max() > 1.5:
                vsh_series = vsh_series / 100
            filtered_mask &= (vsh_series <= st.session_state.vsh_value / 100)
        
        # Store original data but use filtered for display
        filtered_df = df[filtered_mask].copy()
    else:
        filtered_df = df.copy()

    # Depth filter
    min_d, max_d = float(df['DEPTH'].min()), float(df['DEPTH'].max())
    depth_range = st.slider("Depth Range (m)", min_d, max_d, (min_d, max_d), 0.1, key="depth_slider")
    display_df = filtered_df[(filtered_df['DEPTH'] >= depth_range[0]) & (filtered_df['DEPTH'] <= depth_range[1])].copy()

    st.markdown(f'<h3 style="color: #1A3C6D;">{clean_text(selected_well)}</h3>', unsafe_allow_html=True)
    
    # Show available curves in sidebar for reference
    with st.sidebar.expander(f"Available curves in {selected_well}"):
        st.write(", ".join(df.columns.tolist()))
        if 'RESFLAG' in df.columns:
            resflag_stats = df['RESFLAG'].value_counts()
            st.write(f"RESFLAG distribution: {dict(resflag_stats)}")
        if 'PAYFLAG' in df.columns:
            payflag_stats = df['PAYFLAG'].value_counts()
            st.write(f"PAYFLAG distribution: {dict(payflag_stats)}")

    # Create tabs for Standard and Customized Visualization only
    standard_tab, custom_tab = st.tabs(["Standard Visualization", "Customized Visualization"])

    # Standard Visualization Tab
    with standard_tab:
        # Determine available tracks
        available_tracks = []
        track_labels = {
            'tops': 'Tops',
            'phit': 'Porosity (PHIT/PHIE)',
            'sw': 'Water Saturation (SW)',
            'net_reservoir': 'Net Reservoir',
            'net_pay': 'Net Pay',
            'shpor': 'SHPOR',
            'pornet': 'PORNET',
            'perf': 'Perforations',
            'unperf_pay': 'Unperf Net Pay',
            'vsh': 'Clay Volume (VSH)',
            'resflag': 'Reservoir Flag (RESFLAG)',
            'payflag': 'Pay Flag (PAYFLAG)',
        }

        # Check which tracks have data
        if 'tops' in well and st.session_state.show_colored_tops_track:
            # Only add tops track if tops data is not empty
            if not well['tops'].empty:
                available_tracks.append('tops')

        # Check for porosity logs
        porosity_logs = ['PHIT', 'PHIE']
        if any(col in display_df.columns for col in porosity_logs) and st.session_state.show_porosity:
            available_tracks.append('phit')

        # Check for saturation logs
        if 'SW' in display_df.columns and st.session_state.show_saturation:
            available_tracks.append('sw')

        # Check for VSH
        if 'VSH' in display_df.columns:
            available_tracks.append('vsh')

        # Check for ResFlag (from CPI)
        if 'RESFLAG' in display_df.columns and not display_df['RESFLAG'].isna().all() and st.session_state.show_resflag:
            available_tracks.append('resflag')

        # Check for PayFlag (from CPI)
        if 'PAYFLAG' in display_df.columns and not display_df['PAYFLAG'].isna().all() and st.session_state.show_payflag:
            available_tracks.append('payflag')

        # Check for special logs
        if 'SHPOR' in display_df.columns and st.session_state.show_porosity:
            available_tracks.append('shpor')
        if 'PORNET' in display_df.columns and st.session_state.show_porosity:
            available_tracks.append('pornet')

        # Check for perforations
        if 'PERF' in display_df.columns and st.session_state.show_perforations:
            available_tracks.append('perf')

        # Check for unperforated net pay (calculated from PAYFLAG and PERF)
        if 'UNPERF_NET_PAY' in display_df.columns and not display_df['UNPERF_NET_PAY'].isna().all():
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
                figsize=(3 * len(selected_tracks), 18),
                ncols=len(selected_tracks),
                gridspec_kw={'wspace': 0.1, 'hspace': 0.1}
            )
            axes = np.atleast_1d(axes)

            for i, track in enumerate(selected_tracks):
                ax = axes[i]
                ax.invert_yaxis()
                ax.grid(True, alpha=0.3)
                ax.set_ylabel("Depth (m)", fontsize=10)
                
                # Set y-axis limits
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
                        if not tops.empty:
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
    
                    # Plot PHIT if available
                    if 'PHIT' in display_df.columns:
                        phit_values = display_df['PHIT'].copy()
                        # Normalize to percentage for display
                        if phit_values.max() <= 1.0:
                            phit_values = phit_values * 100
                        elif phit_values.max() > 1.0 and phit_values.max() <= 1.5:
                            phit_values = phit_values * 100
                        ax.plot(phit_values, display_df['DEPTH'], color=colors['porosity'], label='PHIT', lw=1.5)
    
                    # Plot PHIE if available (as dashed line)
                    if 'PHIE' in display_df.columns:
                        phie_values = display_df['PHIE'].copy()
                        if phie_values.max() <= 1.0:
                            phie_values = phie_values * 100
                        elif phie_values.max() > 1.0 and phie_values.max() <= 1.5:
                            phie_values = phie_values * 100
                        ax.plot(phie_values, display_df['DEPTH'], color=colors['porosity'], ls='--', label='PHIE', lw=1)
    
                    # Add cutoff line if applicable
                    if st.session_state.apply_cutoffs:
                        ax.axvline(st.session_state.phit_value, color='red', ls=':', lw=1, label=f'Cutoff: {st.session_state.phit_value}%')

                    ax.set_xlim(0, 50)
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)

                elif track == 'sw':
                    ax.set_title("Water Saturation (%)", fontsize=10)
                    if 'SW' in display_df.columns:
                        sw_values = display_df['SW'].copy()
                        if sw_values.max() <= 1.0:
                            sw_values = sw_values * 100
                        elif sw_values.max() > 1.0 and sw_values.max() <= 1.5:
                            sw_values = sw_values * 100
        
                        ax.plot(sw_values, display_df['DEPTH'], color=colors['saturation'], label='Sw', lw=1.5)
        
                        if st.session_state.apply_cutoffs:
                            ax.axvline(st.session_state.sw_value, color='red', ls=':', lw=1, label=f'Cutoff: {st.session_state.sw_value}%')
        
                        ax.set_xlim(0, 100)
                        ax.legend(fontsize=8)
                        ax.grid(True, alpha=0.3)

                elif track == 'vsh':
                    ax.set_title("Clay Volume - VSH (%)", fontsize=10)
                    if 'VSH' in display_df.columns:
                        vsh_values = display_df['VSH'].copy()
                        if vsh_values.max() <= 1.0:
                            vsh_values = vsh_values * 100
                        elif vsh_values.max() > 1.0 and vsh_values.max() <= 1.5:
                            vsh_values = vsh_values * 100
                        
                        ax.plot(vsh_values, display_df['DEPTH'], color='#8c564b', label='VSH', lw=1.5)
        
                        if st.session_state.apply_cutoffs:
                            ax.axvline(st.session_state.vsh_value, color='red', ls=':', lw=1, label=f'Cutoff: {st.session_state.vsh_value}%')
        
                        ax.set_xlim(0, 100)
                        ax.legend(fontsize=8)
                        ax.grid(True, alpha=0.3)

                elif track == 'resflag':
                    ax.set_title("Reservoir Flag (from CPI)", fontsize=10)
                    if 'RESFLAG' in display_df.columns:
                        values = display_df['RESFLAG'].dropna()
                        if not values.empty:
                            values = values.clip(0, 1).astype(int)
                            ax.fill_betweenx(display_df.loc[values.index, 'DEPTH'], 0, values,
                                             color=colors['resflag'], step='pre', alpha=0.7)
                            ax.set_xlim(0, 1)
                            ax.set_xticks([0, 1])
                            ax.text(0.5, depth_range[0] + 5, 'Reservoir Zone', 
                                    ha='center', va='bottom', fontsize=7)
                        else:
                            st.warning("No RESFLAG data to plot.")

                elif track == 'payflag':
                    ax.set_title("Pay Flag (from CPI)", fontsize=10)
                    if 'PAYFLAG' in display_df.columns:
                        values = display_df['PAYFLAG'].dropna()
                        if not values.empty:
                            values = values.clip(0, 1).astype(int)
                            ax.fill_betweenx(display_df.loc[values.index, 'DEPTH'], 0, values,
                                             color=colors['payflag'], step='pre', alpha=0.7)
                            ax.set_xlim(0, 1)
                            ax.set_xticks([0, 1])
                            ax.text(0.5, depth_range[0] + 5, 'Pay Zone', 
                                    ha='center', va='bottom', fontsize=7)
                        else:
                            st.warning("No PAYFLAG data to plot.")

                elif track == 'perf':
                    ax.set_title("Perforations", fontsize=10)
                    if 'PERF' in display_df.columns:
                        values = display_df['PERF'].dropna()
                        if not values.empty:
                            ax.step(values, display_df.loc[values.index, 'DEPTH'], where='mid', 
                                   color=colors['perforation'], lw=1.5)
                            ax.set_xlim(-1.5, 1.5)
                            ax.text(-1, depth_range[0], 'plugged perf', ha='right', va='bottom', fontsize=8)
                            ax.text(1, depth_range[0], 'open perf', ha='left', va='bottom', fontsize=8)
                            ax.set_xticks([-1, 0, 1])
                        else:
                            st.warning("No perforation data to plot.")

                elif track == 'unperf_pay':
                    ax.set_title("Unperf Net Pay", fontsize=10)
                    if 'UNPERF_NET_PAY' in display_df.columns:
                        values = display_df['UNPERF_NET_PAY'].dropna()
                        if not values.empty:
                            ax.fill_betweenx(display_df.loc[values.index, 'DEPTH'], 0, values,
                                             color=colors['unperf_net_pay'], step='pre', alpha=0.7)
                            ax.set_xlim(0, 1)
                            ax.set_xticks([0, 1])
                        else:
                            st.warning("No unperforated net pay data to plot.")

                elif track == 'shpor':
                    ax.set_title("SHPOR", fontsize=10)
                    if 'SHPOR' in display_df.columns:
                        shpor_values = display_df['SHPOR']
                        if not shpor_values.isna().all():
                            mask = ~shpor_values.isna()
                            segments = np.split(np.where(mask)[0], np.where(np.diff(np.where(mask)[0]) > 1)[0] + 1)
                            for seg in segments:
                                if len(seg) > 1:
                                    ax.plot(shpor_values.iloc[seg], display_df['DEPTH'].iloc[seg], 
                                           color=colors['shpor'], lw=1.5)
                            min_val = shpor_values.min() * 0.9 if shpor_values.min() > 0 else shpor_values.min() * 1.1 if shpor_values.min() < 0 else 0
                            max_val = shpor_values.max() * 1.1 if shpor_values.max() > 0 else shpor_values.max() * 0.9 if shpor_values.max() < 0 else 1
                            ax.set_xlim(min_val, max_val)
                            ax.legend([f'SHPOR'], fontsize=8)
                        else:
                            st.warning("No SHPOR data to plot.")

                elif track == 'pornet':
                    ax.set_title("PORNET", fontsize=10)
                    if 'PORNET' in display_df.columns:
                        pornet_values = display_df['PORNET']
                        if not pornet_values.isna().all():
                            mask = ~pornet_values.isna()
                            segments = np.split(np.where(mask)[0], np.where(np.diff(np.where(mask)[0]) > 1)[0] + 1)
                            for seg in segments:
                                if len(seg) > 1:
                                    ax.plot(pornet_values.iloc[seg], display_df['DEPTH'].iloc[seg], 
                                           color=colors['pornet'], lw=1.5)
                            min_val = pornet_values.min() * 0.9 if pornet_values.min() > 0 else pornet_values.min() * 1.1 if pornet_values.min() < 0 else 0
                            max_val = pornet_values.max() * 1.1 if pornet_values.max() > 0 else pornet_values.max() * 0.9 if pornet_values.max() < 0 else 1
                            ax.set_xlim(min_val, max_val)
                            ax.legend([f'PORNET'], fontsize=8)
                        else:
                            st.warning("No PORNET data to plot.")

            plt.tight_layout(pad=2.0, h_pad=1.0)
            st.pyplot(fig, use_container_width=True)

        # Summary and analysis tabs
        tab1, tab2 = st.tabs(["Summary Table", "Unperforated Net Pay"])

        with tab1:
            st.subheader("Well Log Summary")
            # Select important columns to display
            important_cols = ['DEPTH', 'PHIT', 'SW', 'VSH', 'RESFLAG', 'PAYFLAG', 'PERF', 'UNPERF_NET_PAY']
            available_cols = [col for col in important_cols if col in display_df.columns and not display_df[col].isna().all()]
            
            # Add mineral volumes if available
            mineral_cols = ['VSAND', 'VSILT', 'VGLAU', 'VILITE', 'VLIME', 'VOIL', 'VWATER']
            available_minerals = [col for col in mineral_cols if col in display_df.columns and not display_df[col].isna().all()]
            
            summary_cols = available_cols + available_minerals
            summary_df = display_df[summary_cols].round(3)
            st.dataframe(summary_df, use_container_width=True)
            
            # Show summary statistics using CPI flags
            st.subheader("Summary Statistics")
            
            # Calculate thicknesses using depth differences
            depth_diff = display_df['DEPTH'].diff().fillna(0)
            
            if 'RESFLAG' in display_df.columns:
                resflag_thickness = depth_diff[display_df['RESFLAG'] == 1].sum()
                st.write(f"Reservoir Flag Thickness: **{resflag_thickness:.2f} m**")
            
            if 'PAYFLAG' in display_df.columns:
                payflag_thickness = depth_diff[display_df['PAYFLAG'] == 1].sum()
                st.write(f"Pay Flag Thickness: **{payflag_thickness:.2f} m**")
            
            if 'UNPERF_NET_PAY' in display_df.columns:
                unperf_thickness = depth_diff[display_df['UNPERF_NET_PAY'] == 1].sum()
                st.write(f"Unperforated Net Pay Thickness: **{unperf_thickness:.2f} m**")
            
            # Show flag statistics
            if 'RESFLAG' in display_df.columns:
                resflag_counts = display_df['RESFLAG'].value_counts().to_dict()
                st.write(f"Reservoir Flag counts: {resflag_counts}")
            
            if 'PAYFLAG' in display_df.columns:
                payflag_counts = display_df['PAYFLAG'].value_counts().to_dict()
                st.write(f"Pay Flag counts: {payflag_counts}")

        with tab2:
            st.subheader("Unperforated Net Pay Intervals")
            
            # Create two tabs within this tab: one for selected well, one for all wells
            well_tab, all_wells_tab = st.tabs(["Selected Well", "All Wells"])
            
            with well_tab:
                st.write(f"**Unperforated Net Pay Intervals for {selected_well}**")
                
                # Get unperforated intervals for selected well
                if 'UNPERF_NET_PAY' in display_df.columns:
                    unperf_df = display_df[display_df['UNPERF_NET_PAY'] == 1].copy()
                else:
                    unperf_df = pd.DataFrame()
                
                if unperf_df.empty:
                    st.success(f"No unperforated net pay found in {selected_well}! 🎉")
                else:
                    # Sort by depth and create continuous intervals
                    unperf_df = unperf_df.sort_values('DEPTH')
                    
                    # Create groups for continuous intervals
                    unperf_df['GROUP'] = 0
                    group_num = 0
                    
                    if len(unperf_df) > 0:
                        unperf_df.iloc[0, unperf_df.columns.get_loc('GROUP')] = group_num
                        
                        for i in range(1, len(unperf_df)):
                            current_depth = unperf_df.iloc[i]['DEPTH']
                            prev_depth = unperf_df.iloc[i-1]['DEPTH']
                            
                            if current_depth - prev_depth > 0.6:
                                group_num += 1
                            
                            unperf_df.iloc[i, unperf_df.columns.get_loc('GROUP')] = group_num
                    
                    # Group by the GROUP column
                    grouped = unperf_df.groupby('GROUP').agg(
                        Top=('DEPTH', 'min'),
                        Base=('DEPTH', 'max'),
                        Avg_Porosity=('PHIT', 'mean') if 'PHIT' in unperf_df.columns else ('DEPTH', lambda x: np.nan),
                        Avg_Sw=('SW', 'mean') if 'SW' in unperf_df.columns else ('DEPTH', lambda x: np.nan),
                        Avg_VSH=('VSH', 'mean') if 'VSH' in unperf_df.columns else ('DEPTH', lambda x: np.nan),
                        Data_Points=('DEPTH', 'count')
                    ).reset_index(drop=True)
                    
                    # Calculate thickness
                    grouped['Thickness (m)'] = (grouped['Base'] - grouped['Top']).round(2)
                    
                    # Estimate thickness for single-point intervals
                    for idx, row in grouped.iterrows():
                        if row['Data_Points'] == 1:
                            grouped.at[idx, 'Thickness (m)'] = 0.1524
                            grouped.at[idx, 'Base'] = row['Top'] + 0.1524
                    
                    # Apply AIT cutoff if enabled
                    if st.session_state.apply_cutoffs:
                        grouped = grouped[grouped['Thickness (m)'] >= st.session_state.ait_cutoff]
                    
                    if grouped.empty:
                        st.info(f"No unperforated net pay intervals meet the AIT cutoff of {st.session_state.ait_cutoff} meters.")
                    else:
                        grouped['Well'] = selected_well
                        
                        grouped['Zone'] = 'Unknown'
                        if 'tops' in well:
                            tops = well['tops']
                            if not tops.empty and 'DEPTH' in tops.columns and 'TOP' in tops.columns:
                                tops = tops.sort_values('DEPTH')
                                for i, row in grouped.iterrows():
                                    mid_depth = (row['Top'] + row['Base']) / 2
                                    grouped.at[i, 'Zone'] = assign_formation(mid_depth, tops)
                        
                        display_cols = ['Well', 'Zone', 'Top', 'Base', 'Thickness (m)', 'Avg_Porosity', 'Avg_Sw', 'Avg_VSH']
                        display_df_result = grouped[display_cols].copy()
                        
                        # Convert to percentages for display
                        for col in ['Avg_Porosity', 'Avg_Sw', 'Avg_VSH']:
                            if col in display_df_result.columns:
                                display_df_result[col] = display_df_result[col].apply(lambda x: round(x * 100, 2) if pd.notna(x) else np.nan)
                        
                        st.dataframe(display_df_result, use_container_width=True)
                        
                        # Summary statistics
                        st.subheader("Summary Statistics")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            total_thickness = grouped['Thickness (m)'].sum()
                            st.metric("Total Unperf Thickness", f"{total_thickness:.2f} m")
                        with col2:
                            num_intervals = len(grouped)
                            st.metric("Number of Intervals", num_intervals)
                        with col3:
                            avg_thickness = grouped['Thickness (m)'].mean() if num_intervals > 0 else 0
                            st.metric("Average Interval Thickness", f"{avg_thickness:.2f} m")
                        
                        # Download button for selected well
                        csv = display_df_result.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            f"Download {selected_well} Unperforated Intervals",
                            csv,
                            f"{selected_well}_unperforated_net_pay.csv",
                            "text/csv",
                            key="download_selected_well_unperf"
                        )
            
            with all_wells_tab:
                st.write("**Unperforated Net Pay Intervals for ALL Wells**")
                
                # Get unperforated intervals for all wells
                all_intervals_df = get_all_wells_unperf_intervals()
                
                if all_intervals_df.empty:
                    st.info("No unperforated net pay intervals found across all wells.")
                else:
                    # Display the detailed intervals dataframe
                    st.subheader("Detailed Intervals by Well and Formation")
                    st.dataframe(all_intervals_df, use_container_width=True)
                    
                    # Summary statistics for all wells
                    st.subheader("Summary Statistics (All Wells)")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        total_thickness_all = all_intervals_df['Thickness (m)'].sum()
                        st.metric("Total Thickness (All Wells)", f"{total_thickness_all:.2f} m")
                    
                    with col2:
                        num_intervals_all = len(all_intervals_df)
                        st.metric("Total Intervals (All Wells)", num_intervals_all)
                    
                    with col3:
                        avg_thickness_all = all_intervals_df['Thickness (m)'].mean() if num_intervals_all > 0 else 0
                        st.metric("Avg Thickness (All Wells)", f"{avg_thickness_all:.2f} m")
                    
                    with col4:
                        num_wells = all_intervals_df['Well'].nunique()
                        st.metric("Wells with Unperf Pay", num_wells)
                    
                    # Create formation breakdown table similar to the image
                    st.subheader("Formation Breakdown by Well")
                    st.markdown("Thickness (m) of unperforated net pay by formation for each well")
                    
                    formation_breakdown = create_formation_breakdown(all_intervals_df)
                    
                    if not formation_breakdown.empty:
                        # Display the formation breakdown table
                        st.dataframe(formation_breakdown, use_container_width=True)
                        
                        # Create a heatmap-style visualization of the formation breakdown
                        st.subheader("Formation Thickness Heatmap")
                        
                        # Prepare data for heatmap (excluding 'Well' and 'Total' columns)
                        heatmap_data = formation_breakdown.set_index('Well')
                        total_col = heatmap_data.pop('Total') if 'Total' in heatmap_data.columns else None
                        
                        if not heatmap_data.empty and heatmap_data.shape[1] > 0:
                            fig, ax = plt.subplots(figsize=(12, max(4, len(heatmap_data) * 0.5)))
                            
                            # Create heatmap
                            im = ax.imshow(heatmap_data.values, cmap='YlOrRd', aspect='auto', vmin=0)
                            
                            # Set ticks and labels
                            ax.set_xticks(np.arange(len(heatmap_data.columns)))
                            ax.set_yticks(np.arange(len(heatmap_data.index)))
                            ax.set_xticklabels(heatmap_data.columns, rotation=45, ha='right', fontsize=9)
                            ax.set_yticklabels(heatmap_data.index, fontsize=9)
                            
                            # Add colorbar
                            plt.colorbar(im, ax=ax, label='Thickness (m)')
                            
                            # Add text annotations
                            for i in range(len(heatmap_data.index)):
                                for j in range(len(heatmap_data.columns)):
                                    value = heatmap_data.iloc[i, j]
                                    if value > 0:
                                        text_color = 'white' if value > heatmap_data.values.max() * 0.6 else 'black'
                                        ax.text(j, i, f'{value:.1f}', 
                                               ha='center', va='center', 
                                               color=text_color, fontsize=8, fontweight='bold')
                            
                            ax.set_title('Unperforated Net Pay Thickness by Well and Formation', fontsize=12, fontweight='bold')
                            plt.tight_layout()
                            st.pyplot(fig, use_container_width=True)
                        
                        # Add total thickness back for display
                        if total_col is not None:
                            formation_breakdown['Total'] = total_col
                    
                    # Simple bar chart of total thickness by well
                    st.subheader("Total Unperforated Net Pay by Well")
                    
                    # Group by well and sum thickness
                    well_totals = all_intervals_df.groupby('Well')['Thickness (m)'].sum().sort_values()
                    
                    # Create horizontal bar chart
                    fig2, ax2 = plt.subplots(figsize=(10, max(4, len(well_totals) * 0.5)))
                    y_pos = np.arange(len(well_totals))
                    bars = ax2.barh(y_pos, well_totals.values, color='#3498db', alpha=0.7)
                    ax2.set_yticks(y_pos)
                    ax2.set_yticklabels(well_totals.index)
                    ax2.set_xlabel('Total Thickness (m)')
                    ax2.set_title('Total Unperforated Net Pay by Well')
                    
                    # Add value labels
                    for i, (bar, val) in enumerate(zip(bars, well_totals.values)):
                        ax2.text(val + 0.1, i, f'{val:.1f}m', va='center')
                    
                    plt.tight_layout()
                    st.pyplot(fig2, use_container_width=True)
                    
                    # Formation summary (total by formation across all wells)
                    st.subheader("Total Thickness by Formation (All Wells)")
                    formation_totals = all_intervals_df.groupby('Zone')['Thickness (m)'].sum().sort_values(ascending=False)
                    
                    fig3, ax3 = plt.subplots(figsize=(10, max(4, len(formation_totals) * 0.5)))
                    bars3 = ax3.barh(np.arange(len(formation_totals)), formation_totals.values, color='#e67e22', alpha=0.7)
                    ax3.set_yticks(np.arange(len(formation_totals)))
                    ax3.set_yticklabels(formation_totals.index)
                    ax3.set_xlabel('Total Thickness (m)')
                    ax3.set_title('Total Unperforated Net Pay by Formation')
                    
                    # Add value labels
                    for i, (bar, val) in enumerate(zip(bars3, formation_totals.values)):
                        ax3.text(val + 0.1, i, f'{val:.1f}m', va='center')
                    
                    plt.tight_layout()
                    st.pyplot(fig3, use_container_width=True)
                    
                    # Download buttons
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        csv_all = all_intervals_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "Download Detailed Intervals",
                            csv_all,
                            "all_wells_unperforated_net_pay.csv",
                            "text/csv",
                            key="download_all_wells_unperf"
                        )
                    
                    with col2:
                        if not formation_breakdown.empty:
                            csv_formation = formation_breakdown.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "Download Formation Breakdown",
                                csv_formation,
                                "formation_breakdown.csv",
                                "text/csv",
                                key="download_formation_breakdown"
                            )
                    
                    with col3:
                        # Well summary
                        well_summary = all_intervals_df.groupby('Well').agg(
                            Intervals=('Thickness (m)', 'count'),
                            Total_Thickness=('Thickness (m)', 'sum'),
                            Avg_Porosity=('Avg_Porosity', 'mean'),
                            Avg_Sw=('Avg_Sw', 'mean')
                        ).round(2).reset_index()
                        
                        csv_summary = well_summary.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "Download Well Summary",
                            csv_summary,
                            "well_summary.csv",
                            "text/csv",
                            key="download_well_summary"
                        )

    # Customized Visualization Tab
    with custom_tab:
        st.info("Customized Visualization allows you to select specific curves for each track. Use the dropdowns below to configure your view.")
        
        # Get all available curves
        available_curves = list(df.columns)
        available_curves.insert(0, "None")
        
        # Track configuration
        track_configs = {
            'track1': {'label': 'Track 1', 'default': 'DEPTH'},
            'track2': {'label': 'Track 2', 'default': 'PHIT'},
            'track3': {'label': 'Track 3', 'default': 'SW'},
            'track4': {'label': 'Track 4', 'default': 'VSH'},
            'track5': {'label': 'Track 5', 'default': 'RESFLAG'},
            'track6': {'label': 'Track 6', 'default': 'PAYFLAG'},
            'track7': {'label': 'Track 7', 'default': 'PERF'},
            'track8': {'label': 'Track 8', 'default': 'UNPERF_NET_PAY'}
        }
        
        # Initialize session state
        for track in track_configs.keys():
            if f"custom_curve_{track}" not in st.session_state:
                default = track_configs[track]['default']
                st.session_state[f"custom_curve_{track}"] = default if default in available_curves else "None"
        
        # Create selection dropdowns
        st.subheader("Configure Tracks")
        cols = st.columns(len(track_configs))
        selected_curves = {}
        
        for idx, (track, config) in enumerate(track_configs.items()):
            with cols[idx]:
                selected_curve = st.selectbox(
                    config['label'],
                    options=available_curves,
                    index=available_curves.index(st.session_state[f"custom_curve_{track}"]) if st.session_state[f"custom_curve_{track}"] in available_curves else 0,
                    key=f"custom_select_{track}"
                )
                selected_curves[track] = selected_curve
                st.session_state[f"custom_curve_{track}"] = selected_curve
        
        # Filter out empty tracks
        active_tracks = {k: v for k, v in selected_curves.items() if v != "None"}
        
        if not active_tracks:
            st.warning("Please select at least one curve to display.")
        else:
            # Create the plot
            fig, axes = plt.subplots(
                figsize=(3 * len(active_tracks), 18),
                ncols=len(active_tracks),
                gridspec_kw={'wspace': 0.1, 'hspace': 0.1}
            )
            axes = np.atleast_1d(axes)
            
            for i, (track_name, curve_name) in enumerate(active_tracks.items()):
                ax = axes[i]
                ax.invert_yaxis()
                ax.grid(True, alpha=0.3)
                
                # Set y-axis limits and labels
                ax.set_ylim(depth_range[1], depth_range[0])
                ax.tick_params(axis='y', which='both', left=True, labelleft=True, labelsize=8)
                
                if i == 0:
                    ax.set_ylabel("Depth (m)", fontsize=10)
                else:
                    ax.set_ylabel("")
                
                ax.set_title(f"{curve_name}", fontsize=10)
                
                if curve_name in display_df.columns:
                    values = display_df[curve_name].copy()
                    
                    # Determine plot type based on curve name
                    if curve_name in ['RESFLAG', 'PAYFLAG', 'PERF', 'UNPERF_NET_PAY']:
                        # Binary or categorical track
                        values = values.dropna()
                        if not values.empty:
                            if curve_name == 'PERF':
                                ax.step(values, display_df.loc[values.index, 'DEPTH'], where='mid', 
                                       color=colors.get('perforation', '#2ca02c'), lw=1.5)
                                ax.set_xlim(-1.5, 1.5)
                                ax.set_xticks([-1, 0, 1])
                            else:
                                # Get color for the specific track
                                color_key = curve_name.lower().replace('_', '')
                                color = colors.get(color_key, '#1f77b4')
                                ax.fill_betweenx(display_df.loc[values.index, 'DEPTH'], 0, values.clip(0, 1),
                                                 step='pre', alpha=0.7,
                                                 color=color)
                                ax.set_xlim(0, 1)
                                ax.set_xticks([0, 1])
                    else:
                        # Continuous track
                        if not values.isna().all():
                            # Check if values are in percentage
                            if values.max() > 1.5 and curve_name not in ['DEPTH']:
                                values = values * 100
                            
                            # Get color for the curve
                            color_key = curve_name.lower().replace('_', '')
                            color = colors.get(color_key, '#1f77b4')
                            ax.plot(values, display_df['DEPTH'], lw=1.5, color=color)
                            
                            # Add cutoff lines if applicable
                            if curve_name == 'PHIT' and st.session_state.apply_cutoffs:
                                ax.axvline(st.session_state.phit_value, color='red', ls=':', lw=1)
                            elif curve_name == 'SW' and st.session_state.apply_cutoffs:
                                ax.axvline(st.session_state.sw_value, color='red', ls=':', lw=1)
                            elif curve_name == 'VSH' and st.session_state.apply_cutoffs:
                                ax.axvline(st.session_state.vsh_value, color='red', ls=':', lw=1)
                            
                            # Set appropriate x-limits
                            if curve_name == 'PHIT':
                                ax.set_xlim(0, 50)
                            elif curve_name in ['SW', 'VSH']:
                                ax.set_xlim(0, 100)
            
            plt.tight_layout(pad=2.0, h_pad=1.0)
            st.pyplot(fig, use_container_width=True)

else:
    st.info("Upload LAS files to begin visualization.")

# Footer
st.markdown('''
---
**Streamlit App** – Interactive well log, tops, and perforation visualization.  
Developed by Egypt Technical Team.
''', unsafe_allow_html=True)
