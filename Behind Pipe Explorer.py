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

# Function to calculate Net Reservoir and Net Pay
def calculate_net_curves(df, null_value=-999.0):
    """Calculate NET_RESERVOIR and NET_PAY based on available curves"""
    df = df.copy()
    
    # Create empty columns initially
    df['NET_RESERVOIR'] = 0
    df['NET_PAY'] = 0
    
    # Replace null values with NaN
    df = df.replace(null_value, np.nan)
    
    # --- Calculate NET RESERVOIR ---
    # Method 1: If VSH is available, use it with default cutoff (50%)
    if 'VSH' in df.columns:
        # Ensure VSH is in decimal (0-1) range
        vsh_series = df['VSH'].copy()
        # Replace any remaining null values
        vsh_series = vsh_series.replace(null_value, np.nan)
        
        # Convert to decimal if needed
        if vsh_series.max() > 1.5 and vsh_series.min() >= 0:  # If in percentage, convert to decimal
            vsh_series = vsh_series / 100
        
        # Apply cutoff for net reservoir (default: VSH <= 0.5)
        # Only apply where we have valid VSH data
        valid_vsh_mask = vsh_series.notna()
        df.loc[valid_vsh_mask, 'NET_RESERVOIR'] = (vsh_series[valid_vsh_mask] <= 0.5).astype(int)
    
    # Method 2: If mineral volumes are available, calculate VSH from them
    elif all(col in df.columns for col in ['VGLAU', 'VILITE', 'VSAND', 'VSILT']):
        # Calculate total shale/clay volume
        df['VSH_CALC'] = df[['VGLAU', 'VILITE']].sum(axis=1, skipna=True)
        if df['VSH_CALC'].max() > 1.5:  # If in percentage, convert to decimal
            df['VSH_CALC'] = df['VSH_CALC'] / 100
        df['NET_RESERVOIR'] = (df['VSH_CALC'] <= 0.5).astype(int)
    
    # --- Calculate NET PAY ---
    # Need PHIT/PHIE, SW, and NET_RESERVOIR
    conditions = []
    
    # Condition 1: NET_RESERVOIR must be 1
    conditions.append(df['NET_RESERVOIR'] == 1)
    
    # Condition 2: PHIT or PHIE >= cutoff (default: 0.1 or 10%)
    porosity_col = None
    if 'PHIT' in df.columns:
        porosity_col = 'PHIT'
    elif 'PHIE' in df.columns:
        porosity_col = 'PHIE'
    
    if porosity_col:
        phit_series = df[porosity_col].copy()
        # Replace null values
        phit_series = phit_series.replace(null_value, np.nan)
        
        # Convert to decimal if needed
        if phit_series.max() > 1.5 and phit_series.min() >= 0:  # If in percentage, convert to decimal
            phit_series = phit_series / 100
        
        conditions.append(phit_series >= 0.1)  # Default porosity cutoff 10%
    
    # Condition 3: SW <= cutoff (default: 0.5 or 50%)
    if 'SW' in df.columns:
        sw_series = df['SW'].copy()
        # Replace null values
        sw_series = sw_series.replace(null_value, np.nan)
        
        # Convert to decimal if needed
        if sw_series.max() > 1.5 and sw_series.min() >= 0:  # If in percentage, convert to decimal
            sw_series = sw_series / 100
        
        conditions.append(sw_series <= 0.5)  # Default Sw cutoff 50%
    
    # If we have at least 2 conditions (NET_RESERVOIR + one more), calculate NET_PAY
    if len(conditions) >= 2:
        # Combine conditions, but only where we have valid data
        valid_mask = pd.Series(True, index=df.index)
        for cond in conditions[1:]:  # Skip NET_RESERVOIR condition
            valid_mask = valid_mask & cond.notna()
        
        # Calculate NET_PAY only where we have valid data
        df['NET_PAY'] = 0
        if valid_mask.any():
            combined_conditions = pd.concat(conditions, axis=1)
            net_pay_mask = combined_conditions.all(axis=1) & valid_mask
            df.loc[net_pay_mask, 'NET_PAY'] = 1
    
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

            # Standardize column names
            mapping = {
                'PHIT_D': 'PHIT', 'PHIE_T': 'PHIT','PHI_T': 'PHIT',
                'PHI_TOTAL': 'PHIT', 'SW_AR': 'SW','PHIE_D': 'PHIE',
                'PHIE_T': 'PHIE','SW_AR': 'SW', 'SW_T': 'SW',
                'SWT': 'SW', 'VSH_GR': 'VSH',
                'VSHL': 'VSH',
                'VCL': 'VSH',  'NET_PAY': 'NET_PAY','NET_RES': 'NET_RES',
                'SWT_NET': 'SW_NET', 'VSH': 'VSH', 'NET_PAY': 'NET_PAY',
                'NET_RES': 'NET_RES', 'SH_POR': 'SHPOR', 'PORNET_D': 'PORNET',
                'RESFLAG': 'RESFLAG', 'PAYFLAG': 'PAYFLAG'
            }
            for orig, std in mapping.items():
                if orig in df.columns and std not in df.columns:
                    df[std] = df[orig]
            
            # Get NULL value from LAS header if available
            null_value = -999.25  # Default
            if hasattr(las, 'well') and 'NULL' in las.well:
                try:
                    null_value = float(las.well['NULL'].value)
                except:
                    pass
            
            # Replace NULL values with NaN for calculations
            df = df.replace(null_value, np.nan)
            
            # --- AUTO-DETECT & NORMALIZE PERCENT LOGS ---
            percent_logs = ['PHIT', 'PHIE', 'SW', 'VSH', 'VGLAU', 'VILITE', 'VLIME', 'VOIL', 'VSAND', 'VSILT', 'VWATER']
            for col in percent_logs:
                if col in df.columns:
                    # Check if values are likely in percentage (0-100) vs decimal (0-1)
                    non_null_vals = df[col].dropna()
                    if not non_null_vals.empty:
                        max_val = non_null_vals.max()
                        min_val = non_null_vals.min()
                        # If values are between 0-100 and typical for percentages
                        if max_val > 1.5 and min_val >= 0:
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
            
            # Calculate NET_RESERVOIR and NET_PAY if not already present
            if 'NET_RES' not in df.columns and 'NET_PAY' not in df.columns:
                df = calculate_net_curves(df, null_value)
            
            # Handle ResFlag and PayFlag - ensure they're integers (0 or 1)
            if 'RESFLAG' in df.columns:
                df['RESFLAG'] = df['RESFLAG'].fillna(0).astype(int)
            if 'PAYFLAG' in df.columns:
                df['PAYFLAG'] = df['PAYFLAG'].fillna(0).astype(int)
            
            # Store the null value for reference
            df.attrs['null_value'] = null_value
            
            st.session_state.well_data[well_name] = {
                'data': df,
                'las': las,
                'header': clean_text(str(las.header)),
                'available_curves': list(df.columns),
                'null_value': null_value
            }
            
            # Show available curves for debugging
            with st.sidebar.expander(f"Curves in {well_name}"):
                st.write(f"Available curves: {', '.join(df.columns)}")
                
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
            lambda x: 1 if x == 'Open' else -1 if x == 'Plugged' else 0
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
        null_value = well.get('null_value', -999.25)
        
        # Apply cutoffs if selected
        if st.session_state.apply_cutoffs:
            # Apply custom cutoffs
            conditions = []
            
            # NET RESERVOIR condition (VSH cutoff)
            if 'VSH' in df.columns:
                vsh_series = df['VSH'].copy()
                vsh_series = vsh_series.replace(null_value, np.nan)
                if vsh_series.max() > 1.5 and vsh_series.min() >= 0:
                    vsh_series = vsh_series / 100
                df['NET_RESERVOIR'] = (vsh_series <= st.session_state.vsh_value / 100).astype(int)
                conditions.append(df['NET_RESERVOIR'] == 1)
            elif 'NET_RESERVOIR' in df.columns:
                conditions.append(df['NET_RESERVOIR'] == 1)
            
            # Porosity condition
            porosity_col = None
            if 'PHIT' in df.columns:
                porosity_col = 'PHIT'
            elif 'PHIE' in df.columns:
                porosity_col = 'PHIE'
            
            if porosity_col:
                phit_series = df[porosity_col].copy()
                phit_series = phit_series.replace(null_value, np.nan)
                if phit_series.max() > 1.5 and phit_series.min() >= 0:
                    phit_series = phit_series / 100
                conditions.append(phit_series >= st.session_state.phit_value / 100)
            
            # Sw condition
            if 'SW' in df.columns:
                sw_series = df['SW'].copy()
                sw_series = sw_series.replace(null_value, np.nan)
                if sw_series.max() > 1.5 and sw_series.min() >= 0:
                    sw_series = sw_series / 100
                conditions.append(sw_series <= st.session_state.sw_value / 100)
            
            if len(conditions) >= 2:
                # Calculate NET_PAY only where we have valid data
                df['NET_PAY'] = 0
                if conditions:
                    # Combine conditions
                    combined = pd.concat(conditions, axis=1)
                    # Check for valid data (not NaN)
                    valid_mask = combined.notna().all(axis=1)
                    # Calculate NET_PAY where all conditions are met and data is valid
                    net_pay_mask = combined.all(axis=1) & valid_mask
                    df.loc[net_pay_mask, 'NET_PAY'] = 1
            else:
                df['NET_PAY'] = 0
        else:
            # Use pre-calculated values
            if 'NET_RESERVOIR' not in df.columns or 'NET_PAY' not in df.columns:
                df = calculate_net_curves(df, null_value)
            
            # Ensure NET_PAY exists
            if 'NET_PAY' not in df.columns:
                df['NET_PAY'] = 0
        
        # Process perforations
        df['PERF'] = 0
        if 'perforations' in well and st.session_state.show_perforations:
            for _, row in well['perforations'].iterrows():
                df.loc[(df['DEPTH'] >= row['TOP']) & (df['DEPTH'] <= row['BASE']), 'PERF'] = row['PERF_VALUE']
        
        # Calculate unperforated net pay
        df['UNPERF_NET_PAY'] = ((df['NET_PAY'] == 1) & (df['PERF'] == 0)).astype(int)
        
        # Find unperforated net pay intervals
        unperf_df = df[df['UNPERF_NET_PAY'] == 1].copy()
        if unperf_df.empty:
            continue
        
        # Group consecutive depths into intervals
        unperf_df['GROUP'] = (unperf_df['DEPTH'].diff() > 0.2).cumsum()
        grouped = unperf_df.groupby('GROUP').agg(
            Top=('DEPTH', 'min'),
            Base=('DEPTH', 'max'),
            Avg_Porosity=('PHIT', 'mean') if 'PHIT' in unperf_df.columns and not unperf_df['PHIT'].isna().all() else 
                         ('PHIE', 'mean') if 'PHIE' in unperf_df.columns and not unperf_df['PHIE'].isna().all() else 
                         ('DEPTH', lambda x: np.nan),
            Avg_Sw=('SW', 'mean') if 'SW' in unperf_df.columns and not unperf_df['SW'].isna().all() else 
                   ('DEPTH', lambda x: np.nan),
            Avg_VSH=('VSH', 'mean') if 'VSH' in unperf_df.columns and not unperf_df['VSH'].isna().all() else 
                    ('DEPTH', lambda x: np.nan)
        ).reset_index(drop=True)
        
        grouped['Thickness (m)'] = (grouped['Base'] - grouped['Top']).round(2)
        
        # Apply AIT cutoff if enabled
        if st.session_state.apply_cutoffs:
            grouped = grouped[grouped['Thickness (m)'] >= st.session_state.ait_cutoff]
        
        if grouped.empty:
            continue
        
        grouped['Well'] = well_name
        
        # Assign zone based on tops
        grouped['Zone'] = 'Unknown'
        if 'tops' in well:
            tops = well['tops'].sort_values('DEPTH')
            for i, row in grouped.iterrows():
                valid_tops = tops[tops['DEPTH'] <= row['Top']]
                if not valid_tops.empty:
                    grouped.at[i, 'Zone'] = clean_text(valid_tops.iloc[-1]['TOP'])
        
        # Select and format columns
        grouped = grouped[['Well', 'Zone', 'Top', 'Base', 'Thickness (m)', 'Avg_Porosity', 'Avg_Sw', 'Avg_VSH']]
        
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
    null_value = well.get('null_value', -999.25)
    
    # Ensure NET_RESERVOIR and NET_PAY are calculated
    if 'NET_RESERVOIR' not in df.columns or 'NET_PAY' not in df.columns:
        df = calculate_net_curves(df, null_value)
        st.session_state.well_data[selected_well]['data'] = df

    # Apply cutoffs if selected
    if st.session_state.apply_cutoffs:
        df = df.copy()
        conditions = []
        
        # NET RESERVOIR condition
        if 'VSH' in df.columns:
            vsh_series = df['VSH'].copy()
            vsh_series = vsh_series.replace(null_value, np.nan)
            if vsh_series.max() > 1.5 and vsh_series.min() >= 0:
                vsh_series = vsh_series / 100
            df['NET_RESERVOIR'] = (vsh_series <= st.session_state.vsh_value / 100).astype(int)
            conditions.append(df['NET_RESERVOIR'] == 1)
        
        # Porosity condition
        porosity_col = None
        if 'PHIT' in df.columns:
            porosity_col = 'PHIT'
        elif 'PHIE' in df.columns:
            porosity_col = 'PHIE'
        
        if porosity_col:
            phit_series = df[porosity_col].copy()
            phit_series = phit_series.replace(null_value, np.nan)
            if phit_series.max() > 1.5 and phit_series.min() >= 0:
                phit_series = phit_series / 100
            conditions.append(phit_series >= st.session_state.phit_value / 100)
        
        # Sw condition
        if 'SW' in df.columns:
            sw_series = df['SW'].copy()
            sw_series = sw_series.replace(null_value, np.nan)
            if sw_series.max() > 1.5 and sw_series.min() >= 0:
                sw_series = sw_series / 100
            conditions.append(sw_series <= st.session_state.sw_value / 100)
        
        if len(conditions) >= 2:
            # Calculate NET_PAY only where we have valid data
            df['NET_PAY'] = 0
            if conditions:
                # Combine conditions
                combined = pd.concat(conditions, axis=1)
                # Check for valid data (not NaN)
                valid_mask = combined.notna().all(axis=1)
                # Calculate NET_PAY where all conditions are met and data is valid
                net_pay_mask = combined.all(axis=1) & valid_mask
                df.loc[net_pay_mask, 'NET_PAY'] = 1
        else:
            df['NET_PAY'] = df.get('NET_PAY', 0)
    
    # Process perforations
    df['PERF'] = 0
    if 'perforations' in well and st.session_state.show_perforations:
        for _, row in well['perforations'].iterrows():
            df.loc[(df['DEPTH'] >= row['TOP']) & (df['DEPTH'] <= row['BASE']), 'PERF'] = row['PERF_VALUE']
    
    # Calculate unperforated net pay
    df['UNPERF_NET_PAY'] = ((df['NET_PAY'] == 1) & (df['PERF'] == 0)).astype(int)

    # Depth filter
    min_d, max_d = float(df['DEPTH'].min()), float(df['DEPTH'].max())
    depth_range = st.slider("Depth Range (m)", min_d, max_d, (min_d, max_d), 0.1, key="depth_slider")
    df = df[(df['DEPTH'] >= depth_range[0]) & (df['DEPTH'] <= depth_range[1])].copy()

    st.markdown(f'<h3 style="color: #1A3C6D;">{clean_text(selected_well)}</h3>', unsafe_allow_html=True)
    
    # Show available curves in sidebar for reference
    with st.sidebar.expander(f"Available curves in {selected_well}"):
        st.write(", ".join(df.columns.tolist()))
        st.write(f"NULL value: {null_value}")
        st.write(f"Data range: {min_d:.2f} - {max_d:.2f} m")

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
            available_tracks.append('tops')

        # Check for porosity logs
        porosity_logs = ['PHIT', 'PHIE']
        if any(col in df.columns for col in porosity_logs) and st.session_state.show_porosity:
            # Check if there's actual data (not all null)
            for col in porosity_logs:
                if col in df.columns:
                    valid_data = df[col].replace(null_value, np.nan).dropna()
                    if not valid_data.empty:
                        available_tracks.append('phit')
                        break

        # Check for saturation logs
        if 'SW' in df.columns and st.session_state.show_saturation:
            valid_data = df['SW'].replace(null_value, np.nan).dropna()
            if not valid_data.empty:
                available_tracks.append('sw')

        # Check for VSH
        if 'VSH' in df.columns:
            valid_data = df['VSH'].replace(null_value, np.nan).dropna()
            if not valid_data.empty:
                available_tracks.append('vsh')

        # Check for net reservoir
        if 'NET_RESERVOIR' in df.columns and not df['NET_RESERVOIR'].isna().all() and st.session_state.show_net_reservoir:
            available_tracks.append('net_reservoir')

        # Check for net pay
        if 'NET_PAY' in df.columns and not df['NET_PAY'].isna().all() and st.session_state.show_net_pay:
            available_tracks.append('net_pay')

        # Check for ResFlag
        if 'RESFLAG' in df.columns and not df['RESFLAG'].isna().all() and st.session_state.show_resflag:
            available_tracks.append('resflag')

        # Check for PayFlag
        if 'PAYFLAG' in df.columns and not df['PAYFLAG'].isna().all() and st.session_state.show_payflag:
            available_tracks.append('payflag')

        # Check for special logs
        if 'SHPOR' in df.columns and st.session_state.show_porosity:
            valid_data = df['SHPOR'].replace(null_value, np.nan).dropna()
            if not valid_data.empty:
                available_tracks.append('shpor')
        if 'PORNET' in df.columns and st.session_state.show_porosity:
            valid_data = df['PORNET'].replace(null_value, np.nan).dropna()
            if not valid_data.empty:
                available_tracks.append('pornet')

        # Check for perforations
        if 'PERF' in df.columns and st.session_state.show_perforations:
            available_tracks.append('perf')

        # Check for unperforated net pay
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
                figsize=(3 * len(selected_tracks), 18),
                ncols=len(selected_tracks),
                gridspec_kw={'wspace': 0.1, 'hspace': 0.1}
            )
            axes = np.atleast_1d(axes)

            for i, track in enumerate(selected_tracks):
                ax = axes[i]
                ax.invert_yaxis()
                ax.grid(True, alpha=0.3)
                
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
    
                    # Plot PHIT if available and has data
                    if 'PHIT' in df.columns:
                        phit_values = df['PHIT'].copy()
                        phit_values = phit_values.replace(null_value, np.nan)
                        valid_mask = phit_values.notna()
                        
                        if valid_mask.any():
                            # Check if values need normalization
                            if phit_values.max() > 1.5 and phit_values.min() >= 0:
                                phit_values = phit_values * 100
                            
                            # Plot only valid data
                            ax.plot(phit_values[valid_mask], df.loc[valid_mask, 'DEPTH'], 
                                   color=colors['porosity'], label='PHIT', lw=1.5)
    
                    # Plot PHIE if available and has data
                    if 'PHIE' in df.columns:
                        phie_values = df['PHIE'].copy()
                        phie_values = phie_values.replace(null_value, np.nan)
                        valid_mask = phie_values.notna()
                        
                        if valid_mask.any():
                            # Check if values need normalization
                            if phie_values.max() > 1.5 and phie_values.min() >= 0:
                                phie_values = phie_values * 100
                            
                            ax.plot(phie_values[valid_mask], df.loc[valid_mask, 'DEPTH'], 
                                   color=colors['porosity'], ls='--', label='PHIE', lw=1)
    
                    # Add cutoff line if applicable
                    if st.session_state.apply_cutoffs:
                        ax.axvline(st.session_state.phit_value, color='red', ls=':', lw=1, label=f'Cutoff: {st.session_state.phit_value}%')

                    ax.set_xlim(0, 50)  # Porosity typically 0-50%
                    if 'PHIT' in df.columns or 'PHIE' in df.columns:
                        ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)

                elif track == 'sw':
                    ax.set_title("Water Saturation (%)", fontsize=10)
                    if 'SW' in df.columns:
                        sw_values = df['SW'].copy()
                        sw_values = sw_values.replace(null_value, np.nan)
                        valid_mask = sw_values.notna()
                        
                        if valid_mask.any():
                            # Normalize to percentage for display
                            if sw_values.max() > 1.5 and sw_values.min() >= 0:
                                sw_values = sw_values * 100
        
                            # Plot the curve
                            ax.plot(sw_values[valid_mask], df.loc[valid_mask, 'DEPTH'], 
                                   color=colors['saturation'], label='Sw', lw=1.5)
        
                            # Add cutoff line if applicable
                            if st.session_state.apply_cutoffs:
                                ax.axvline(st.session_state.sw_value, color='red', ls=':', lw=1, label=f'Cutoff: {st.session_state.sw_value}%')
        
                            # Set limits and legend
                            ax.set_xlim(0, 100)
                            ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)

                elif track == 'vsh':
                    ax.set_title("Clay Volume - VSH (%)", fontsize=10)
                    if 'VSH' in df.columns:
                        vsh_values = df['VSH'].copy()
                        vsh_values = vsh_values.replace(null_value, np.nan)
                        valid_mask = vsh_values.notna()
                        
                        if valid_mask.any():
                            # Normalize to percentage for display
                            if vsh_values.max() > 1.5 and vsh_values.min() >= 0:
                                vsh_values = vsh_values * 100
                            
                            # Plot the curve
                            ax.plot(vsh_values[valid_mask], df.loc[valid_mask, 'DEPTH'], 
                                   color='#8c564b', label='VSH', lw=1.5)
        
                            # Add cutoff line if applicable
                            if st.session_state.apply_cutoffs:
                                ax.axvline(st.session_state.vsh_value, color='red', ls=':', lw=1, label=f'Cutoff: {st.session_state.vsh_value}%')
        
                            # Set limits and legend
                            ax.set_xlim(0, 100)
                            ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)

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

                elif track == 'resflag':
                    ax.set_title("Reservoir Flag", fontsize=10)
                    values = df['RESFLAG'].dropna()
                    if not values.empty:
                        # Ensure values are 0 or 1
                        values = values.clip(0, 1).astype(int)
                        ax.fill_betweenx(df.loc[values.index, 'DEPTH'], 0, values,
                                         color=colors['resflag'], step='pre', alpha=0.7)
                        ax.set_xlim(0, 1)
                        ax.set_xticks([0, 1])
                        # Add labels
                        ax.text(0.5, depth_range[0] + 5, 'Reservoir Zone', 
                                ha='center', va='bottom', fontsize=7)
                    else:
                        st.warning("No valid data for Reservoir Flag to plot.")

                elif track == 'payflag':
                    ax.set_title("Pay Flag", fontsize=10)
                    values = df['PAYFLAG'].dropna()
                    if not values.empty:
                        # Ensure values are 0 or 1
                        values = values.clip(0, 1).astype(int)
                        ax.fill_betweenx(df.loc[values.index, 'DEPTH'], 0, values,
                                         color=colors['payflag'], step='pre', alpha=0.7)
                        ax.set_xlim(0, 1)
                        ax.set_xticks([0, 1])
                        # Add labels
                        ax.text(0.5, depth_range[0] + 5, 'Pay Zone', 
                                ha='center', va='bottom', fontsize=7)
                    else:
                        st.warning("No valid data for Pay Flag to plot.")

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
                        shpor_values = df['SHPOR'].copy()
                        shpor_values = shpor_values.replace(null_value, np.nan)
                        valid_mask = shpor_values.notna()
                        
                        if valid_mask.any():
                            ax.plot(shpor_values[valid_mask], df.loc[valid_mask, 'DEPTH'], 
                                   color=colors['shpor'], lw=1.5)
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
                        pornet_values = df['PORNET'].copy()
                        pornet_values = pornet_values.replace(null_value, np.nan)
                        valid_mask = pornet_values.notna()
                        
                        if valid_mask.any():
                            ax.plot(pornet_values[valid_mask], df.loc[valid_mask, 'DEPTH'], 
                                   color=colors['pornet'], lw=1.5)
                            min_val = pornet_values.min() * 0.9 if pornet_values.min() > 0 else pornet_values.min() * 1.1 if pornet_values.min() < 0 else 0
                            max_val = pornet_values.max() * 1.1 if pornet_values.max() > 0 else pornet_values.max() * 0.9 if pornet_values.max() < 0 else 1
                            ax.set_xlim(min_val, max_val)
                            ax.legend([f'PORNET'], fontsize=8)
                        else:
                            st.warning("No valid data for PORNET to plot.")
                    else:
                        st.warning("PORNET column not found in data.")

            plt.tight_layout(pad=2.0, h_pad=1.0)
            st.pyplot(fig, use_container_width=True)

        # Summary and analysis tabs
        tab1, tab2, tab3 = st.tabs(["Summary Table", "Unperforated Net Pay", "Well Statistics"])

        with tab1:
            st.subheader("Well Log Summary")
            # Select important columns to display
            important_cols = ['DEPTH', 'PHIT', 'PHIE', 'SW', 'VSH', 'NET_RESERVOIR', 'NET_PAY', 
                             'RESFLAG', 'PAYFLAG', 'PERF', 'UNPERF_NET_PAY']
            available_cols = [col for col in important_cols if col in df.columns]
            
            # Add mineral volumes if available
            mineral_cols = ['VSAND', 'VSILT', 'VGLAU', 'VILITE', 'VLIME', 'VOIL', 'VWATER']
            available_minerals = [col for col in mineral_cols if col in df.columns]
            
            summary_cols = available_cols + available_minerals
            
            # Create a copy for display, replacing null values with NaN
            display_df = df[summary_cols].copy()
            for col in display_df.columns:
                if col != 'DEPTH':
                    display_df[col] = display_df[col].replace(null_value, np.nan)
            
            st.dataframe(display_df.round(3), use_container_width=True)
            
        with tab2:
            st.subheader("Unperforated Net Pay Intervals")
            
            # Find unperforated net pay intervals
            unperf_df = df[df['UNPERF_NET_PAY'] == 1].copy() if 'UNPERF_NET_PAY' in df.columns else pd.DataFrame()
            
            if unperf_df.empty:
                st.success("No unperforated net pay intervals found! All net pay zones are either perforated or there is no net pay.")
            else:
                # Group consecutive depths into intervals
                unperf_df['GROUP'] = (unperf_df['DEPTH'].diff() > 0.2).cumsum()
                
                grouped = unperf_df.groupby('GROUP').agg(
                    Top=('DEPTH', 'min'),
                    Base=('DEPTH', 'max'),
                    Avg_Porosity=('PHIT', 'mean') if 'PHIT' in unperf_df.columns and not unperf_df['PHIT'].isna().all() else 
                                 ('PHIE', 'mean') if 'PHIE' in unperf_df.columns and not unperf_df['PHIE'].isna().all() else 
                                 ('DEPTH', lambda x: np.nan),
                    Avg_Sw=('SW', 'mean') if 'SW' in unperf_df.columns and not unperf_df['SW'].isna().all() else 
                           ('DEPTH', lambda x: np.nan),
                    Avg_VSH=('VSH', 'mean') if 'VSH' in unperf_df.columns and not unperf_df['VSH'].isna().all() else 
                            ('DEPTH', lambda x: np.nan),
                    Avg_RESFLAG=('RESFLAG', 'mean') if 'RESFLAG' in unperf_df.columns else ('DEPTH', lambda x: np.nan),
                    Avg_PAYFLAG=('PAYFLAG', 'mean') if 'PAYFLAG' in unperf_df.columns else ('DEPTH', lambda x: np.nan)
                ).reset_index(drop=True)
                
                grouped['Thickness (m)'] = (grouped['Base'] - grouped['Top']).round(2)
                
                # Apply AIT cutoff if enabled
                if st.session_state.apply_cutoffs:
                    grouped = grouped[grouped['Thickness (m)'] >= st.session_state.ait_cutoff]
                
                if grouped.empty:
                    st.info(f"No unperforated net pay intervals meet the AIT cutoff of {st.session_state.ait_cutoff} meters.")
                else:
                    grouped['Well'] = selected_well
                    
                    # Assign zone based on tops
                    grouped['Zone'] = 'Unknown'
                    if 'tops' in well:
                        tops = well['tops'].sort_values('DEPTH')
                        for i, row in grouped.iterrows():
                            valid_tops = tops[tops['DEPTH'] <= row['Top']]
                            if not valid_tops.empty:
                                grouped.at[i, 'Zone'] = clean_text(valid_tops.iloc[-1]['TOP'])
                    
                    # Select and format columns
                    result_cols = ['Well', 'Zone', 'Top', 'Base', 'Thickness (m)']
                    if 'Avg_Porosity' in grouped.columns:
                        result_cols.append('Avg_Porosity')
                    if 'Avg_Sw' in grouped.columns:
                        result_cols.append('Avg_Sw')
                    if 'Avg_VSH' in grouped.columns:
                        result_cols.append('Avg_VSH')
                    
                    grouped = grouped[result_cols]
                    
                    # Convert to percentages for display
                    for col in ['Avg_Porosity', 'Avg_Sw', 'Avg_VSH']:
                        if col in grouped.columns:
                            grouped[col] = grouped[col].apply(lambda x: round(x * 100, 2) if pd.notna(x) else np.nan)
                    
                    st.dataframe(grouped, use_container_width=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        csv = grouped.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "Download Unperforated Intervals",
                            csv,
                            f"{selected_well}_unperforated_net_pay.csv",
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
                    
                    # Display summary statistics
                    st.subheader("Unperforated Net Pay Summary")
                    total_unperf_thickness = grouped['Thickness (m)'].sum()
                    num_intervals = len(grouped)
                    avg_interval_thickness = grouped['Thickness (m)'].mean()
                    
                    st.write(f"**Total Unperforated Net Pay Thickness:** {total_unperf_thickness:.2f} m")
                    st.write(f"**Number of Intervals:** {num_intervals}")
                    st.write(f"**Average Interval Thickness:** {avg_interval_thickness:.2f} m")
                    
                    # Display histogram of interval thicknesses
                    if num_intervals > 1:
                        fig2, ax2 = plt.subplots(figsize=(8, 4))
                        ax2.hist(grouped['Thickness (m)'], bins=10, edgecolor='black', alpha=0.7)
                        ax2.set_xlabel('Thickness (m)')
                        ax2.set_ylabel('Frequency')
                        ax2.set_title('Distribution of Unperforated Net Pay Interval Thicknesses')
                        ax2.grid(True, alpha=0.3)
                        st.pyplot(fig2)
        
        with tab3:
            st.subheader("Well Statistics")
            
            # Calculate statistics
            stats_data = []
            
            # Total thickness statistics
            total_depth_range = df['DEPTH'].max() - df['DEPTH'].min()
            stats_data.append(("Total Depth Range", f"{total_depth_range:.2f} m"))
            
            # Net Reservoir statistics
            if 'NET_RESERVOIR' in df.columns:
                net_res_thickness = (df['NET_RESERVOIR'] == 1).sum() * 0.5  # Assuming 0.5m sampling
                net_res_percent = (net_res_thickness / total_depth_range * 100) if total_depth_range > 0 else 0
                stats_data.append(("Net Reservoir Thickness", f"{net_res_thickness:.2f} m ({net_res_percent:.1f}%)"))
            
            # Net Pay statistics
            if 'NET_PAY' in df.columns:
                net_pay_thickness = (df['NET_PAY'] == 1).sum() * 0.5  # Assuming 0.5m sampling
                net_pay_percent = (net_pay_thickness / total_depth_range * 100) if total_depth_range > 0 else 0
                stats_data.append(("Net Pay Thickness", f"{net_pay_thickness:.2f} m ({net_pay_percent:.1f}%)"))
            
            # Unperforated Net Pay statistics
            if 'UNPERF_NET_PAY' in df.columns:
                unperf_thickness = (df['UNPERF_NET_PAY'] == 1).sum() * 0.5  # Assuming 0.5m sampling
                if net_pay_thickness > 0:
                    unperf_percent = (unperf_thickness / net_pay_thickness * 100)
                else:
                    unperf_percent = 0
                stats_data.append(("Unperforated Net Pay Thickness", f"{unperf_thickness:.2f} m ({unperf_percent:.1f}% of net pay)"))
            
            # Perforation statistics
            if 'PERF' in df.columns:
                open_perf_thickness = (df['PERF'] == 1).sum() * 0.5
                plugged_perf_thickness = (df['PERF'] == -1).sum() * 0.5
                stats_data.append(("Open Perforation Thickness", f"{open_perf_thickness:.2f} m"))
                stats_data.append(("Plugged Perforation Thickness", f"{plugged_perf_thickness:.2f} m"))
            
            # ResFlag statistics
            if 'RESFLAG' in df.columns:
                resflag_thickness = (df['RESFLAG'] == 1).sum() * 0.5
                resflag_percent = (resflag_thickness / total_depth_range * 100) if total_depth_range > 0 else 0
                stats_data.append(("Reservoir Flag Thickness", f"{resflag_thickness:.2f} m ({resflag_percent:.1f}%)"))
            
            # PayFlag statistics
            if 'PAYFLAG' in df.columns:
                payflag_thickness = (df['PAYFLAG'] == 1).sum() * 0.5
                payflag_percent = (payflag_thickness / total_depth_range * 100) if total_depth_range > 0 else 0
                stats_data.append(("Pay Flag Thickness", f"{payflag_thickness:.2f} m ({payflag_percent:.1f}%)"))
            
            # Display statistics in a table
            stats_df = pd.DataFrame(stats_data, columns=["Metric", "Value"])
            st.table(stats_df)
            
            # Data quality information
            st.subheader("Data Quality")
            quality_data = []
            
            # Check for null/missing data
            for col in ['PHIT', 'PHIE', 'SW', 'VSH']:
                if col in df.columns:
                    total_points = len(df[col])
                    null_points = (df[col] == null_value).sum() if null_value in df[col].values else df[col].isna().sum()
                    valid_points = total_points - null_points
                    valid_percent = (valid_points / total_points * 100) if total_points > 0 else 0
                    quality_data.append((col, f"{valid_points}/{total_points} ({valid_percent:.1f}%)"))
            
            if quality_data:
                quality_df = pd.DataFrame(quality_data, columns=["Curve", "Valid Data Points"])
                st.table(quality_df)

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
            'track5': {'label': 'Track 5', 'default': 'NET_RESERVOIR'},
            'track6': {'label': 'Track 6', 'default': 'NET_PAY'},
            'track7': {'label': 'Track 7', 'default': 'RESFLAG'},
            'track8': {'label': 'Track 8', 'default': 'PAYFLAG'},
            'track9': {'label': 'Track 9', 'default': 'UNPERF_NET_PAY'},
            'track10': {'label': 'Track 10', 'default': 'PERF'}
        }
        
        # Initialize session state
        for track in track_configs.keys():
            if f"custom_curve_{track}" not in st.session_state:
                default = track_configs[track]['default']
                st.session_state[f"custom_curve_{track}"] = default if default in available_curves else "None"
        
        # Create selection dropdowns
        st.subheader("Configure Tracks")
        cols = st.columns(5)
        selected_curves = {}
        
        for idx, (track, config) in enumerate(track_configs.items()):
            with cols[idx % 5]:
                selected_curve = st.selectbox(
                    config['label'],
                    options=available_curves,
                    index=available_curves.index(st.session_state[f"custom_curve_{track}"]),
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
                
                if curve_name in df.columns:
                    values = df[curve_name].copy()
                    
                    # Handle null values
                    values = values.replace(null_value, np.nan)
                    valid_mask = values.notna()
                    
                    if not valid_mask.any():
                        ax.text(0.5, 0.5, "No Data", transform=ax.transAxes, 
                               ha='center', va='center', fontsize=10, color='red')
                        continue
                    
                    # Determine plot type based on curve name
                    if curve_name in ['NET_RESERVOIR', 'NET_PAY', 'PERF', 'UNPERF_NET_PAY', 'RESFLAG', 'PAYFLAG']:
                        # Binary track
                        if curve_name == 'PERF':
                            ax.step(values[valid_mask], df.loc[valid_mask, 'DEPTH'], where='mid', 
                                   color=colors.get('perforation', '#2ca02c'), lw=1.5)
                            ax.set_xlim(-1.5, 1.5)
                            ax.set_xticks([-1, 0, 1])
                        else:
                            # Get color for the specific track
                            color_key = curve_name.lower().replace('_', '')
                            color = colors.get(color_key, '#1f77b4')
                            ax.fill_betweenx(df.loc[valid_mask, 'DEPTH'], 0, values[valid_mask],
                                             step='pre', alpha=0.7,
                                             color=color)
                            ax.set_xlim(0, 1)
                            ax.set_xticks([0, 1])
                    else:
                        # Continuous track
                        # Check if values are in percentage
                        if values.max() > 1.5 and curve_name not in ['DEPTH']:
                            values = values * 100
                        
                        # Get color for the curve
                        color_key = curve_name.lower().replace('_', '')
                        color = colors.get(color_key, '#1f77b4')
                        ax.plot(values[valid_mask], df.loc[valid_mask, 'DEPTH'], lw=1.5, color=color)
                        
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
**Streamlit App** – Interactive well log, tops, and perforation visualization with unperforated net pay detection.  
Developed by Egypt Technical Team.
''', unsafe_allow_html=True)
