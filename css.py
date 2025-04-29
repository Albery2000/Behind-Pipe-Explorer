CSS ="""
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
"""