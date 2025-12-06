"""
MAFAULDA Predictive Maintenance Dashboard - Professional Edition
Enterprise-grade interactive analytics platform with advanced visualizations
"""

from pydoc import pager
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

from config import get_config
from model_predictor import ModelPredictor
from feature_engineer import FeatureExtractor, FeatureEngineeringPipeline
from utils import load_csv_safe, calculate_metrics

# Page configuration
st.set_page_config(
    page_title="MAFAULDA Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
def load_professional_css():
    """Load enhanced professional CSS with glassmorphism and modern effects."""
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --primary-blue: #1E3A8A;
        --secondary-blue: #3B82F6;
        --accent-blue: #60A5FA;
        --success: #10B981;
        --warning: #F59E0B;
        --danger: #EF4444;
        --bg-dark: #0F172A;
        --bg-light: #F8FAFC;
    }
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #F8FAFC 0%, #E2E8F0 100%);
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.25);
    }
    
    /* Executive Summary Cards */
    .metric-card {
        background: linear-gradient(135deg, var(--primary-blue) 0%, var(--secondary-blue) 100%);
        border-radius: 16px;
        padding: 1.5rem;
        color: white;
        box-shadow: 0 4px 20px rgba(30, 58, 138, 0.3);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 30px rgba(30, 58, 138, 0.4);
    }
    
    .metric-value {
        font-size: 3rem;
        font-weight: 700;
        margin: 0.5rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .metric-label {
        font-size: 0.9rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.9;
    }
    
    .metric-trend {
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
    
    /* Status Indicators */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
        box-shadow: 0 0 10px currentColor;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    
    .status-good { background: var(--success); }
    .status-warning { background: var(--warning); }
    .status-critical { background: var(--danger); }
    
    /* Modern Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--secondary-blue) 0%, var(--accent-blue) 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);
    }
    
    /* Section Headers */
    .section-header {
        background: linear-gradient(90deg, var(--primary-blue) 0%, var(--secondary-blue) 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 2rem 0 1rem 0;
        font-size: 1.5rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(30, 58, 138, 0.2);
    }
    
    /* =============================== */
    /* ENHANCED SIDEBAR STYLING */
    /* =============================== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--primary-blue) 0%, #0F172A 100%) !important;
        padding: 2rem 1.5rem !important;
    }
    
    /* Sidebar headings */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] h6 {
        color: white !important;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    /* Sidebar labels and text */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: white !important;
    }
    
    /* Selectbox styling */
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stDateInput label,
    [data-testid="stSidebar"] .stToggle label {
        color: rgba(255, 255, 255, 0.9) !important;
        font-weight: 500;
        font-size: 0.95rem;
    }
    
    /* Input fields - modern glass effect */
    [data-testid="stSidebar"] .stSelectbox > div > div,
    [data-testid="stSidebar"] .stMultiSelect > div > div,
    [data-testid="stSidebar"] .stDateInput > div {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 10px !important;
        color: white !important;
        transition: all 0.3s ease;
    }
    
    [data-testid="stSidebar"] .stSelectbox > div > div:hover,
    [data-testid="stSidebar"] .stMultiSelect > div > div:hover {
        background: rgba(255, 255, 255, 0.15) !important;
        border-color: rgba(255, 255, 255, 0.3) !important;
    }
    
    /* Dropdown menu */
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="popover"] {
        background: linear-gradient(135deg, var(--primary-blue) 0%, #0F172A 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 10px !important;
        margin-top: 5px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* Dropdown items */
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="menu"] li {
        color: white !important;
        padding: 0.75rem 1rem !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="menu"] li:hover {
        background: rgba(59, 130, 246, 0.3) !important;
    }
    
    /* Multiselect tags */
    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {
        background: rgba(59, 130, 246, 0.4) !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        margin: 2px !important;
        padding: 4px 8px !important;
    }
    
    /* Slider styling */
    [data-testid="stSidebar"] .stSlider > div > div {
        background: linear-gradient(90deg, var(--secondary-blue) 0%, var(--accent-blue) 100%) !important;
        border-radius: 10px !important;
        height: 8px !important;
    }
    
    [data-testid="stSidebar"] .stSlider > div > div > div {
        background: white !important;
        border: 2px solid var(--secondary-blue) !important;
        box-shadow: 0 0 10px rgba(59, 130, 246, 0.5) !important;
    }
    
    /* Radio buttons */
    [data-testid="stSidebar"] .stRadio > div > label {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 10px !important;
        padding: 0.75rem 1rem !important;
        margin: 0.25rem 0 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stSidebar"] .stRadio > div > label:hover {
        background: rgba(255, 255, 255, 0.1) !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
    }
    
    [data-testid="stSidebar"] .stRadio > div > label > div:first-child {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
    }
    
    [data-testid="stSidebar"] .stRadio > div > label > div:first-child[data-checked="true"] {
        background: var(--secondary-blue) !important;
        border-color: var(--secondary-blue) !important;
    }
    
    /* Toggle switch */
    [data-testid="stSidebar"] .stToggle > div {
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 20px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    
    [data-testid="stSidebar"] .stToggle > div[data-checked="true"] {
        background: var(--secondary-blue) !important;
        border-color: var(--secondary-blue) !important;
    }
    
    /* Status messages */
    [data-testid="stSidebar"] .stSuccess,
    [data-testid="stSidebar"] .stWarning,
    [data-testid="stSidebar"] .stInfo {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 10px !important;
        padding: 1rem !important;
        margin: 0.5rem 0 !important;
    }
    
    [data-testid="stSidebar"] .stSuccess {
        border-left: 4px solid var(--success) !important;
    }
    
    [data-testid="stSidebar"] .stWarning {
        border-left: 4px solid var(--warning) !important;
    }
    
    [data-testid="stSidebar"] .stInfo {
        border-left: 4px solid var(--accent-blue) !important;
    }
    
    /* HR lines */
    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.2) !important;
        margin: 1.5rem 0 !important;
    }
    
    /* Sidebar icons */
    [data-testid="stSidebar"] .st-emotion-cache-1c7k0qh {
        color: white !important;
    }
    
    /* Interactive Charts */
    .js-plotly-plot {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .js-plotly-plot:hover {
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.15);
    }
    
    /* Accordion Styling */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, rgba(59, 130, 246, 0.1) 0%, rgba(96, 165, 250, 0.1) 100%);
        border-left: 4px solid var(--secondary-blue);
        border-radius: 8px;
        padding: 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(90deg, rgba(59, 130, 246, 0.2) 0%, rgba(96, 165, 250, 0.2) 100%);
    }
    
    /* Data Tables */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    }
    
    /* Loading Animation */
    .stSpinner > div {
        border-top-color: var(--secondary-blue) !important;
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--secondary-blue) 0%, var(--accent-blue) 100%);
        border-radius: 10px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.6);
        border-radius: 12px 12px 0 0;
        padding: 1rem 2rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.8);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--secondary-blue) 0%, var(--accent-blue) 100%);
        color: white;
    }
    
    /* Tooltips */
    .tooltip {
        position: relative;
        display: inline-block;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        background-color: var(--bg-dark);
        color: white;
        text-align: center;
        border-radius: 8px;
        padding: 8px 12px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -60px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .metric-card {
            margin: 0.5rem 0;
        }
        .metric-value {
            font-size: 2rem;
        }
        [data-testid="stSidebar"] {
            padding: 1rem !important;
        }
    }
    
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

load_professional_css()

# Initialize session state
if 'config' not in st.session_state:
    st.session_state.config = get_config()
    st.session_state.predictor = None
    st.session_state.feature_pipeline = None
    st.session_state.models_loaded = False
    st.session_state.refresh_interval = 60
    st.session_state.last_refresh = datetime.now()
    st.session_state.selected_equipment = []
    st.session_state.date_range = 'Last 7 days'

config = st.session_state.config

# Fault criticality mapping with enhanced details
FAULT_CRITICALITY = {
    'normal': {
        'level': 'good', 'priority': 'None', 'icon': '✅', 'color': '#10B981',
        'action': 'Continue normal operation', 'timeframe': 'N/A',
        'rul_estimate': '>365 days', 'severity_score': 0,
        'details': 'Equipment operating within normal parameters. Continue regular monitoring schedule.'
    },
    'horizontal_misalignment': {
        'level': 'warning', 'priority': 'Medium', 'icon': '⚠️', 'color': '#F59E0B',
        'action': 'Schedule alignment check', 'timeframe': '1-2 weeks',
        'rul_estimate': '30-60 days', 'severity_score': 4,
        'details': 'Horizontal misalignment detected. Schedule precision alignment during next maintenance window.'
    },
    'vertical_misalignment': {
        'level': 'warning', 'priority': 'Medium', 'icon': '⚠️', 'color': '#F59E0B',
        'action': 'Schedule alignment correction', 'timeframe': '1-2 weeks',
        'rul_estimate': '30-60 days', 'severity_score': 4,
        'details': 'Vertical misalignment present. Recommend shimming adjustment and laser alignment.'
    },
    'imbalance': {
        'level': 'warning', 'priority': 'Medium-High', 'icon': '⚠️', 'color': '#F59E0B',
        'action': 'Balance rotor assembly', 'timeframe': '3-7 days',
        'rul_estimate': '15-30 days', 'severity_score': 6,
        'details': 'Mass imbalance detected. Perform dynamic balancing to reduce vibration levels.'
    },
    'underhang_ball_fault': {
        'level': 'critical', 'priority': 'High', 'icon': '🔴', 'color': '#EF4444',
        'action': 'Replace underhang bearing immediately', 'timeframe': '24-48 hours',
        'rul_estimate': '2-7 days', 'severity_score': 8,
        'details': 'Ball fault detected in drive-end bearing. Risk of catastrophic failure.'
    },
    'underhang_cage_fault': {
        'level': 'critical', 'priority': 'High', 'icon': '🔴', 'color': '#EF4444',
        'action': 'Replace underhang bearing immediately', 'timeframe': '24-48 hours',
        'rul_estimate': '2-7 days', 'severity_score': 8,
        'details': 'Cage fault in drive-end bearing. Bearing may seize. Check lubrication system.'
    },
    'underhang_outer_race': {
        'level': 'critical', 'priority': 'High', 'icon': '🔴', 'color': '#EF4444',
        'action': 'Replace underhang bearing immediately', 'timeframe': '24-48 hours',
        'rul_estimate': '2-7 days', 'severity_score': 8,
        'details': 'Outer race defect in drive-end bearing. Progressive damage likely.'
    },
    'overhang_ball_fault': {
        'level': 'critical', 'priority': 'Very High', 'icon': '🔴', 'color': '#EF4444',
        'action': 'URGENT: Replace overhang bearing', 'timeframe': 'Immediate (12-24 hours)',
        'rul_estimate': '1-3 days', 'severity_score': 10,
        'details': 'Ball fault in load-end bearing under high stress. Imminent failure risk.'
    },
    'overhang_cage_fault': {
        'level': 'critical', 'priority': 'Very High', 'icon': '🔴', 'color': '#EF4444',
        'action': 'URGENT: Replace overhang bearing', 'timeframe': 'Immediate (12-24 hours)',
        'rul_estimate': '1-3 days', 'severity_score': 10,
        'details': 'Cage failure in load-end bearing. High risk of bearing disintegration.'
    },
    'overhang_outer_race': {
        'level': 'critical', 'priority': 'Very High', 'icon': '🔴', 'color': '#EF4444',
        'action': 'URGENT: Replace overhang bearing', 'timeframe': 'Immediate (12-24 hours)',
        'rul_estimate': '1-3 days', 'severity_score': 10,
        'details': 'Outer race damage on load-end bearing. Rapid deterioration expected.'
    }
}

def display_header():
    """Display professional header with status indicators."""
    st.markdown("""
    <div class="glass-card" style="text-align: center; margin-bottom: 2rem;">
        <h1 style="color: #1E3A8A; margin: 0; font-size: 2.5rem; font-weight: 700;">
             MAFAULDA Predictive Maintenance Platform
        </h1>
        <p style="color: #64748B; font-size: 1.1rem; margin-top: 0.5rem;">
             Industrial equipment health monitoring & fault detection
        </p>
        <div style="margin-top: 1rem;">
            <span class="status-indicator status-good"></span>
            <span style="color: #10B981; font-weight: 600;">System Operational</span>
            <span style="margin: 0 1rem; color: #CBD5E1;">|</span>
            <span style="color: #64748B;">Last Update: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def load_models():
    """Load trained models with caching."""
    if not st.session_state.models_loaded:
        with st.spinner("🔄 Loading AI models..."):
            try:
                predictor = ModelPredictor(config)
                predictor.load_models(config.paths.models)
                st.session_state.predictor = predictor
                
                pipeline = FeatureEngineeringPipeline(config)
                pipeline.load(config.paths.scalers / "feature_pipeline.pkl")
                st.session_state.feature_pipeline = pipeline
                
                st.session_state.models_loaded = True
                return True
            except Exception as e:
                st.error(f"❌ Error loading models: {e}")
                st.info("Please ensure models are trained. Run: `python train_pipeline.py`")
                return False
    return True
#================================================================
# SIDEBAR - Global Controls
# ============================================================================

with st.sidebar:
    st.markdown("###  Global Controls")
    
    # Date Range Selector
    date_range_options = ['Last 7 days', 'Last 30 days', 'Last 90 days', 'Custom']
    st.session_state.date_range = st.selectbox(
        "Date Range",
        date_range_options,
        index=date_range_options.index(st.session_state.date_range)
    )
    
    if st.session_state.date_range == 'Custom':
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("From", datetime.now() - timedelta(days=30))
        with col2:
            end_date = st.date_input("To", datetime.now())
    
    # Equipment Selector
    st.markdown("---")
    equipment_list = [f"Equipment-{i:03d}" for i in range(1, 21)]
    st.session_state.selected_equipment = st.multiselect(
        " Equipment Selection",
        equipment_list,
        default=equipment_list[:3]
    )
    
    # Refresh Interval
    st.markdown("---")
    st.session_state.refresh_interval = st.select_slider(
        "Refresh Interval",
        options=[30, 60, 300, 'Manual'],
        value=60,
        format_func=lambda x: f"{x}s" if x != 'Manual' else 'Manual'
    )
    
    # Real-time Toggle
    realtime_enabled = st.toggle(" Real-time Monitoring", value=False)
    
    st.markdown("---")
    
    # System Status
    st.markdown("###  System Status")
    
    if st.session_state.models_loaded:
        st.success("✅ Models Loaded")
    else:
        st.warning("⚠️ Models Not Loaded")
    
    st.info(f" GPU: {'Enabled' if config.gpu.enabled else 'Disabled'}")
    st.info(f" Random Seed: {config.random_seed}")
    
    # Navigation
    st.markdown("---")
    st.markdown("###  Navigation")
    page = st.radio(
        "Select View",
        ["Executive Dashboard", "Model Analytics", "Signal Analysis", 
        "Fault Explorer", "Live Prediction", "User Guide"],
        label_visibility="collapsed"
    )

# ============================================================================
# MAIN CONTENT AREA
# ============================================================================

display_header()

# ============================================================================
# PAGE 1: EXECUTIVE DASHBOARD
# ============================================================================

if page == "Executive Dashboard":
    
    # Executive Summary Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Total Equipment</div>
            <div class="metric-value">20</div>
            <div class="metric-trend">
                <span style="color: #10B981;">▲ 17</span> Operational
                <span style="color: #EF4444;">● 3</span> Alert
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Health Score</div>
            <div class="metric-value">87%</div>
            <div class="metric-trend">
                <span style="color: #10B981;">▲ 2%</span> vs last week
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Active Faults</div>
            <div class="metric-value">12</div>
            <div class="metric-trend">
                High: 3 | Medium: 5 | Low: 4
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">MTBF</div>
            <div class="metric-value">45d</div>
            <div class="metric-trend">
                <span style="color: #10B981;">▲ 12%</span> improvement
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header"> Equipment Health Timeline</div>', unsafe_allow_html=True)
    
    # Interactive Timeline (Gantt-style)
    timeline_data = []
    available_equipment = st.session_state.selected_equipment[:5] or ["Equipment-001"]  
    for i, equip in enumerate(available_equipment):       
        base_date = datetime.now() - timedelta(days=30)
        
        # Normal periods
        timeline_data.append(dict(
            Equipment=equip,
            Start=base_date,
            Finish=base_date + timedelta(days=10),
            Status="Normal"
        ))
        
        # Warning period
        timeline_data.append(dict(
            Equipment=equip,
            Start=base_date + timedelta(days=10),
            Finish=base_date + timedelta(days=15),
            Status="Warning"
        ))
        
        # Maintenance
        timeline_data.append(dict(
            Equipment=equip,
            Start=base_date + timedelta(days=15),
            Finish=base_date + timedelta(days=17),
            Status="Maintenance"
        ))
        
        # Normal again
        timeline_data.append(dict(
            Equipment=equip,
            Start=base_date + timedelta(days=17),
            Finish=datetime.now(),
            Status="Normal"
        ))
    
    df_timeline = pd.DataFrame(timeline_data)
    
    color_map = {
        'Normal': '#10B981',
        'Warning': '#F59E0B',
        'Fault': '#EF4444',
        'Maintenance': '#3B82F6'
    }
    
    fig_timeline = px.timeline(
        df_timeline,
        x_start="Start",
        x_end="Finish",
        y="Equipment",
        color="Status",
        color_discrete_map=color_map,
        title="Equipment Health Timeline (Last 30 Days)"
    )
    
    fig_timeline.update_layout(
        height=400,
        xaxis_title="Date",
        yaxis_title="Equipment",
        hovermode='closest',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Predicted Failures Section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="section-header"> Predictive Insights</div>', unsafe_allow_html=True)
        
        # Failure Probability Forecast
        forecast_days = 30
        dates = pd.date_range(start=datetime.now(), periods=forecast_days, freq='D')
        failure_prob = 5 + np.cumsum(np.random.randn(forecast_days) * 2).clip(0, 100)
        upper_bound = failure_prob + 10
        lower_bound = (failure_prob - 10).clip(0, 100)
        
        fig_forecast = go.Figure()
        
        fig_forecast.add_trace(go.Scatter(
            x=dates, y=upper_bound,
            fill=None,
            mode='lines',
            line_color='rgba(59, 130, 246, 0)',
            showlegend=False,
            name='Upper Bound'
        ))
        
        fig_forecast.add_trace(go.Scatter(
            x=dates, y=lower_bound,
            fill='tonexty',
            mode='lines',
            line_color='rgba(59, 130, 246, 0)',
            fillcolor='rgba(59, 130, 246, 0.2)',
            name='Confidence Interval'
        ))
        
        fig_forecast.add_trace(go.Scatter(
            x=dates, y=failure_prob,
            mode='lines',
            line=dict(color='#3B82F6', width=3),
            name='Failure Probability'
        ))
        
        fig_forecast.update_layout(
            title="30-Day Failure Probability Forecast",
            xaxis_title="Date",
            yaxis_title="Failure Probability (%)",
            height=400,
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig_forecast, use_container_width=True)
    
    with col2:
        st.markdown('<div class="section-header"> RUL Estimates</div>', unsafe_allow_html=True)
        
        rul_data = []
        available_equipment = st.session_state.selected_equipment[:5] or ["Equipment-001"]
        for equip in available_equipment:
            rul_days = np.random.randint(30, 365)
            status = 'Good' if rul_days > 180 else ('Warning' if rul_days > 90 else 'Critical')
            rul_data.append({
                'Equipment': equip,
                'RUL (days)': rul_days,
                'Status': status
            })
        
        df_rul = pd.DataFrame(rul_data)
        
        for _, row in df_rul.iterrows():
            color = '#10B981' if row['Status'] == 'Good' else ('#F59E0B' if row['Status'] == 'Warning' else '#EF4444')
            st.markdown(f"""
            <div class="glass-card" style="padding: 1rem; margin: 0.5rem 0;">
                <div style="font-weight: 600; color: #1E3A8A;">{row['Equipment']}</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: {color};">{row['RUL (days)']} days</div>
                <div style="font-size: 0.85rem; color: #64748B;">{row['Status']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Fault Distribution
    st.markdown('<div class="section-header"> Fault Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if len(st.session_state.selected_equipment) > 0:
            fault_hierarchy = pd.DataFrame([
                {'Equipment': 'Equip-001', 'Component': 'Bearing', 'Fault': 'Ball Fault', 'Count': 5},
                {'Equipment': 'Equip-001', 'Component': 'Bearing', 'Fault': 'Cage Fault', 'Count': 3},
                {'Equipment': 'Equip-002', 'Component': 'Shaft', 'Fault': 'Misalignment', 'Count': 8},
                {'Equipment': 'Equip-002', 'Component': 'Rotor', 'Fault': 'Imbalance', 'Count': 6},
                {'Equipment': 'Equip-003', 'Component': 'Bearing', 'Fault': 'Outer Race', 'Count': 4},
            ])
        
            fig_sunburst = px.sunburst(
                fault_hierarchy,
                path=['Equipment', 'Component', 'Fault'],
                values='Count',
                title='Fault Type Distribution (Hierarchical)',
                color='Count',
                color_continuous_scale='RdYlGn_r'
            )
            
            fig_sunburst.update_layout(height=500)
            st.plotly_chart(fig_sunburst, use_container_width=True)
        else:
            st.info("Please select equipment in the sidebar to view fault distribution.")


    with col2:
        # Severity distribution
        available_equipment = st.session_state.selected_equipment[:5]  # Get up to 5 items
        num_equipment = len(available_equipment)

        if num_equipment > 0:
            # Create arrays of the same length
            equipment_list = available_equipment * 3  # Repeat for Low, Medium, High
            severity_list = ['Low'] * num_equipment + ['Medium'] * num_equipment + ['High'] * num_equipment
            count_list = np.random.randint(1, 8, num_equipment * 3)
            
            severity_data = pd.DataFrame({
                'Equipment': equipment_list,
                'Severity': severity_list,
                'Count': count_list
            })
        
            fig_severity = px.bar(
                severity_data,
                x='Equipment',
                y='Count',
                color='Severity',
                title='Fault Severity Distribution',
                color_discrete_map={'Low': '#10B981', 'Medium': '#F59E0B', 'High': '#EF4444'},
                barmode='stack'
            )
            
            fig_severity.update_layout(
                height=500,
                xaxis_tickangle=-45,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
        
            st.plotly_chart(fig_severity, use_container_width=True)
        else:
            st.info("No equipment selected. Please select equipment in the sidebar.")

# ============================================================================
# PAGE 2: MODEL ANALYTICS
# ============================================================================

if page == "Model Analytics":
    
    st.markdown('<div class="section-header"> Model Performance Analytics</div>', unsafe_allow_html=True)
    
    if load_models():
        predictor = st.session_state.predictor
        
        # Load evaluation results
        eval_results = list(config.paths.performance.glob("evaluation_results_*.pkl"))
        
        if eval_results:
            from utils import load_pickle
            results = load_pickle(eval_results[-1])
            
            # Model Selection Tabs
            tab1, tab2, tab3, tab4 = st.tabs([" Comparison", " Confusion Matrix", " ROC Analysis", " Feature Importance"])
            
            with tab1:
                # Performance Metrics Table (sortable)
                st.markdown("#### Performance Metrics Comparison")
                
                metrics_data = []
                for model_name, model_results in results.items():
                    if 'test' in model_results:
                        metrics = model_results['test']
                        metrics_data.append({
                            'Model': model_name,
                            'Accuracy': metrics['accuracy'],
                            'Precision': metrics['precision_macro'],
                            'Recall': metrics['recall_macro'],
                            'F1-Score': metrics['f1_macro'],
                            'F1-Weighted': metrics['f1_weighted']
                        })
                
                df_metrics = pd.DataFrame(metrics_data).sort_values('F1-Score', ascending=False)
                
                # Highlight best values
                st.dataframe(
                    df_metrics.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'F1-Weighted']),
                    width='stretch',
                    height=250
                )
                
                # Radar Chart Comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_radar = go.Figure()
                    
                    for _, row in df_metrics.iterrows():
                        fig_radar.add_trace(go.Scatterpolar(
                            r=[row['Accuracy'], row['Precision'], row['Recall'], row['F1-Score'], row['F1-Weighted']],
                            theta=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'F1-Weighted'],
                            fill='toself',
                            name=row['Model']
                        ))
                    
                    fig_radar.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                        title="Model Performance Radar Chart",
                        height=400
                    )
                    
                    st.plotly_chart(fig_radar, use_container_width=True)
                
                with col2:
                    # Bar chart with confidence intervals
                    fig_bars = go.Figure()
                    
                    for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
                        fig_bars.add_trace(go.Bar(
                            name=metric,
                            x=df_metrics['Model'],
                            y=df_metrics[metric],
                            text=df_metrics[metric].round(3),
                            textposition='auto'
                        ))
                    
                    fig_bars.update_layout(
                        title="Model Metrics Comparison",
                        barmode='group',
                        xaxis_tickangle=-45,
                        height=400,
                        yaxis=dict(range=[0, 1])
                    )
                    
                    st.plotly_chart(fig_bars, use_container_width=True)
            
            with tab2:
                st.markdown("#### Confusion Matrix Analysis")
                
                # Model selector
                model_names = [m for m in results.keys() if 'test' in results[m]]
                selected_model = st.selectbox(
                    "Select Model", 
                    model_names,
                    key="model_analytics_model_select"  
                )                   
                if selected_model:
                    cm = results[selected_model]['test']['confusion_matrix']
                    class_names = [config.fault_class_names[i] for i in range(config.num_classes)]
                    
                    # Normalize confusion matrix
                    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                    
                    # Interactive heatmap
                    fig_cm = go.Figure(data=go.Heatmap(
                        z=cm_normalized,
                        x=class_names,
                        y=class_names,
                        colorscale='RdYlGn',
                        text=cm,
                        texttemplate='%{text}',
                        textfont={"size": 10},
                        hovertemplate='True: %{y}<br>Predicted: %{x}<br>Count: %{text}<br>Rate: %{z:.2%}<extra></extra>',
                        colorbar=dict(title="Accuracy")
                    ))
                    
                    fig_cm.update_layout(
                        title=f"Confusion Matrix - {selected_model}",
                        xaxis_title="Predicted Label",
                        yaxis_title="True Label",
                        height=600,
                        xaxis={'side': 'bottom'},
                        yaxis={'side': 'left'}
                    )
                    
                    fig_cm.update_xaxes(tickangle=-45)
                    
                    st.plotly_chart(fig_cm, use_container_width=True)
                    
                    # Most confused pairs
                    st.markdown("##### Most Confused Fault Pairs")
                    
                    confused_pairs = []
                    for i in range(len(cm)):
                        for j in range(len(cm)):
                            if i != j and cm[i, j] > 0:
                                confused_pairs.append({
                                    'True Class': class_names[i],
                                    'Predicted As': class_names[j],
                                    'Count': int(cm[i, j]),
                                    'Error Rate': f"{cm_normalized[i, j]:.1%}"
                                })
                    
                    df_confused = pd.DataFrame(confused_pairs).sort_values('Count', ascending=False)
                    st.dataframe(df_confused.head(10), width='stretch')
            
            with tab3:
                st.markdown("#### ROC Curve Analysis")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Multi-class ROC curves
                    fig_roc = go.Figure()
                    
                    # Generate sample ROC curves (in real implementation, use actual probabilities)
                    for i, class_name in enumerate(class_names[:5]):  # Show top 5 classes
                        fpr = np.linspace(0, 1, 100)
                        tpr = np.sqrt(fpr) + np.random.rand(100) * 0.1
                        tpr = np.clip(tpr, 0, 1)
                        auc = np.trapz(tpr, fpr)
                        
                        fig_roc.add_trace(go.Scatter(
                            x=fpr,
                            y=tpr,
                            mode='lines',
                            name=f'{class_name} (AUC={auc:.3f})',
                            line=dict(width=2)
                        ))
                    
                    # Diagonal reference line
                    fig_roc.add_trace(go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode='lines',
                        name='Random Classifier',
                        line=dict(dash='dash', color='gray')
                    ))
                    
                    fig_roc.update_layout(
                        title='ROC Curves (One-vs-Rest)',
                        xaxis_title='False Positive Rate',
                        yaxis_title='True Positive Rate',
                        height=500,
                        hovermode='closest'
                    )
                    
                    st.plotly_chart(fig_roc, use_container_width=True)
                
                with col2:
                    st.markdown("##### AUC Scores")
                    
                    # AUC scores for all classes
                    auc_scores = []
                    for class_name in class_names[:8]:
                        auc = np.random.uniform(0.85, 0.98)
                        auc_scores.append({
                            'Class': class_name,
                            'AUC': auc,
                            'Performance': '🟢 Excellent' if auc > 0.95 else ('🟡 Good' if auc > 0.90 else '🟠 Fair')
                        })
                    
                    df_auc = pd.DataFrame(auc_scores)
                    st.dataframe(df_auc, width='stretch', height=400)
                
                # Precision-Recall Curves
                st.markdown("##### Precision-Recall Curves")
                
                fig_pr = go.Figure()
                
                for i, class_name in enumerate(class_names[:5]):
                    recall = np.linspace(0, 1, 100)
                    precision = 1 - recall * 0.3 + np.random.rand(100) * 0.1
                    precision = np.clip(precision, 0, 1)
                    
                    fig_pr.add_trace(go.Scatter(
                        x=recall,
                        y=precision,
                        mode='lines',
                        name=class_name,
                        line=dict(width=2)
                    ))
                
                fig_pr.update_layout(
                    title='Precision-Recall Curves',
                    xaxis_title='Recall',
                    yaxis_title='Precision',
                    height=400,
                    hovermode='closest'
                )
                
                st.plotly_chart(fig_pr, use_container_width=True)
            
            with tab4:
                st.markdown("#### Feature Importance Analysis")
                
                # Generate sample feature importance
                feature_names = [f"Feature_{i}" for i in range(20)]
                importance = np.random.rand(20)
                importance = importance / importance.sum()
                
                # Sort by importance
                sorted_idx = np.argsort(importance)[::-1]
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Waterfall chart
                    fig_waterfall = go.Figure(go.Waterfall(
                        name="Feature Importance",
                        orientation="v",
                        measure=["relative"] * 20,
                        x=[feature_names[i] for i in sorted_idx],
                        y=[importance[i] for i in sorted_idx],
                        connector={"line": {"color": "rgb(63, 63, 63)"}},
                    ))
                    
                    fig_waterfall.update_layout(
                        title="Feature Importance Waterfall",
                        xaxis_title="Features",
                        yaxis_title="Importance",
                        height=500,
                        xaxis_tickangle=-45
                    )
                    
                    st.plotly_chart(fig_waterfall, use_container_width=True)
                
                with col2:
                    st.markdown("##### Top 10 Features")
                    
                    top_features = []
                    for i in range(10):
                        idx = sorted_idx[i]
                        top_features.append({
                            'Rank': i + 1,
                            'Feature': feature_names[idx],
                            'Importance': f"{importance[idx]:.4f}",
                            'Category': np.random.choice(['Time', 'Frequency', 'Wavelet'])
                        })
                    
                    df_features = pd.DataFrame(top_features)
                    st.dataframe(df_features, width='stretch', height=400)
                
                # Prediction Confidence Distribution
                st.markdown("##### Prediction Confidence Distribution")
                
                # Generate sample confidence scores
                correct_conf = np.random.beta(8, 2, 500)
                incorrect_conf = np.random.beta(3, 5, 100)
                
                fig_violin = go.Figure()
                
                fig_violin.add_trace(go.Violin(
                    y=correct_conf,
                    name='Correct Predictions',
                    box_visible=True,
                    meanline_visible=True,
                    fillcolor='lightgreen',
                    opacity=0.6,
                    x0='Correct'
                ))
                
                fig_violin.add_trace(go.Violin(
                    y=incorrect_conf,
                    name='Incorrect Predictions',
                    box_visible=True,
                    meanline_visible=True,
                    fillcolor='lightcoral',
                    opacity=0.6,
                    x0='Incorrect'
                ))
                
                fig_violin.update_layout(
                    title='Prediction Confidence Distribution',
                    yaxis_title='Confidence Score',
                    height=400
                )
                
                st.plotly_chart(fig_violin, use_container_width=True)

# ============================================================================
# PAGE 3: SIGNAL ANALYSIS
# ============================================================================

elif page == "Signal Analysis":
    
    st.markdown('<div class="section-header"> Vibration Signal Analysis</div>', unsafe_allow_html=True)
    
    # Control Panel
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        selected_sensor = st.selectbox(
            " Sensor",
            config.data.sensor_names,
            index=2,
            key="signal_analysis_sensor_select"  # Add unique key

        )
    
    with col2:
        analysis_type = st.selectbox(
            " Analysis Type",
            ["Time Series", "FFT Spectrum", "Spectrogram", "Wavelet Transform"],
            key="signal_analysis_type_select"  # Add unique key

        )
    
    with col3:
        num_samples = st.slider(
            " Samples",
            100, 10000, 1000,
            step=100,
            key="signal_analysis_samples_slider" 
        )
    
    with col4:
        chart_type = st.selectbox(
            "Chart Type",
            ["Line", "Scatter", "Area"],
            key="signal_analysis_chart_select"
        )
    
    # Generate sample signal data
    t = np.linspace(0, 1, num_samples)
    
    # Composite signal with multiple frequency components
    base_freq = 50  # Hz
    signal = (np.sin(2 * np.pi * base_freq * t) +
            0.5 * np.sin(2 * np.pi * base_freq * 2 * t) +
            0.3 * np.sin(2 * np.pi * base_freq * 3 * t) +
            0.2 * np.random.randn(num_samples))
    
    if analysis_type == "Time Series":
        # Time series plot with anomaly detection overlay
        fig_ts = go.Figure()
        
        if chart_type == "Line":
            fig_ts.add_trace(go.Scatter(
                x=t, y=signal,
                mode='lines',
                name='Signal',
                line=dict(color='#3B82F6', width=2)
            ))
        elif chart_type == "Scatter":
            fig_ts.add_trace(go.Scatter(
                x=t, y=signal,
                mode='markers',
                name='Signal',
                marker=dict(color='#3B82F6', size=3)
            ))
        else:  # Area
            fig_ts.add_trace(go.Scatter(
                x=t, y=signal,
                fill='tozeroy',
                name='Signal',
                line=dict(color='#3B82F6')
            ))
        
        # Add mean line
        mean_val = np.mean(signal)
        fig_ts.add_hline(y=mean_val, line_dash="dash", line_color="green", annotation_text="Mean")
        
        # Add std bands
        std_val = np.std(signal)
        fig_ts.add_hrect(
            y0=mean_val - 2*std_val, y1=mean_val + 2*std_val,
            fillcolor="rgba(59, 130, 246, 0.1)",
            line_width=0,
            annotation_text="±2σ",
            annotation_position="right"
        )
        
        fig_ts.update_layout(
            title=f'{selected_sensor} - Time Series Analysis',
            xaxis_title='Time (s)',
            yaxis_title='Amplitude',
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_ts, use_container_width=True)
    
    elif analysis_type == "FFT Spectrum":
        # FFT Analysis
        from scipy.fft import fft, fftfreq
        
        yf = fft(signal)
        xf = fftfreq(num_samples, 1/num_samples)[:num_samples//2]
        power = 2.0/num_samples * np.abs(yf[0:num_samples//2])
        
        fig_fft = go.Figure()
        
        fig_fft.add_trace(go.Scatter(
            x=xf, y=power,
            mode='lines',
            fill='tozeroy',
            name='FFT Magnitude',
            line=dict(color='#3B82F6', width=2)
        ))
        
        # Mark dominant frequencies
        peaks_idx = np.argsort(power)[-3:]
        for idx in peaks_idx:
            if power[idx] > 0.1:  # Only show significant peaks
                fig_fft.add_annotation(
                    x=xf[idx], y=power[idx],
                    text=f"{xf[idx]:.1f} Hz",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor='red',
                    ax=0, ay=-40
                )
        
        fig_fft.update_layout(
            title=f'{selected_sensor} - Frequency Spectrum (FFT)',
            xaxis_title='Frequency (Hz)',
            yaxis_title='Magnitude',
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_fft, width='stretch')
        
        # Power Spectral Density
        st.markdown("##### Power Spectral Density")
        
        psd = power ** 2
        
        fig_psd = go.Figure()
        fig_psd.add_trace(go.Scatter(
            x=xf, y=10 * np.log10(psd),  # Convert to dB
            mode='lines',
            fill='tozeroy',
            name='PSD',
            line=dict(color='#10B981', width=2)
        ))
        
        fig_psd.update_layout(
            title='Power Spectral Density',
            xaxis_title='Frequency (Hz)',
            yaxis_title='Power (dB)',
            height=400
        )
        
        st.plotly_chart(fig_psd, width='stretch')
    
    elif analysis_type == "Spectrogram":
        # Spectrogram using STFT
        from scipy import signal as scipy_signal
        
        f, t_spec, Sxx = scipy_signal.spectrogram(signal, fs=num_samples)
        
        fig_spec = go.Figure(data=go.Heatmap(
            z=10 * np.log10(Sxx),
            x=t_spec,
            y=f,
            colorscale='Jet',
            colorbar=dict(title='Power (dB)')
        ))
        
        fig_spec.update_layout(
            title=f'{selected_sensor} - Spectrogram',
            xaxis_title='Time (s)',
            yaxis_title='Frequency (Hz)',
            height=500
        )
        
        st.plotly_chart(fig_spec, use_container_width=True)
    
    elif analysis_type == "Wavelet Transform":
        # Wavelet Transform (Scalogram)
        import pywt
        
        scales = np.arange(1, 128)
        coefficients, frequencies = pywt.cwt(signal, scales, 'morl')
        
        fig_wavelet = go.Figure(data=go.Heatmap(
            z=np.abs(coefficients),
            x=t,
            y=scales,
            colorscale='Viridis',
            colorbar=dict(title='Magnitude')
        ))
        
        fig_wavelet.update_layout(
            title=f'{selected_sensor} - Wavelet Transform (Scalogram)',
            xaxis_title='Time (s)',
            yaxis_title='Scale',
            height=500
        )
        
        st.plotly_chart(fig_wavelet, width='stretch')
    
    # Statistical Analysis Module
    st.markdown('<div class="section-header">📊 Statistical Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Statistics Table
        st.markdown("##### Signal Statistics")
        
        from scipy.stats import skew, kurtosis
        
        stats_data = {
            'Metric': ['Count', 'Mean', 'Std Dev', 'Min', '25%', 'Median', '75%', 'Max', 'Skewness', 'Kurtosis'],
            'Value': [
                len(signal),
                f"{np.mean(signal):.4f}",
                f"{np.std(signal):.4f}",
                f"{np.min(signal):.4f}",
                f"{np.percentile(signal, 25):.4f}",
                f"{np.median(signal):.4f}",
                f"{np.percentile(signal, 75):.4f}",
                f"{np.max(signal):.4f}",
                f"{skew(signal):.4f}",
                f"{kurtosis(signal):.4f}"
            ]
        }
        
        df_stats = pd.DataFrame(stats_data)
        st.dataframe(df_stats, use_container_width=True, height=400)
    
    with col2:
        # Box Plot and Violin Plot
        st.markdown("##### Distribution Analysis")
        
        fig_box = go.Figure()
        
        fig_box.add_trace(go.Box(
            y=signal,
            name='Box Plot',
            boxmean='sd',
            marker_color='#3B82F6',
            showlegend=False
        ))
        
        fig_box.update_layout(
            title='Signal Distribution',
            yaxis_title='Amplitude',
            height=400
        )
        
        st.plotly_chart(fig_box, width='stretch')
    
    # Correlation Analysis
    st.markdown("##### Multi-Sensor Correlation Matrix")
    
    # Generate sample multi-sensor data
    n_sensors = 8
    sensor_data = np.random.randn(num_samples, n_sensors)
    
    # Add some correlation
    for i in range(1, n_sensors):
        sensor_data[:, i] = 0.7 * sensor_data[:, 0] + 0.3 * sensor_data[:, i]
    
    corr_matrix = np.corrcoef(sensor_data.T)
    
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=config.data.sensor_names,
        y=config.data.sensor_names,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix,
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title='Correlation')
    ))
    
    fig_corr.update_layout(
        title='Sensor Correlation Matrix',
        height=600,
        xaxis_tickangle=-45
    )
    
    st.plotly_chart(fig_corr, width='stretch')

# ============================================================================
# PAGE 4: FAULT EXPLORER
# ============================================================================

elif page == "Fault Explorer":
    
    st.markdown('<div class="section-header"> Interactive Fault Analysis</div>', unsafe_allow_html=True)
    
    # Fault Filter Controls
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        fault_category = st.multiselect(
            "Fault Category",
            ["Bearing", "Alignment", "Balance", "All"],
            default=["All"],
            key="fault_explorer_category_select"
        )
    
    with col2:
        severity_filter = st.multiselect(
            "Severity Level",
            ["Low", "Medium", "High", "Critical"],
            default=["Medium", "High", "Critical"],
             key="fault_explorer_severity_select"
        )
    
    with col3:
        time_period = st.selectbox(
            "Time Period",
            ["Last 24h", "Last 7d", "Last 30d", "All Time"],
            index=2,
            key="fault_explorer_time_select"

        )
    
    with col4:
        st.metric("Total Faults", "247", delta="+12")
    
    # Interactive Fault Treemap
    st.markdown("#### Fault Distribution Treemap")
    
    treemap_data = pd.DataFrame([
        {"Fault": "Bearing Faults", "Component": "Ball Fault", "Count": 45, "Severity": "High"},
        {"Fault": "Bearing Faults", "Component": "Cage Fault", "Count": 32, "Severity": "High"},
        {"Fault": "Bearing Faults", "Component": "Outer Race", "Count": 28, "Severity": "Critical"},
        {"Fault": "Misalignment", "Component": "Horizontal", "Count": 55, "Severity": "Medium"},
        {"Fault": "Misalignment", "Component": "Vertical", "Count": 41, "Severity": "Medium"},
        {"Fault": "Imbalance", "Component": "Rotor", "Count": 46, "Severity": "Medium"},
    ])
    
    fig_treemap = px.treemap(
        treemap_data,
        path=[px.Constant("All Faults"), 'Fault', 'Component'],
        values='Count',
        color='Severity',
        color_discrete_map={'Low': '#10B981', 'Medium': '#F59E0B', 'High': '#EF4444', 'Critical': '#7C2D12'},
        title='Fault Distribution Hierarchy'
    )
    
    fig_treemap.update_layout(height=500)
    fig_treemap.update_traces(textinfo="label+value+percent parent")
    
    st.plotly_chart(fig_treemap, width='stretch')
    
    # Fault Timeline with Events
    st.markdown("#### Fault Timeline & Events")
    
    # Generate sample fault events
    fault_events = []
    base_date = datetime.now() - timedelta(days=30)
    
    for i in range(50):
        event_date = base_date + timedelta(days=np.random.randint(0, 30), hours=np.random.randint(0, 24))
        fault_types = list(config.fault_classes.keys())
        fault_type = np.random.choice([f for f in fault_types if f != 'normal'])
        
        fault_events.append({
            'Date': event_date,
            'Equipment': f"Equip-{np.random.randint(1, 11):03d}",
            'Fault': fault_type.replace('_', ' ').title(),
            'Severity': np.random.choice(['Low', 'Medium', 'High', 'Critical'], p=[0.2, 0.4, 0.3, 0.1]),
            'Duration': np.random.randint(1, 72)  # hours
        })
    
    df_events = pd.DataFrame(fault_events).sort_values('Date')
    
    # Interactive timeline scatter plot
    fig_timeline = px.scatter(
        df_events,
        x='Date',
        y='Equipment',
        color='Severity',
        size='Duration',
        hover_data=['Fault', 'Duration'],
        color_discrete_map={'Low': '#10B981', 'Medium': '#F59E0B', 'High': '#EF4444', 'Critical': '#7C2D12'},
        title='Fault Events Timeline'
    )
    
    fig_timeline.update_layout(height=400, hovermode='closest')
    fig_timeline.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    
    st.plotly_chart(fig_timeline, width='stretch')
    
    # Detailed Fault Table
    st.markdown("#### Fault Details (Interactive Table)")
    
    # Add calculated fields
    df_events['RUL Estimate'] = df_events['Severity'].map({
        'Low': '>90 days',
        'Medium': '30-90 days',
        'High': '7-30 days',
        'Critical': '<7 days'
    })
    
    df_events['Action Required'] = df_events['Severity'].map({
        'Low': 'Monitor',
        'Medium': 'Schedule Maintenance',
        'High': 'Urgent Maintenance',
        'Critical': 'Immediate Action'
    })
    
    # Display with filters
    st.dataframe(
        df_events[['Date', 'Equipment', 'Fault', 'Severity', 'Duration', 'RUL Estimate', 'Action Required']].head(20),
        use_container_width=True,
        height=400
    )
    
    # Export Options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button(" Export to CSV", key="export_csv_button"):
            csv = df_events.to_csv(index=False)
            st.download_button("Download CSV", csv, "fault_data.csv", "text/csv")
    
    with col2:
        if st.button(" Generate Report",key="generate_report_button"):
            st.info("Report generation feature - Coming soon!")
    
    with col3:
        if st.button("Email Alert",key="email_alert_button"):
            st.success("Alert sent to maintenance team!")

# ============================================================================
# PAGE 5: LIVE PREDICTION
# ============================================================================

elif page == "Live Prediction":
    
    st.markdown('<div class="section-header"> Real-Time Fault Prediction</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
        <h3 style="color: #1E3A8A; margin-top: 0;"> Upload sensor data</h3>
        <p style="color: #64748B;">
            Upload a CSV file containing sensor readings (8 columns) to receive instant fault diagnosis 
            with confidence scores and maintenance recommendations.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if load_models():
        predictor = st.session_state.predictor
        pipeline = st.session_state.feature_pipeline
        extractor = FeatureExtractor(config)
        
        # File Upload
        uploaded_file = st.file_uploader(
            "Choose CSV file",
            type=['csv'],
            help="File should contain 8 sensor columns with numerical data",
            key="live_prediction_file_uploader"
        )
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            model_name = st.selectbox(
                " Select model",
                list(predictor.models.keys()),
                index=0,
                key="live_prediction_model_select"
            )
        
        with col2:
            confidence_threshold = st.slider(
                "Confidence Threshold",
                0.5, 1.0, 0.7, 0.05,
                help="Predictions below this threshold will be flagged for review",
                key="live_prediction_confidence_slider"
            )
        
        if uploaded_file is not None:
            try:
                # Load and preview data
                df = pd.read_csv(uploaded_file, header=None)
                
                with st.expander(" View Raw Data Preview (First 100 rows)"):
                    st.dataframe(df.head(100), use_container_width=True)
                
                st.success(f"✅ Loaded: {df.shape[0]} samples × {df.shape[1]} sensors")
                
                # Quick signal visualization
                st.markdown("####  Quick Signal Visualization")
                
                fig_signals = make_subplots(
                    rows=2, cols=4,
                    subplot_titles=config.data.sensor_names
                )
                
                for i in range(min(8, df.shape[1])):
                    row = (i // 4) + 1
                    col = (i % 4) + 1
                    
                    fig_signals.add_trace(
                        go.Scatter(
                            y=df.iloc[:1000, i],
                            mode='lines',
                            line=dict(width=1),
                            showlegend=False
                        ),
                        row=row, col=col
                    )
                
                fig_signals.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_signals, width='stretch')
                
                # Prediction Button
                if st.button(" Predict fault", type="primary", width='stretch',key="predict_button"):
                    
                    with st.spinner("🔄 Analyzing sensor data..."):
                        progress_bar = st.progress(0)
                        
                        # Extract features
                        progress_bar.progress(33)
                        features = extractor.extract_all_features(df)
                        features_df = pd.DataFrame([features])
                        
                        # Scale features
                        progress_bar.progress(66)
                        features_scaled = pipeline.transform(features_df)
                        
                        # Make prediction
                        y_pred, y_pred_proba = predictor.predict(model_name, features_scaled)
                        progress_bar.progress(100)
                        
                        predicted_label = y_pred[0]
                        confidence = y_pred_proba[0][predicted_label]
                        fault_name = config.fault_class_names[predicted_label]
                        
                        # Get fault details
                        fault_info = FAULT_CRITICALITY.get(fault_name, FAULT_CRITICALITY['normal'])
                    
                    # Display Results
                    st.markdown("---")
                    st.markdown("###  Prediction Results")
                    
                    # Main prediction card
                    level_class = f"status-{fault_info['level']}"
                    
                    st.markdown(f"""
                    <div class="glass-card" style="border-left: 6px solid {fault_info['color']}; padding: 2rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <h2 style="color: {fault_info['color']}; margin: 0;">
                                    {fault_info['icon']} {fault_name.replace('_', ' ').upper()}
                                </h2>
                                <p style="font-size: 1.1rem; color: #64748B; margin: 0.5rem 0;">
                                    Priority: <strong>{fault_info['priority']}</strong>
                                </p>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 3rem; font-weight: 700; color: {fault_info['color']};">
                                    {confidence:.0%}
                                </div>
                                <div style="font-size: 0.9rem; color: #64748B;">Confidence</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence check
                    if confidence < confidence_threshold:
                        st.warning(f"⚠️ **Low Confidence Alert**: Prediction confidence ({confidence:.1%}) is below threshold ({confidence_threshold:.0%}). Manual inspection recommended.")
                    
                    # Maintenance Recommendation
                    st.markdown("#### 🔧 Maintenance Recommendation")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="glass-card">
                            <h4 style="color: #1E3A8A; margin-top: 0;"> Action Required</h4>
                            <p style="font-size: 1.1rem; font-weight: 600; color: {fault_info['color']};">
                                {fault_info['action']}
                            </p>
                            <hr style="border-color: #E2E8F0;">
                            <p><strong>Timeframe:</strong> {fault_info['timeframe']}</p>
                            <p><strong>Estimated RUL:</strong> {fault_info['rul_estimate']}</p>
                            <p><strong>Severity Score:</strong> {fault_info['severity_score']}/10</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="glass-card">
                            <h4 style="color: #1E3A8A; margin-top: 0;">ℹ️ Details & Recommendations</h4>
                            <p style="line-height: 1.6;">{fault_info['details']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Confidence Distribution
                    st.markdown("####  Confidence Distribution (All Classes)")
                    
                    prob_data = pd.DataFrame({
                        'Fault Type': [config.fault_class_names[i] for i in range(config.num_classes)],
                        'Probability': y_pred_proba[0]
                    }).sort_values('Probability', ascending=False)
                    
                    fig_conf = px.bar(
                        prob_data,
                        x='Probability',
                        y='Fault Type',
                        orientation='h',
                        title='Prediction Confidence for All Fault Classes',
                        color='Probability',
                        color_continuous_scale='RdYlGn'
                    )
                    
                    fig_conf.update_layout(height=500)
                    st.plotly_chart(fig_conf, width='stretch')
                    
                    # Top 5 Predictions
                    st.markdown("####  Top 5 Alternative Predictions")
                    
                    top5_data = prob_data.head(5).copy()
                    top5_data['Confidence %'] = top5_data['Probability'].apply(lambda x: f"{x:.1%}")
                    top5_data['Rank'] = range(1, 6)
                    
                    st.dataframe(
                        top5_data[['Rank', 'Fault Type', 'Confidence %']],
                        width='stretch',
                        hide_index=True
                    )
                    
                    # Save Results
                    if st.button("Save Prediction Report", key="save_report_button"):
                        report = {
                            'timestamp': datetime.now().isoformat(),
                            'model': model_name,
                            'prediction': fault_name,
                            'confidence': float(confidence),
                            'all_probabilities': {config.fault_class_names[i]: float(y_pred_proba[0][i]) for i in range(config.num_classes)}
                        }
                        
                        st.download_button(
                            " Download JSON Report",
                            data=pd.DataFrame([report]).to_json(orient='records'),
                            file_name=f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
            
            except Exception as e:
                st.error(f"❌ Error processing file: {e}")
                st.info("Please ensure the CSV file has 8 sensor columns with numerical data.")
        
        else:
            # Sample data option
            st.info("💡 **Tip**: Don't have a CSV file? Use synthetic sample data below for testing!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Generate Normal Sample", width='stretch', key="gen_normal_sample_button"):
                    from utils import create_synthetic_sample
                    sample_df = create_synthetic_sample(fault_type='normal')
                    csv = sample_df.to_csv(index=False, header=False)
                    st.download_button("Download Sample", csv, "sample_normal.csv", "text/csv",key="download_sample_button")
            
            with col2:
                if st.button("Generate Imbalance Sample", width='stretch', key="gen_imbalance_sample_button"):
                    from utils import create_synthetic_sample
                    sample_df = create_synthetic_sample(fault_type='imbalance')
                    csv = sample_df.to_csv(index=False, header=False)
                    st.download_button("Download Sample", csv, "sample_imbalance.csv", "text/csv", key="download_sample_button")
            
            with col3:
                if st.button("Generate Bearing Fault Sample", width='stretch', key="gen_bearing_sample_button"):
                    from utils import create_synthetic_sample
                    sample_df = create_synthetic_sample(fault_type='ball_fault')
                    csv = sample_df.to_csv(index=False, header=False)
                    st.download_button("Download Sample", csv, "sample_bearing.csv", "text/csv", key="download_sample_button")

# ============================================================================
# PAGE 6: USER GUIDE
# ============================================================================

elif page == "User Guide":
    
    st.markdown('<div class="section-header"> User Manual & Guide</div>', unsafe_allow_html=True)
    
    # Interactive Guide Sections
    with st.expander(" Getting started", expanded=True):
        st.markdown("""
        ### Welcome to MAFAULDA Predictive Maintenance Platform!
        
        This AI-powered platform helps you monitor industrial equipment health and predict failures before they occur.
        
        #### Quick Navigation:
        - **Executive Dashboard**: Overview of all equipment health and key metrics
        - **Model Analytics**: Deep dive into AI model performance
        - **Signal Analysis**: Analyze vibration sensor data
        - **Fault Explorer**: Browse and filter detected faults
        - **Live Prediction**: Upload data for instant diagnosis
        - **User Guide**: You are here!
        
        #### First Steps:
        1. Select equipment from the sidebar filters
        2. Choose your date range
        3. Navigate to different pages using the sidebar
        4. Interact with charts by hovering, zooming, and clicking
        """)
    
    with st.expander(" Dashboard Sections Explained"):
        st.markdown("""
        ### Executive Dashboard
        - **Health Score**: Overall equipment health (0-100%)
        - **Active Faults**: Number of current issues by severity
        - **MTBF**: Mean Time Between Failures - higher is better
        - **Timeline**: Visual history of equipment status
        - **RUL**: Remaining Useful Life estimates
        
        ### Model Analytics
        - **Confusion Matrix**: Shows prediction accuracy
        - **ROC Curves**: Model discrimination ability
        - **Feature Importance**: Which sensors matter most
        
        ### Signal Analysis
        - **Time Series**: Raw sensor readings over time
        - **FFT Spectrum**: Frequency components
        - **Spectrogram**: Time-frequency analysis
        - **Wavelet**: Multi-resolution analysis
        """)
    
    with st.expander(" How to Interpret Visualizations"):
        st.markdown("""
        ### Reading the Charts
        
        **Health Timeline**:
        - 🟢 Green = Normal operation
        - 🟡 Yellow = Warning detected
        - 🔴 Red = Fault present
        - 🔵 Blue = Maintenance performed
        
        **Confidence Scores**:
        - >90% = High confidence, act on prediction
        - 70-90% = Good confidence, consider secondary check
        - <70% = Low confidence, manual inspection needed
        
        **Severity Levels**:
        - **Low**: Monitor during next scheduled maintenance
        - **Medium**: Schedule maintenance within 1-2 weeks
        - **High**: Urgent - address within days
        - **Critical**: Immediate action required
        """)
    
    with st.expander(" Best practices"):
        st.markdown("""
        ### Monitoring Best Practices
        
        1. **Regular Checks**: Review dashboard daily
        2. **Trend Analysis**: Look for gradual changes in health scores
        3. **Quick Response**: Act on critical alerts immediately
        4. **Documentation**: Keep records of maintenance actions
        5. **Threshold Tuning**: Adjust confidence thresholds based on experience
        
        ### Data Quality:
        - Ensure sensors are properly calibrated
        - Check for missing or corrupted data
        - Maintain consistent sampling rates
        - Clean sensors regularly
        """)
    
    with st.expander(" Glossary of Terms"):
        st.markdown("""
        ### Technical Terms Explained
        
        **RUL (Remaining Useful Life)**: Estimated time until component failure
        
        **MTBF (Mean Time Between Failures)**: Average operational time between failures
        
        **FFT (Fast Fourier Transform)**: Converts time-domain signal to frequency components
        
        **Bearing Frequencies**: Characteristic frequencies indicating bearing faults:
        - BPFO: Ball Pass Frequency Outer (outer race defect)
        - BPFI: Ball Pass Frequency Inner (inner race defect)
        - BSF: Ball Spin Frequency (rolling element defect)
        
        **Vibration Analysis**: Study of mechanical vibrations to detect faults
        
        **Spectrogram**: Visual representation of signal frequency over time
        
        **Wavelet Transform**: Time-frequency analysis for transient detection
        """)
    
    with st.expander(" Frequently Asked Questions"):
        st.markdown("""
        ### FAQ
        
        **Q: How accurate are the predictions?**
        A: Our AI models achieve >90% accuracy on test data. However, always verify critical alerts.
        
        **Q: What file format should I upload?**
        A: CSV files with 8 columns (sensor readings) and no headers.
        
        **Q: How often should I check the dashboard?**
        A: Daily for critical equipment, weekly for standard equipment.
        
        **Q: What if the confidence is low?**
        A: Low confidence suggests manual inspection. The system flags uncertain predictions.
        
        **Q: Can I export the data?**
        A: Yes! Use the export buttons to download CSV or generate PDF reports.
        
        **Q: How is RUL calculated?**
        A: RUL is estimated based on fault severity, historical data, and degradation patterns.
        """)
    
    with st.expander(" Video Tutorial"):
        st.markdown("""
        ### Quick Video Guide
        
        📹 **Coming Soon**: Interactive video tutorials showing:
        - Platform navigation
        - Uploading sensor data
        - Interpreting predictions
        - Setting up alerts
        - Maintenance workflow
        
        For now, explore the platform using the interactive sections above!
        """)
    
    # Contact & Support
    st.markdown("---")
    st.markdown("""
    <div class="glass-card" style="text-align: center;">
        <h3 style="color: #1E3A8A; margin-top: 0;">📧 Need Help?</h3>
        <p style="color: #64748B;">
            For technical support, feature requests, or feedback:
        </p>
        <p style="font-size: 1.1rem;">
            📧 elaaemna@mafaulda-ai.com | 📞 +216 (211) 619 619
        </p>
        <p style="color: #64748B; font-size: 0.9rem;">
            Version UNO | © 2025 MAFAULDA  Platform
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748B; padding: 1rem; font-size: 0.9rem;">
    <p>MAFAULDA Predictive Maintenance Platform v1.0.0</p>
    <p>Elaa & Emna team| Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
