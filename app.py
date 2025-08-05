import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Import custom modules
from models.xray_model import XRayAnalyzer
from models.symptom_assessment import SymptomAssessment
from utils.image_processing import ImageProcessor
from utils.result_processor import ResultProcessor
from data.tb_symptoms import TB_SYMPTOMS, DEMOGRAPHIC_FACTORS

# Configure page
st.set_page_config(
    page_title="TB Detection System",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'patient_data' not in st.session_state:
    st.session_state.patient_data = {}
if 'xray_result' not in st.session_state:
    st.session_state.xray_result = None
if 'symptom_result' not in st.session_state:
    st.session_state.symptom_result = None

def main():
    # Header
    st.title("🫁 Tuberculosis Detection System")
    st.markdown("""
    **Advanced TB Detection using AI-powered X-ray Analysis and Symptom Assessment**
    
    ⚠️ **Medical Disclaimer**: This system is designed to assist healthcare professionals in TB screening. 
    It should not replace professional medical diagnosis and treatment decisions.
    """)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Analysis Type",
        ["Complete Assessment", "X-ray Analysis Only", "Symptom Assessment Only", "Results Dashboard"]
    )
    
    if page == "Complete Assessment":
        complete_assessment()
    elif page == "X-ray Analysis Only":
        xray_analysis_only()
    elif page == "Symptom Assessment Only":
        symptom_assessment_only()
    elif page == "Results Dashboard":
        results_dashboard()

def complete_assessment():
    st.header("Complete TB Assessment")
    
    # Create tabs for organized input
    tab1, tab2, tab3 = st.tabs(["Patient Information", "X-ray Analysis", "Symptom Assessment"])
    
    with tab1:
        collect_patient_info()
    
    with tab2:
        xray_analysis_section()
    
    with tab3:
        symptom_assessment_section()
    
    # Analysis button
    if st.button("🔍 Perform Complete Analysis", type="primary", use_container_width=True):
        if validate_complete_assessment():
            perform_complete_analysis()
        else:
            st.error("Please complete all required sections before analysis.")

def collect_patient_info():
    st.subheader("Patient Demographics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.patient_data['age'] = st.number_input(
            "Age", min_value=0, max_value=120, value=30
        )
        st.session_state.patient_data['gender'] = st.selectbox(
            "Gender", ["Male", "Female", "Other"]
        )
        st.session_state.patient_data['weight'] = st.number_input(
            "Weight (kg)", min_value=0.0, max_value=300.0, value=70.0
        )
    
    with col2:
        st.session_state.patient_data['height'] = st.number_input(
            "Height (cm)", min_value=0.0, max_value=250.0, value=170.0
        )
        st.session_state.patient_data['smoking_history'] = st.selectbox(
            "Smoking History", ["Never", "Former", "Current"]
        )
        st.session_state.patient_data['hiv_status'] = st.selectbox(
            "HIV Status", ["Unknown", "Negative", "Positive"]
        )

def xray_analysis_section():
    st.subheader("Chest X-ray Upload and Analysis")
    
    uploaded_file = st.file_uploader(
        "Upload Chest X-ray Image",
        type=['png', 'jpg', 'jpeg', 'tiff'],
        help="Please upload a clear chest X-ray image in PNG, JPG, JPEG, or TIFF format"
    )
    
    if uploaded_file is not None:
        try:
            # Display uploaded image
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.image(image, caption="Uploaded Chest X-ray", use_column_width=True)
            
            with col2:
                st.info("Image Details:")
                st.write(f"**Size:** {image.size}")
                st.write(f"**Format:** {image.format}")
                st.write(f"**Mode:** {image.mode}")
            
            # Process image for analysis
            processor = ImageProcessor()
            processed_image = processor.preprocess_image(image)
            
            st.session_state.processed_xray = processed_image
            st.success("✅ Image uploaded and preprocessed successfully!")
            
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

def symptom_assessment_section():
    st.subheader("TB Symptom Assessment")
    
    symptom_responses = {}
    
    # Primary symptoms
    st.write("**Primary Symptoms** (Select all that apply)")
    primary_cols = st.columns(2)
    
    for i, (symptom, details) in enumerate(TB_SYMPTOMS['primary'].items()):
        col = primary_cols[i % 2]
        with col:
            symptom_responses[symptom] = st.checkbox(
                f"{symptom.replace('_', ' ').title()}",
                help=details['description']
            )
    
    # Secondary symptoms
    st.write("**Secondary Symptoms** (Select all that apply)")
    secondary_cols = st.columns(3)
    
    for i, (symptom, details) in enumerate(TB_SYMPTOMS['secondary'].items()):
        col = secondary_cols[i % 3]
        with col:
            symptom_responses[symptom] = st.checkbox(
                f"{symptom.replace('_', ' ').title()}",
                help=details['description']
            )
    
    # Duration and severity
    st.write("**Symptom Details**")
    duration_col, severity_col = st.columns(2)
    
    with duration_col:
        symptom_responses['symptom_duration'] = st.selectbox(
            "Duration of symptoms",
            ["Less than 2 weeks", "2-4 weeks", "1-3 months", "More than 3 months"]
        )
    
    with severity_col:
        symptom_responses['symptom_severity'] = st.slider(
            "Overall symptom severity (1-10)", 1, 10, 5
        )
    
    st.session_state.symptom_responses = symptom_responses

def xray_analysis_only():
    st.header("X-ray Analysis Only")
    
    xray_analysis_section()
    
    if st.button("🔍 Analyze X-ray", type="primary") and 'processed_xray' in st.session_state:
        perform_xray_analysis()

def symptom_assessment_only():
    st.header("Symptom Assessment Only")
    
    symptom_assessment_section()
    
    if st.button("🔍 Assess Symptoms", type="primary") and 'symptom_responses' in st.session_state:
        perform_symptom_analysis()

def validate_complete_assessment():
    return (
        'processed_xray' in st.session_state and
        'symptom_responses' in st.session_state and
        len(st.session_state.patient_data) > 0
    )

def perform_xray_analysis():
    with st.spinner("Analyzing X-ray image..."):
        try:
            analyzer = XRayAnalyzer()
            result = analyzer.predict(st.session_state.processed_xray)
            st.session_state.xray_result = result
            
            display_xray_results(result)
            
        except Exception as e:
            st.error(f"Error during X-ray analysis: {str(e)}")

def perform_symptom_analysis():
    with st.spinner("Assessing symptoms..."):
        try:
            assessor = SymptomAssessment()
            result = assessor.calculate_risk_score(
                st.session_state.symptom_responses,
                st.session_state.patient_data
            )
            st.session_state.symptom_result = result
            
            display_symptom_results(result)
            
        except Exception as e:
            st.error(f"Error during symptom analysis: {str(e)}")

def perform_complete_analysis():
    with st.spinner("Performing comprehensive TB analysis..."):
        try:
            # X-ray analysis
            analyzer = XRayAnalyzer()
            xray_result = analyzer.predict(st.session_state.processed_xray)
            
            # Symptom assessment
            assessor = SymptomAssessment()
            symptom_result = assessor.calculate_risk_score(
                st.session_state.symptom_responses,
                st.session_state.patient_data
            )
            
            # Combined analysis
            processor = ResultProcessor()
            combined_result = processor.combine_results(xray_result, symptom_result)
            
            st.session_state.xray_result = xray_result
            st.session_state.symptom_result = symptom_result
            st.session_state.combined_result = combined_result
            st.session_state.analysis_complete = True
            
            display_combined_results(combined_result)
            
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")

def display_xray_results(result):
    st.success("X-ray Analysis Complete!")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Confidence gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = result['tb_probability'] * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "TB Detection Confidence (%)"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.metric("TB Probability", f"{result['tb_probability']:.2%}")
        st.metric("Confidence Score", f"{result['confidence']:.2f}")
        
        if result['tb_probability'] > 0.7:
            st.error("⚠️ HIGH RISK: Immediate medical attention recommended")
        elif result['tb_probability'] > 0.4:
            st.warning("⚠️ MODERATE RISK: Further testing recommended")
        else:
            st.success("✅ LOW RISK: Routine monitoring")

def display_symptom_results(result):
    st.success("Symptom Assessment Complete!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Risk Score", f"{result['risk_score']:.1f}/100")
        st.metric("Primary Symptoms", result['primary_symptom_count'])
        st.metric("Secondary Symptoms", result['secondary_symptom_count'])
    
    with col2:
        risk_level = result['risk_level']
        if risk_level == "HIGH":
            st.error(f"⚠️ {risk_level} RISK")
        elif risk_level == "MODERATE":
            st.warning(f"⚠️ {risk_level} RISK")
        else:
            st.success(f"✅ {risk_level} RISK")
        
        st.write("**Key Symptoms Identified:**")
        for symptom in result['identified_symptoms']:
            st.write(f"• {symptom.replace('_', ' ').title()}")

def display_combined_results(result):
    st.success("🎉 Complete Analysis Finished!")
    
    # Main results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Combined TB Risk", f"{result['combined_probability']:.2%}")
    with col2:
        st.metric("X-ray Confidence", f"{result['xray_confidence']:.2%}")
    with col3:
        st.metric("Symptom Score", f"{result['symptom_score']:.1f}/100")
    
    # Risk assessment
    if result['combined_probability'] > 0.7:
        st.error("🚨 **HIGH RISK**: Immediate medical consultation and TB testing strongly recommended")
    elif result['combined_probability'] > 0.4:
        st.warning("⚠️ **MODERATE RISK**: Medical evaluation and further testing recommended")
    else:
        st.success("✅ **LOW RISK**: Continue routine health monitoring")
    
    # Detailed breakdown
    st.subheader("Analysis Breakdown")
    
    breakdown_data = {
        'Analysis Type': ['X-ray Detection', 'Symptom Assessment', 'Combined Score'],
        'Probability': [result['xray_probability'], result['symptom_probability'], result['combined_probability']],
        'Confidence': [result['xray_confidence'], result['symptom_confidence'], result['combined_confidence']]
    }
    
    df = pd.DataFrame(breakdown_data)
    
    fig = px.bar(df, x='Analysis Type', y='Probability', 
                 title='TB Detection Results Comparison',
                 color='Confidence', color_continuous_scale='RdYlBu_r')
    st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    st.subheader("Medical Recommendations")
    for recommendation in result['recommendations']:
        st.write(f"• {recommendation}")

def results_dashboard():
    st.header("Results Dashboard")
    
    if not st.session_state.analysis_complete:
        st.info("No analysis results available. Please complete an assessment first.")
        return
    
    # Display all results
    if 'combined_result' in st.session_state:
        st.subheader("Complete Assessment Results")
        display_combined_results(st.session_state.combined_result)
    
    if 'xray_result' in st.session_state:
        st.subheader("X-ray Analysis Results")
        display_xray_results(st.session_state.xray_result)
    
    if 'symptom_result' in st.session_state:
        st.subheader("Symptom Assessment Results")
        display_symptom_results(st.session_state.symptom_result)
    
    # Export results
    if st.button("📊 Export Results Report"):
        generate_report()

def generate_report():
    st.info("Report generation feature will be implemented based on specific requirements.")

if __name__ == "__main__":
    main()
