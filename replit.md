# TB Detection System

## Overview

This is an AI-powered tuberculosis detection system that combines X-ray image analysis with symptom assessment to provide comprehensive TB screening. The system uses deep learning models to analyze chest X-ray images and correlates findings with patient symptoms and demographic risk factors to generate risk assessments for healthcare professionals.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Streamlit Web Application**: Multi-page interface with sidebar navigation supporting complete assessment, X-ray-only analysis, symptom-only assessment, and results dashboard
- **Session State Management**: Persistent storage of patient data, analysis results, and application state across user interactions
- **Interactive Visualizations**: Plotly integration for generating charts and graphs to display analysis results

### Backend Architecture
- **Modular Design**: Separation of concerns with distinct modules for X-ray analysis, symptom assessment, image processing, and result combination
- **TensorFlow/Keras CNN Model**: Deep learning architecture for X-ray image classification with convolutional layers, max pooling, dropout for regularization, and softmax output for TB/Normal classification
- **Symptom Scoring Engine**: Weighted assessment system that evaluates primary symptoms (cough, hemoptysis, chest pain), secondary symptoms, duration factors, and demographic risk factors

### Data Processing Pipeline
- **Image Preprocessing**: Standardized pipeline for X-ray images including RGB conversion, resizing to 224x224, noise reduction, and normalization
- **Symptom Evaluation**: Structured scoring system with predefined weights for different symptoms and risk factors
- **Result Combination**: Weighted fusion of X-ray analysis (60%) and symptom assessment (40%) with confidence calculations

### Core Components
- **XRayAnalyzer**: Handles model loading, image preprocessing, and TB probability prediction
- **SymptomAssessment**: Processes patient symptoms and demographic data to calculate risk scores
- **ImageProcessor**: Standardizes X-ray image format and quality for model input
- **ResultProcessor**: Combines multiple analysis streams into comprehensive assessments

### Medical Data Management
- **TB Symptoms Database**: Comprehensive catalog of primary and secondary symptoms with severity weights and specificity ratings
- **Demographic Risk Factors**: Age-based, health condition, and lifestyle factors that influence TB susceptibility
- **Risk Stratification**: Multi-level classification system (low, moderate, high) based on combined analysis

## External Dependencies

### Machine Learning Frameworks
- **TensorFlow**: Deep learning model development and inference for X-ray image analysis
- **OpenCV**: Computer vision operations for image preprocessing and manipulation
- **NumPy**: Numerical computing for array operations and mathematical calculations
- **Pandas**: Data manipulation and analysis for patient records and results

### Web Application Framework
- **Streamlit**: Frontend framework for creating the interactive medical assessment interface
- **Plotly**: Data visualization library for creating interactive charts and medical result displays
- **PIL (Python Imaging Library)**: Image processing and format conversion utilities

### Supporting Libraries
- **datetime**: Timestamp management for analysis records and patient data tracking

Note: The system is designed as a standalone application with potential for integration with hospital information systems and medical databases. The current architecture supports future expansion to include electronic health record integration and cloud-based model deployment.