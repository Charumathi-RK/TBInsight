# test_tb_assessment.py

import cv2
import numpy as np
from data.tb_symptoms import TB_SYMPTOMS, DEMOGRAPHIC_FACTORS
from models.symptom_assessment import SymptomAssessment
from models.xray_model import XRayAnalyzer


# ---------- Sample Symptom Data ----------
sample_symptoms = {
    'persistent_cough': True,
    'blood_in_sputum': False,
    'chest_pain': True,
    'difficulty_breathing': False,
    'fever': True,
    'night_sweats': True,
    'weight_loss': False,
    'fatigue': True,
    'loss_of_appetite': True,
    'chills': False,
    'swollen_lymph_nodes': False,
    'back_pain': False,
    'headache': True,
    'nausea': False,
    'abdominal_pain': False,
    'symptom_duration': '1-3 months',
    'symptom_severity': 7
}

# ---------- Sample Patient Data ----------
sample_patient = {
    'age': 70,
    'weight': 60,
    'height': 165,
    'hiv_status': 'Negative',
    'smoking_history': 'Former'
}

# ---------- Symptom Assessment ----------
assessment = SymptomAssessment()
risk_result = assessment.calculate_risk_score(sample_symptoms, sample_patient)

print("\n--- TB Symptom Assessment ---")
for k, v in risk_result.items():
    print(f"{k}: {v}")

# ---------- X-ray Prediction ----------
# Load a sample X-ray image (replace path with your image)
sample_image_path = "sample.png"
image = cv2.imread(sample_image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert to RGB

# Resize to model input shape
image = cv2.resize(image, (224, 224)) / 255.0  # normalize 0-1

xray_analyzer = XRayAnalyzer()
xray_result = xray_analyzer.predict(image)

print("\n--- X-ray Analysis ---")
for k, v in xray_result.items():
    print(f"{k}: {v}")
