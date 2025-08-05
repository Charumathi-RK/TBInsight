"""
TB Symptoms and Demographic Factors Data
Contains comprehensive information about tuberculosis symptoms and risk factors.
"""

TB_SYMPTOMS = {
    'primary': {
        'persistent_cough': {
            'description': 'Cough lasting more than 2-3 weeks',
            'weight': 15,
            'severity': 'high',
            'specificity': 'moderate'
        },
        'blood_in_sputum': {
            'description': 'Coughing up blood or blood-tinged sputum (hemoptysis)',
            'weight': 20,
            'severity': 'very_high',
            'specificity': 'high'
        },
        'chest_pain': {
            'description': 'Persistent chest pain, especially with breathing or coughing',
            'weight': 10,
            'severity': 'moderate',
            'specificity': 'low'
        },
        'difficulty_breathing': {
            'description': 'Shortness of breath or difficulty breathing',
            'weight': 12,
            'severity': 'high',
            'specificity': 'moderate'
        },
        'fever': {
            'description': 'Prolonged fever, often low-grade and persistent',
            'weight': 8,
            'severity': 'moderate',
            'specificity': 'low'
        },
        'night_sweats': {
            'description': 'Profuse sweating during sleep',
            'weight': 8,
            'severity': 'moderate',
            'specificity': 'moderate'
        },
        'weight_loss': {
            'description': 'Unexplained weight loss over weeks or months',
            'weight': 12,
            'severity': 'high',
            'specificity': 'moderate'
        },
        'fatigue': {
            'description': 'Persistent tiredness and weakness',
            'weight': 6,
            'severity': 'moderate',
            'specificity': 'low'
        }
    },
    'secondary': {
        'loss_of_appetite': {
            'description': 'Decreased desire to eat',
            'weight': 4,
            'severity': 'low',
            'specificity': 'low'
        },
        'chills': {
            'description': 'Feeling cold and shivering',
            'weight': 3,
            'severity': 'low',
            'specificity': 'low'
        },
        'swollen_lymph_nodes': {
            'description': 'Enlarged lymph nodes, especially in neck or armpits',
            'weight': 6,
            'severity': 'moderate',
            'specificity': 'moderate'
        },
        'back_pain': {
            'description': 'Back pain, especially if TB affects the spine',
            'weight': 2,
            'severity': 'low',
            'specificity': 'low'
        },
        'headache': {
            'description': 'Persistent or recurring headaches',
            'weight': 1,
            'severity': 'low',
            'specificity': 'very_low'
        },
        'nausea': {
            'description': 'Feeling sick to the stomach',
            'weight': 1,
            'severity': 'low',
            'specificity': 'very_low'
        },
        'abdominal_pain': {
            'description': 'Stomach or abdominal discomfort',
            'weight': 2,
            'severity': 'low',
            'specificity': 'very_low'
        }
    }
}

DEMOGRAPHIC_FACTORS = {
    'age_risk': {
        'infants': {
            'age_range': '0-5',
            'risk_multiplier': 1.3,
            'description': 'Children under 5 have higher risk due to developing immune systems'
        },
        'elderly': {
            'age_range': '65+',
            'risk_multiplier': 1.2,
            'description': 'Adults over 65 have increased risk due to weakened immune systems'
        }
    },
    'medical_conditions': {
        'hiv_positive': {
            'risk_multiplier': 1.8,
            'description': 'HIV infection significantly increases TB risk'
        },
        'diabetes': {
            'risk_multiplier': 1.3,
            'description': 'Diabetes mellitus increases TB susceptibility'
        },
        'immunosuppression': {
            'risk_multiplier': 1.5,
            'description': 'Immunosuppressive medications or conditions'
        },
        'chronic_kidney_disease': {
            'risk_multiplier': 1.3,
            'description': 'Chronic kidney disease increases TB risk'
        },
        'cancer': {
            'risk_multiplier': 1.4,
            'description': 'Cancer, especially hematologic malignancies'
        }
    },
    'lifestyle_factors': {
        'smoking_current': {
            'risk_multiplier': 1.4,
            'description': 'Current smoking significantly increases TB risk'
        },
        'smoking_former': {
            'risk_multiplier': 1.1,
            'description': 'Former smoking slightly increases TB risk'
        },
        'alcohol_abuse': {
            'risk_multiplier': 1.3,
            'description': 'Alcohol abuse increases TB susceptibility'
        },
        'drug_use': {
            'risk_multiplier': 1.3,
            'description': 'Intravenous drug use increases TB risk'
        },
        'malnutrition': {
            'risk_multiplier': 1.3,
            'description': 'Malnutrition or underweight status'
        }
    },
    'exposure_factors': {
        'close_contact': {
            'risk_multiplier': 2.0,
            'description': 'Close contact with active TB case'
        },
        'healthcare_worker': {
            'risk_multiplier': 1.2,
            'description': 'Healthcare workers have occupational exposure'
        },
        'congregate_settings': {
            'risk_multiplier': 1.3,
            'description': 'Living in congregate settings (prisons, shelters, etc.)'
        },
        'high_endemic_area': {
            'risk_multiplier': 1.2,
            'description': 'Travel or residence in high TB endemic areas'
        }
    },
    'socioeconomic_factors': {
        'poverty': {
            'risk_multiplier': 1.3,
            'description': 'Poverty and poor living conditions'
        },
        'homelessness': {
            'risk_multiplier': 1.5,
            'description': 'Homelessness increases TB risk'
        },
        'overcrowding': {
            'risk_multiplier': 1.2,
            'description': 'Overcrowded living conditions'
        }
    }
}

SYMPTOM_CATEGORIES = {
    'constitutional': [
        'fever', 'night_sweats', 'weight_loss', 'fatigue', 
        'loss_of_appetite', 'chills'
    ],
    'pulmonary': [
        'persistent_cough', 'blood_in_sputum', 'chest_pain', 
        'difficulty_breathing'
    ],
    'systemic': [
        'swollen_lymph_nodes', 'back_pain', 'headache', 
        'nausea', 'abdominal_pain'
    ]
}

DURATION_SIGNIFICANCE = {
    'acute': {
        'duration': 'Less than 2 weeks',
        'tb_likelihood': 'low',
        'multiplier': 1.0,
        'note': 'Most TB symptoms are chronic in nature'
    },
    'subacute': {
        'duration': '2-4 weeks',
        'tb_likelihood': 'moderate',
        'multiplier': 1.3,
        'note': 'Symptoms persisting 2-4 weeks warrant evaluation'
    },
    'chronic_short': {
        'duration': '1-3 months',
        'tb_likelihood': 'high',
        'multiplier': 1.6,
        'note': 'Chronic symptoms are characteristic of TB'
    },
    'chronic_long': {
        'duration': 'More than 3 months',
        'tb_likelihood': 'very_high',
        'multiplier': 2.0,
        'note': 'Prolonged symptoms strongly suggest TB or other chronic disease'
    }
}

SEVERITY_IMPACT = {
    'mild': {
        'severity_range': '1-3',
        'impact_multiplier': 0.7,
        'description': 'Mild symptoms may not significantly impact daily activities'
    },
    'moderate': {
        'severity_range': '4-6',
        'impact_multiplier': 1.0,
        'description': 'Moderate symptoms affect daily functioning'
    },
    'severe': {
        'severity_range': '7-8',
        'impact_multiplier': 1.3,
        'description': 'Severe symptoms significantly impair quality of life'
    },
    'critical': {
        'severity_range': '9-10',
        'impact_multiplier': 1.5,
        'description': 'Critical symptoms require immediate medical attention'
    }
}

# Clinical decision support rules
CLINICAL_RULES = {
    'immediate_referral': [
        'blood_in_sputum',
        'severe_difficulty_breathing',
        'high_fever_with_multiple_symptoms'
    ],
    'urgent_evaluation': [
        'multiple_primary_symptoms',
        'chronic_cough_with_weight_loss',
        'night_sweats_with_fever'
    ],
    'routine_monitoring': [
        'single_mild_symptom',
        'recent_onset_symptoms'
    ]
}

def get_symptom_by_category(category):
    """Get symptoms by category."""
    if category not in SYMPTOM_CATEGORIES:
        return []
    
    symptom_names = SYMPTOM_CATEGORIES[category]
    symptoms = {}
    
    # Get from primary symptoms
    for symptom in symptom_names:
        if symptom in TB_SYMPTOMS['primary']:
            symptoms[symptom] = TB_SYMPTOMS['primary'][symptom]
        elif symptom in TB_SYMPTOMS['secondary']:
            symptoms[symptom] = TB_SYMPTOMS['secondary'][symptom]
    
    return symptoms

def calculate_bmi_risk(height_cm, weight_kg):
    """Calculate BMI and associated TB risk."""
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    
    if bmi < 16:
        return {'bmi': bmi, 'category': 'severely_underweight', 'risk_multiplier': 1.5}
    elif bmi < 18.5:
        return {'bmi': bmi, 'category': 'underweight', 'risk_multiplier': 1.3}
    elif bmi < 25:
        return {'bmi': bmi, 'category': 'normal', 'risk_multiplier': 1.0}
    elif bmi < 30:
        return {'bmi': bmi, 'category': 'overweight', 'risk_multiplier': 1.0}
    else:
        return {'bmi': bmi, 'category': 'obese', 'risk_multiplier': 1.1}

def get_risk_assessment_guidelines():
    """Get comprehensive risk assessment guidelines."""
    return {
        'symptom_weights': TB_SYMPTOMS,
        'demographic_factors': DEMOGRAPHIC_FACTORS,
        'duration_significance': DURATION_SIGNIFICANCE,
        'severity_impact': SEVERITY_IMPACT,
        'clinical_rules': CLINICAL_RULES,
        'assessment_notes': [
            'TB symptoms are often chronic and progressive',
            'Constitutional symptoms (fever, night sweats, weight loss) are common',
            'Pulmonary symptoms may be absent in extrapulmonary TB',
            'Risk factors significantly modify probability of TB',
            'Clinical correlation with imaging and laboratory tests is essential'
        ]
    }
