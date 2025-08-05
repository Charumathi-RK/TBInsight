import numpy as np
from datetime import datetime
from data.tb_symptoms import TB_SYMPTOMS, DEMOGRAPHIC_FACTORS

class SymptomAssessment:
    def __init__(self):
        """Initialize the symptom assessment system."""
        self.weights = self._initialize_weights()
        self.risk_thresholds = {
            'low': 30,
            'moderate': 60,
            'high': 80
        }
    
    def _initialize_weights(self):
        """Initialize scoring weights for different symptoms and factors."""
        return {
            'primary_symptoms': {
                'persistent_cough': 15,
                'blood_in_sputum': 20,
                'chest_pain': 10,
                'difficulty_breathing': 12,
                'fever': 8,
                'night_sweats': 8,
                'weight_loss': 12,
                'fatigue': 6
            },
            'secondary_symptoms': {
                'loss_of_appetite': 4,
                'chills': 3,
                'swollen_lymph_nodes': 6,
                'back_pain': 2,
                'headache': 1,
                'nausea': 1,
                'abdominal_pain': 2
            },
            'duration_multipliers': {
                'Less than 2 weeks': 1.0,
                '2-4 weeks': 1.3,
                '1-3 months': 1.6,
                'More than 3 months': 2.0
            },
            'demographic_factors': {
                'age_elderly': 1.2,  # Age > 65
                'age_infant': 1.3,   # Age < 5
                'hiv_positive': 1.8,
                'underweight': 1.3,
                'smoking_current': 1.4,
                'smoking_former': 1.1
            }
        }
    
    def calculate_risk_score(self, symptom_responses, patient_data):
        """Calculate comprehensive TB risk score."""
        try:
            # Base symptom score
            symptom_score = self._calculate_symptom_score(symptom_responses)
            
            # Apply demographic risk factors
            demographic_multiplier = self._calculate_demographic_multiplier(patient_data)
            
            # Apply duration multiplier
            duration_multiplier = self._get_duration_multiplier(symptom_responses)
            
            # Apply severity factor
            severity_factor = self._get_severity_factor(symptom_responses)
            
            # Calculate final risk score
            raw_score = symptom_score * demographic_multiplier * duration_multiplier * severity_factor
            
            # Normalize to 0-100 scale
            risk_score = min(100, raw_score)
            
            # Determine risk level
            risk_level = self._determine_risk_level(risk_score)
            
            # Get additional insights
            insights = self._generate_insights(symptom_responses, patient_data, risk_score)
            
            return {
                'risk_score': risk_score,
                'risk_level': risk_level,
                'symptom_score': symptom_score,
                'demographic_multiplier': demographic_multiplier,
                'duration_multiplier': duration_multiplier,
                'severity_factor': severity_factor,
                'primary_symptom_count': self._count_primary_symptoms(symptom_responses),
                'secondary_symptom_count': self._count_secondary_symptoms(symptom_responses),
                'identified_symptoms': self._get_identified_symptoms(symptom_responses),
                'risk_factors': self._identify_risk_factors(patient_data),
                'insights': insights,
                'recommendations': self._generate_recommendations(risk_level, symptom_responses)
            }
            
        except Exception as e:
            raise Exception(f"Risk calculation failed: {str(e)}")
    
    def _calculate_symptom_score(self, responses):
        """Calculate base symptom score."""
        score = 0
        
        # Primary symptoms
        for symptom, weight in self.weights['primary_symptoms'].items():
            if responses.get(symptom, False):
                score += weight
        
        # Secondary symptoms
        for symptom, weight in self.weights['secondary_symptoms'].items():
            if responses.get(symptom, False):
                score += weight
        
        return score
    
    def _calculate_demographic_multiplier(self, patient_data):
        """Calculate demographic risk multiplier."""
        multiplier = 1.0
        
        age = patient_data.get('age', 30)
        
        # Age factors
        if age > 65:
            multiplier *= self.weights['demographic_factors']['age_elderly']
        elif age < 5:
            multiplier *= self.weights['demographic_factors']['age_infant']
        
        # HIV status
        if patient_data.get('hiv_status') == 'Positive':
            multiplier *= self.weights['demographic_factors']['hiv_positive']
        
        # BMI calculation and underweight factor
        height_m = patient_data.get('height', 170) / 100
        weight_kg = patient_data.get('weight', 70)
        bmi = weight_kg / (height_m ** 2)
        
        if bmi < 18.5:  # Underweight
            multiplier *= self.weights['demographic_factors']['underweight']
        
        # Smoking history
        smoking = patient_data.get('smoking_history', 'Never')
        if smoking == 'Current':
            multiplier *= self.weights['demographic_factors']['smoking_current']
        elif smoking == 'Former':
            multiplier *= self.weights['demographic_factors']['smoking_former']
        
        return multiplier
    
    def _get_duration_multiplier(self, responses):
        """Get duration-based risk multiplier."""
        duration = responses.get('symptom_duration', 'Less than 2 weeks')
        return self.weights['duration_multipliers'].get(duration, 1.0)
    
    def _get_severity_factor(self, responses):
        """Get severity-based risk factor."""
        severity = responses.get('symptom_severity', 5)
        # Scale severity (1-10) to factor (0.5-1.5)
        return 0.5 + (severity / 10)
    
    def _determine_risk_level(self, score):
        """Determine risk level based on score."""
        if score >= self.risk_thresholds['high']:
            return "HIGH"
        elif score >= self.risk_thresholds['moderate']:
            return "MODERATE"
        else:
            return "LOW"
    
    def _count_primary_symptoms(self, responses):
        """Count number of primary symptoms present."""
        count = 0
        for symptom in self.weights['primary_symptoms'].keys():
            if responses.get(symptom, False):
                count += 1
        return count
    
    def _count_secondary_symptoms(self, responses):
        """Count number of secondary symptoms present."""
        count = 0
        for symptom in self.weights['secondary_symptoms'].keys():
            if responses.get(symptom, False):
                count += 1
        return count
    
    def _get_identified_symptoms(self, responses):
        """Get list of identified symptoms."""
        symptoms = []
        
        # Check primary symptoms
        for symptom in self.weights['primary_symptoms'].keys():
            if responses.get(symptom, False):
                symptoms.append(symptom)
        
        # Check secondary symptoms
        for symptom in self.weights['secondary_symptoms'].keys():
            if responses.get(symptom, False):
                symptoms.append(symptom)
        
        return symptoms
    
    def _identify_risk_factors(self, patient_data):
        """Identify demographic risk factors."""
        risk_factors = []
        
        age = patient_data.get('age', 30)
        if age > 65:
            risk_factors.append("Elderly (>65 years)")
        elif age < 5:
            risk_factors.append("Infant (<5 years)")
        
        if patient_data.get('hiv_status') == 'Positive':
            risk_factors.append("HIV Positive")
        
        # BMI calculation
        height_m = patient_data.get('height', 170) / 100
        weight_kg = patient_data.get('weight', 70)
        bmi = weight_kg / (height_m ** 2)
        
        if bmi < 18.5:
            risk_factors.append("Underweight (BMI < 18.5)")
        
        smoking = patient_data.get('smoking_history', 'Never')
        if smoking == 'Current':
            risk_factors.append("Current smoker")
        elif smoking == 'Former':
            risk_factors.append("Former smoker")
        
        return risk_factors
    
    def _generate_insights(self, symptom_responses, patient_data, risk_score):
        """Generate insights based on assessment."""
        insights = []
        
        # Primary symptom insights
        primary_count = self._count_primary_symptoms(symptom_responses)
        if primary_count >= 3:
            insights.append(f"Multiple primary TB symptoms present ({primary_count})")
        
        # Duration insight
        duration = symptom_responses.get('symptom_duration', 'Less than 2 weeks')
        if duration in ['1-3 months', 'More than 3 months']:
            insights.append("Prolonged symptom duration increases TB likelihood")
        
        # High-risk symptoms
        if symptom_responses.get('blood_in_sputum', False):
            insights.append("Hemoptysis (blood in sputum) is a significant TB indicator")
        
        if symptom_responses.get('night_sweats', False) and symptom_responses.get('fever', False):
            insights.append("Combination of fever and night sweats is concerning for TB")
        
        # Risk factor insights
        if patient_data.get('hiv_status') == 'Positive':
            insights.append("HIV co-infection significantly increases TB risk")
        
        return insights
    
    def _generate_recommendations(self, risk_level, symptom_responses):
        """Generate medical recommendations based on risk level."""
        recommendations = []
        
        if risk_level == "HIGH":
            recommendations.extend([
                "Immediate medical consultation required",
                "TB skin test (TST) or Interferon-Gamma Release Assay (IGRA)",
                "Chest X-ray examination",
                "Sputum collection for acid-fast bacilli (AFB) testing",
                "Consider isolation precautions until TB is ruled out"
            ])
        elif risk_level == "MODERATE":
            recommendations.extend([
                "Medical evaluation within 1-2 weeks",
                "Consider TB screening tests",
                "Monitor symptoms closely",
                "Chest X-ray if symptoms persist or worsen"
            ])
        else:  # LOW risk
            recommendations.extend([
                "Continue monitoring symptoms",
                "Seek medical attention if symptoms worsen or persist",
                "Maintain good general health practices"
            ])
        
        # Specific symptom-based recommendations
        if symptom_responses.get('blood_in_sputum', False):
            recommendations.append("Blood in sputum requires immediate medical attention")
        
        if symptom_responses.get('persistent_cough', False):
            recommendations.append("Persistent cough should be evaluated for TB")
        
        return recommendations
    
    def get_assessment_summary(self):
        """Return summary of assessment methodology."""
        return {
            'assessment_type': 'Multi-factor TB Risk Assessment',
            'factors_considered': [
                'Primary TB symptoms',
                'Secondary TB symptoms',
                'Symptom duration',
                'Symptom severity',
                'Age factors',
                'HIV status',
                'BMI/nutritional status',
                'Smoking history'
            ],
            'scoring_range': '0-100',
            'risk_levels': list(self.risk_thresholds.keys())
        }
