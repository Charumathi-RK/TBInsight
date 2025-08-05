import numpy as np
from datetime import datetime

class ResultProcessor:
    def __init__(self):
        """Initialize result processor with combination weights."""
        self.combination_weights = {
            'xray_weight': 0.6,      # X-ray analysis weight
            'symptom_weight': 0.4     # Symptom assessment weight
        }
        
        self.confidence_thresholds = {
            'high': 0.8,
            'moderate': 0.6,
            'low': 0.4
        }
    
    def combine_results(self, xray_result, symptom_result):
        """
        Combine X-ray and symptom analysis results into a comprehensive assessment.
        
        Args:
            xray_result: Dictionary containing X-ray analysis results
            symptom_result: Dictionary containing symptom assessment results
            
        Returns:
            Dictionary containing combined analysis results
        """
        try:
            # Extract probabilities
            xray_prob = xray_result['tb_probability']
            xray_confidence = xray_result['confidence']
            
            # Convert symptom risk score to probability (0-1 scale)
            symptom_prob = min(1.0, symptom_result['risk_score'] / 100.0)
            symptom_confidence = self._calculate_symptom_confidence(symptom_result)
            
            # Calculate weighted combined probability
            combined_prob = (
                xray_prob * self.combination_weights['xray_weight'] +
                symptom_prob * self.combination_weights['symptom_weight']
            )
            
            # Calculate combined confidence
            combined_confidence = (
                xray_confidence * self.combination_weights['xray_weight'] +
                symptom_confidence * self.combination_weights['symptom_weight']
            )
            
            # Determine overall risk level
            risk_level = self._determine_combined_risk_level(combined_prob, combined_confidence)
            
            # Generate comprehensive recommendations
            recommendations = self._generate_combined_recommendations(
                xray_result, symptom_result, combined_prob, risk_level
            )
            
            # Calculate consistency score
            consistency_score = self._calculate_consistency(xray_prob, symptom_prob)
            
            # Generate detailed analysis
            detailed_analysis = self._generate_detailed_analysis(
                xray_result, symptom_result, combined_prob, consistency_score
            )
            
            return {
                'combined_probability': combined_prob,
                'combined_confidence': combined_confidence,
                'risk_level': risk_level,
                'xray_probability': xray_prob,
                'xray_confidence': xray_confidence,
                'symptom_probability': symptom_prob,
                'symptom_confidence': symptom_confidence,
                'symptom_score': symptom_result['risk_score'],
                'consistency_score': consistency_score,
                'recommendations': recommendations,
                'detailed_analysis': detailed_analysis,
                'analysis_timestamp': datetime.now().isoformat(),
                'method_weights': self.combination_weights
            }
            
        except Exception as e:
            raise Exception(f"Result combination failed: {str(e)}")
    
    def _calculate_symptom_confidence(self, symptom_result):
        """Calculate confidence score for symptom assessment."""
        # Base confidence on number of symptoms and risk factors
        primary_count = symptom_result['primary_symptom_count']
        secondary_count = symptom_result['secondary_symptom_count']
        risk_factors = len(symptom_result.get('risk_factors', []))
        
        # Higher symptom counts and risk factors increase confidence
        symptom_confidence = min(1.0, (primary_count * 0.15 + secondary_count * 0.05 + risk_factors * 0.1))
        
        # Adjust based on risk score
        risk_score = symptom_result['risk_score']
        if risk_score > 70:
            symptom_confidence = max(symptom_confidence, 0.8)
        elif risk_score > 40:
            symptom_confidence = max(symptom_confidence, 0.6)
        else:
            symptom_confidence = max(symptom_confidence, 0.4)
        
        return symptom_confidence
    
    def _determine_combined_risk_level(self, combined_prob, combined_confidence):
        """Determine overall risk level based on combined probability and confidence."""
        if combined_prob >= 0.7 and combined_confidence >= self.confidence_thresholds['moderate']:
            return "HIGH"
        elif combined_prob >= 0.4 and combined_confidence >= self.confidence_thresholds['low']:
            return "MODERATE"
        else:
            return "LOW"
    
    def _calculate_consistency(self, xray_prob, symptom_prob):
        """Calculate consistency between X-ray and symptom analysis."""
        # Calculate absolute difference
        difference = abs(xray_prob - symptom_prob)
        
        # Convert to consistency score (0-1, where 1 is perfectly consistent)
        consistency = 1.0 - min(1.0, difference)
        
        return consistency
    
    def _generate_combined_recommendations(self, xray_result, symptom_result, combined_prob, risk_level):
        """Generate comprehensive recommendations based on combined analysis."""
        recommendations = []
        
        # Primary recommendations based on risk level
        if risk_level == "HIGH":
            recommendations.extend([
                "🚨 URGENT: Immediate medical attention required",
                "Comprehensive TB testing including sputum examination",
                "Consider hospital referral for specialized care",
                "Implement isolation precautions if TB suspected",
                "Begin contact tracing if TB confirmed"
            ])
        elif risk_level == "MODERATE":
            recommendations.extend([
                "⚠️ Medical evaluation recommended within 48-72 hours",
                "TB screening tests (TST or IGRA) recommended",
                "Follow-up chest imaging if not recently performed",
                "Monitor symptoms closely for any worsening"
            ])
        else:
            recommendations.extend([
                "✅ Continue routine health monitoring",
                "Return for evaluation if symptoms worsen or persist",
                "Consider TB testing if risk factors are present"
            ])
        
        # Specific recommendations based on X-ray findings
        if xray_result['tb_probability'] > 0.7:
            recommendations.append("High-probability TB findings on imaging require immediate follow-up")
        
        # Specific recommendations based on symptoms
        if symptom_result['risk_score'] > 70:
            recommendations.append("Multiple TB symptoms present - urgent clinical correlation needed")
        
        # Consistency-based recommendations
        consistency = self._calculate_consistency(
            xray_result['tb_probability'], 
            min(1.0, symptom_result['risk_score'] / 100.0)
        )
        
        if consistency < 0.6:
            recommendations.append("Inconsistent findings between imaging and symptoms - comprehensive evaluation needed")
        
        return recommendations
    
    def _generate_detailed_analysis(self, xray_result, symptom_result, combined_prob, consistency_score):
        """Generate detailed analysis breakdown."""
        analysis = {
            'summary': self._generate_analysis_summary(combined_prob, consistency_score),
            'xray_analysis': self._format_xray_analysis(xray_result),
            'symptom_analysis': self._format_symptom_analysis(symptom_result),
            'consistency_assessment': self._format_consistency_assessment(consistency_score),
            'risk_factors': self._combine_risk_factors(xray_result, symptom_result),
            'clinical_correlation': self._generate_clinical_correlation(xray_result, symptom_result)
        }
        
        return analysis
    
    def _generate_analysis_summary(self, combined_prob, consistency_score):
        """Generate summary of the analysis."""
        summary = f"Combined TB probability: {combined_prob:.1%}\n"
        summary += f"Analysis consistency: {consistency_score:.1%}\n"
        
        if combined_prob > 0.7:
            summary += "High likelihood of TB - urgent medical attention required"
        elif combined_prob > 0.4:
            summary += "Moderate TB risk - medical evaluation recommended"
        else:
            summary += "Low TB probability - continue monitoring"
        
        return summary
    
    def _format_xray_analysis(self, xray_result):
        """Format X-ray analysis for detailed view."""
        analysis = {
            'tb_probability': f"{xray_result['tb_probability']:.1%}",
            'confidence': f"{xray_result['confidence']:.1%}",
            'model_quality': xray_result.get('prediction_quality', 'Unknown'),
            'key_features': []
        }
        
        # Add feature analysis if available
        if 'features' in xray_result:
            features = xray_result['features']
            if features.get('lung_opacity', {}).get('score', 0) > 0.5:
                analysis['key_features'].append("Increased lung opacity detected")
            if features.get('cavitation_signs', {}).get('score', 0) > 0.3:
                analysis['key_features'].append("Potential cavitation signs")
            if features.get('nodular_patterns', {}).get('score', 0) > 0.3:
                analysis['key_features'].append("Nodular patterns identified")
        
        return analysis
    
    def _format_symptom_analysis(self, symptom_result):
        """Format symptom analysis for detailed view."""
        analysis = {
            'risk_score': f"{symptom_result['risk_score']:.1f}/100",
            'risk_level': symptom_result['risk_level'],
            'primary_symptoms': symptom_result['primary_symptom_count'],
            'secondary_symptoms': symptom_result['secondary_symptom_count'],
            'key_symptoms': symptom_result['identified_symptoms'][:5],  # Top 5
            'duration_factor': symptom_result.get('duration_multiplier', 1.0),
            'demographic_risk': symptom_result.get('demographic_multiplier', 1.0)
        }
        
        return analysis
    
    def _format_consistency_assessment(self, consistency_score):
        """Format consistency assessment."""
        if consistency_score > 0.8:
            level = "High consistency"
            interpretation = "X-ray and symptom findings strongly correlate"
        elif consistency_score > 0.6:
            level = "Moderate consistency"
            interpretation = "X-ray and symptom findings generally align"
        else:
            level = "Low consistency"
            interpretation = "X-ray and symptom findings show discrepancies - further evaluation needed"
        
        return {
            'consistency_score': f"{consistency_score:.1%}",
            'level': level,
            'interpretation': interpretation
        }
    
    def _combine_risk_factors(self, xray_result, symptom_result):
        """Combine risk factors from both analyses."""
        risk_factors = []
        
        # Add symptom-based risk factors
        if 'risk_factors' in symptom_result:
            risk_factors.extend(symptom_result['risk_factors'])
        
        # Add X-ray based risk factors
        if xray_result['tb_probability'] > 0.5:
            risk_factors.append("Radiological evidence suggestive of TB")
        
        if 'features' in xray_result:
            features = xray_result['features']
            if features.get('cavitation_signs', {}).get('score', 0) > 0.3:
                risk_factors.append("Possible cavitary lesions on imaging")
        
        return list(set(risk_factors))  # Remove duplicates
    
    def _generate_clinical_correlation(self, xray_result, symptom_result):
        """Generate clinical correlation insights."""
        correlations = []
        
        xray_prob = xray_result['tb_probability']
        symptom_score = symptom_result['risk_score']
        
        # High correlation scenarios
        if xray_prob > 0.7 and symptom_score > 70:
            correlations.append("Strong correlation: Both imaging and clinical findings suggest TB")
        elif xray_prob < 0.3 and symptom_score < 30:
            correlations.append("Reassuring correlation: Both imaging and clinical findings suggest low TB risk")
        
        # Discordant findings
        elif xray_prob > 0.7 and symptom_score < 30:
            correlations.append("Discordant findings: High imaging suspicion with minimal symptoms - consider subclinical TB")
        elif xray_prob < 0.3 and symptom_score > 70:
            correlations.append("Discordant findings: High symptom burden with minimal imaging changes - consider extrapulmonary TB")
        
        # Specific clinical scenarios
        if symptom_result.get('primary_symptom_count', 0) >= 3:
            correlations.append("Multiple primary TB symptoms present - high clinical suspicion")
        
        if 'blood_in_sputum' in symptom_result.get('identified_symptoms', []):
            correlations.append("Hemoptysis present - requires immediate evaluation regardless of imaging")
        
        return correlations
