import numpy as np
import cv2
from PIL import Image
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

class XRayAnalyzer:
    def __init__(self):
        """Initialize the X-ray analyzer with a machine learning model."""
        self.model = self._load_model()
        self.scaler = StandardScaler()
        self.input_shape = (224, 224, 3)
        self.is_trained = False
    
    def _load_model(self):
        """Load or create the TB detection model."""
        model_path = "models/tb_xray_model.pkl"
        scaler_path = "models/tb_scaler.pkl"
        
        try:
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                self.is_trained = True
                return model
            else:
                # Create a new model for demonstration
                return self._create_model()
        except Exception as e:
            st.warning(f"Creating new model: {str(e)}")
            return self._create_model()
    
    def _create_model(self):
        """Create a Random Forest model for TB detection."""
        # Using Random Forest as it works well for image feature classification
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        return model
    
    def predict(self, processed_image):
        """Predict TB probability from processed X-ray image."""
        try:
            # Extract features from the image
            features = self._extract_image_features(processed_image)
            
            if not self.is_trained:
                # For demonstration purposes, use a heuristic-based approach
                # when no trained model is available
                return self._heuristic_prediction(processed_image, features)
            
            # Prepare features for prediction
            features_array = np.array(features).reshape(1, -1)
            features_scaled = self.scaler.transform(features_array)
            
            # Make prediction
            probabilities = self.model.predict_proba(features_scaled)[0]
            prediction = self.model.predict(features_scaled)[0]
            
            # Extract probabilities (assuming class 0=Normal, class 1=TB)
            if len(probabilities) == 2:
                normal_probability = float(probabilities[0])
                tb_probability = float(probabilities[1])
            else:
                # Fallback for single class prediction
                tb_probability = float(prediction)
                normal_probability = 1.0 - tb_probability
            
            # Calculate confidence as the maximum probability
            confidence = max(tb_probability, normal_probability)
            
            # Analyze image features for additional insights
            image_features = self._analyze_features(processed_image)
            
            return {
                'tb_probability': tb_probability,
                'normal_probability': normal_probability,
                'confidence': confidence,
                'features': image_features,
                'model_used': 'RandomForest_TB_Classifier',
                'prediction_quality': self._assess_prediction_quality(confidence)
            }
            
        except Exception as e:
            # Fallback to heuristic prediction if model fails
            features = self._extract_image_features(processed_image)
            return self._heuristic_prediction(processed_image, features)
    
    def _analyze_features(self, image):
        """Analyze specific features in the X-ray that might indicate TB."""
        features = {}
        
        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        # Analyze lung opacity patterns
        features['lung_opacity'] = self._analyze_opacity(gray)
        
        # Detect potential cavitations
        features['cavitation_signs'] = self._detect_cavitations(gray)
        
        # Analyze nodular patterns
        features['nodular_patterns'] = self._detect_nodules(gray)
        
        # Check for pleural effusion signs
        features['pleural_effusion'] = self._detect_pleural_effusion(gray)
        
        return features
    
    def _analyze_opacity(self, gray_image):
        """Analyze lung opacity patterns."""
        # Calculate histogram
        hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
        
        # Analyze density distribution
        mean_intensity = np.mean(gray_image)
        std_intensity = np.std(gray_image)
        
        # High opacity often indicates pathological changes
        opacity_score = min(1.0, (255 - mean_intensity) / 128)
        
        return {
            'score': float(opacity_score),
            'mean_intensity': float(mean_intensity),
            'std_intensity': float(std_intensity)
        }
    
    def _detect_cavitations(self, gray_image):
        """Detect potential cavitation patterns."""
        # Use blob detection for circular dark regions
        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = True
        params.blobColor = 0  # Dark blobs
        params.filterByArea = True
        params.minArea = 100
        params.maxArea = 2000
        
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(gray_image)
        
        cavitation_score = min(1.0, len(keypoints) / 10)
        
        return {
            'score': float(cavitation_score),
            'count': len(keypoints)
        }
    
    def _detect_nodules(self, gray_image):
        """Detect nodular patterns that might indicate TB."""
        # Apply Gaussian blur and edge detection
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter for small circular contours (potential nodules)
        nodule_count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < 500:  # Size range for nodules
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    if circularity > 0.7:  # Circular shape
                        nodule_count += 1
        
        nodule_score = min(1.0, nodule_count / 20)
        
        return {
            'score': float(nodule_score),
            'count': nodule_count
        }
    
    def _detect_pleural_effusion(self, gray_image):
        """Detect signs of pleural effusion."""
        # Analyze bottom regions of the image for fluid accumulation
        height, width = gray_image.shape
        bottom_region = gray_image[int(height * 0.7):, :]
        
        # Check for horizontal fluid lines
        mean_bottom = np.mean(bottom_region, axis=1)
        horizontal_gradient = np.diff(mean_bottom)
        
        # Strong horizontal changes might indicate fluid levels
        gradient_score = np.std(horizontal_gradient) / 50
        effusion_score = min(1.0, gradient_score)
        
        return {
            'score': float(effusion_score),
            'gradient_strength': float(np.std(horizontal_gradient))
        }
    
    def _assess_prediction_quality(self, confidence):
        """Assess the quality of the prediction based on confidence."""
        if confidence > 0.9:
            return "Excellent"
        elif confidence > 0.8:
            return "Good"
        elif confidence > 0.7:
            return "Fair"
        else:
            return "Poor"
    
    def _extract_image_features(self, image):
        """Extract numerical features from the image for ML model."""
        # Convert to grayscale for feature extraction
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        features = []
        
        # Statistical features
        features.extend([
            np.mean(gray),
            np.std(gray),
            np.min(gray),
            np.max(gray),
            np.median(gray)
        ])
        
        # Texture features using LBP (simplified)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        features.extend(hist.flatten()[:50])  # First 50 histogram bins
        
        # Edge features
        edges = cv2.Canny(gray, 50, 150)
        features.extend([
            np.sum(edges > 0),  # Edge pixel count
            np.mean(edges),     # Mean edge intensity
        ])
        
        # Shape features
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        features.extend([
            len(contours),  # Number of contours
            sum(cv2.contourArea(c) for c in contours) if contours else 0  # Total contour area
        ])
        
        return features
    
    def _heuristic_prediction(self, image, features):
        """Heuristic-based prediction when no trained model is available."""
        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        # Analyze image characteristics
        image_features = self._analyze_features(image)
        
        # Simple heuristic based on image characteristics
        risk_score = 0.0
        
        # Opacity analysis
        opacity_score = image_features.get('lung_opacity', {}).get('score', 0)
        risk_score += opacity_score * 0.4
        
        # Cavitation signs
        cavitation_score = image_features.get('cavitation_signs', {}).get('score', 0)
        risk_score += cavitation_score * 0.3
        
        # Nodular patterns
        nodule_score = image_features.get('nodular_patterns', {}).get('score', 0)
        risk_score += nodule_score * 0.2
        
        # Pleural effusion
        effusion_score = image_features.get('pleural_effusion', {}).get('score', 0)
        risk_score += effusion_score * 0.1
        
        # Add some randomness for demonstration (simulate model uncertainty)
        import random
        random.seed(int(np.sum(gray)) % 1000)  # Deterministic based on image
        noise = random.uniform(-0.1, 0.1)
        risk_score = max(0.0, min(1.0, risk_score + noise))
        
        tb_probability = risk_score
        normal_probability = 1.0 - tb_probability
        confidence = 0.6 + abs(tb_probability - 0.5) * 0.4  # Higher confidence when further from 0.5
        
        return {
            'tb_probability': tb_probability,
            'normal_probability': normal_probability,
            'confidence': confidence,
            'features': image_features,
            'model_used': 'Heuristic_Analysis',
            'prediction_quality': self._assess_prediction_quality(confidence)
        }
    
    def get_model_info(self):
        """Return information about the model."""
        return {
            'model_type': 'Random Forest Classifier' if self.is_trained else 'Heuristic Analysis',
            'input_shape': self.input_shape,
            'classes': ['Normal', 'TB'],
            'is_trained': self.is_trained,
            'features_used': 'Image statistics, texture, edges, and morphological features'
        }
