import numpy as np
import pandas as pd
from collections import defaultdict
import re

class MRIRecommender:
    """MRI scan recommendation system based on medical entities and symptoms"""
    
    def __init__(self):
        self.symptom_weights = {}
        self.duration_weights = {}
        self.severity_weights = {}
        self.setup_recommendation_rules()
    
    def setup_recommendation_rules(self):
        """Setup weights and rules for MRI recommendations"""
        
        # Symptom weights (0.0 to 1.0) - higher means more likely to need MRI
        self.symptom_weights = {
            'headache': 0.6,
            'headaches': 0.6,
            'severe headache': 0.9,
            'migraine': 0.4,
            'nausea': 0.3,
            'vomiting': 0.4,
            'dizziness': 0.5,
            'confusion': 0.8,
            'memory loss': 0.9,
            'seizure': 0.95,
            'seizures': 0.95,
            'blurred vision': 0.7,
            'vision problems': 0.7,
            'weakness': 0.6,
            'numbness': 0.6,
            'speech problems': 0.8,
            'coordination problems': 0.7,
            'personality changes': 0.8,
            'behavioral changes': 0.7,
            'fatigue': 0.2,
            'balance issues': 0.6
        }
        
        # Duration weights - how duration affects recommendation
        self.duration_weights = {
            'acute': 0.8,      # Sudden onset = higher urgency
            'sudden': 0.9,
            'recently': 0.7,
            'chronic': 0.3,    # Long-term = lower urgency
            'persistent': 0.6,
            'intermittent': 0.4,
            'constant': 0.7,
            'recent': 0.7
        }
        
        # Severity weights
        self.severity_weights = {
            'severe': 0.9,
            'intense': 0.8,
            'excruciating': 0.95,
            'unbearable': 0.95,
            'moderate': 0.5,
            'mild': 0.2,
            'worsening': 0.8,
            'getting worse': 0.8,
            'deteriorating': 0.8,
            'improving': 0.1,
            'getting better': 0.1
        }
        
        # Red flag combinations that strongly suggest MRI
        self.red_flag_combinations = [
            ['headache', 'confusion'],
            ['headache', 'seizure'],
            ['headache', 'vision problems'],
            ['headache', 'weakness'],
            ['seizure', 'memory loss'],
            ['confusion', 'personality changes'],
            ['headache', 'nausea', 'vomiting'],  # Classic triad
        ]
        
        # Urgent indicators
        self.urgent_keywords = [
            'seizure', 'seizures', 'convulsion', 'unconscious',
            'severe headache', 'worst headache', 'thunderclap',
            'confusion', 'disorientation', 'memory loss',
            'sudden weakness', 'sudden numbness', 'sudden speech'
        ]
    
    def recommend(self, entities, text):
        """Generate MRI recommendation based on entities and text"""
        try:
            # Extract symptoms, durations, and severities
            symptoms = self.extract_entity_texts(entities, ['SYMPTOM', 'SYMPTOMS'])
            durations = self.extract_entity_texts(entities, ['DURATION'])
            severities = self.extract_entity_texts(entities, ['SEVERITY'])
            
            # Calculate base score from symptoms
            symptom_score = self.calculate_symptom_score(symptoms)
            
            # Adjust for duration
            duration_modifier = self.calculate_duration_modifier(durations, text)
            
            # Adjust for severity
            severity_modifier = self.calculate_severity_modifier(severities, text)
            
            # Check for red flag combinations
            red_flag_score = self.check_red_flags(symptoms, text)
            
            # Calculate final recommendation score
            base_score = symptom_score * (1 + duration_modifier + severity_modifier)
            final_score = min(base_score + red_flag_score, 1.0)
            
            # Generate reasoning
            reasons = self.generate_reasoning(symptoms, durations, severities, red_flag_score > 0)
            
            # Check for urgent indicators
            urgent_indicators = self.check_urgent_indicators(text, symptoms)
            
            return {
                'recommendation_score': final_score,
                'reasons': reasons,
                'urgent_indicators': urgent_indicators,
                'symptom_count': len(symptoms),
                'severity_mentioned': len(severities) > 0,
                'duration_mentioned': len(durations) > 0,
                'red_flags_detected': red_flag_score > 0
            }
            
        except Exception as e:
            # Return safe default
            return {
                'recommendation_score': 0.3,
                'reasons': [f"Error in analysis: {str(e)}", "Consider medical consultation"],
                'urgent_indicators': [],
                'symptom_count': 0,
                'severity_mentioned': False,
                'duration_mentioned': False,
                'red_flags_detected': False
            }
    
    def extract_entity_texts(self, entities, labels):
        """Extract entity texts for specific labels"""
        texts = []
        for entity in entities:
            if entity['label'] in labels:
                texts.append(entity['text'].lower())
        return texts
    
    def calculate_symptom_score(self, symptoms):
        """Calculate score based on symptoms"""
        if not symptoms:
            return 0.0
        
        total_score = 0.0
        symptom_count = 0
        
        for symptom in symptoms:
            # Check for exact matches first
            if symptom in self.symptom_weights:
                total_score += self.symptom_weights[symptom]
                symptom_count += 1
            else:
                # Check for partial matches
                for key_symptom, weight in self.symptom_weights.items():
                    if key_symptom in symptom or symptom in key_symptom:
                        total_score += weight * 0.8  # Reduced weight for partial match
                        symptom_count += 1
                        break
        
        # Average score with bonus for multiple symptoms
        if symptom_count == 0:
            return 0.0
        
        average_score = total_score / symptom_count
        
        # Bonus for multiple symptoms
        if symptom_count > 2:
            average_score *= 1.2
        elif symptom_count > 1:
            average_score *= 1.1
        
        return min(average_score, 1.0)
    
    def calculate_duration_modifier(self, durations, text):
        """Calculate duration modifier"""
        if not durations:
            return 0.0
        
        modifier = 0.0
        text_lower = text.lower()
        
        for duration in durations:
            for key_duration, weight in self.duration_weights.items():
                if key_duration in duration or key_duration in text_lower:
                    modifier = max(modifier, weight - 0.5)  # Convert to modifier (-0.5 to 0.5)
        
        return modifier
    
    def calculate_severity_modifier(self, severities, text):
        """Calculate severity modifier"""
        if not severities:
            return 0.0
        
        modifier = 0.0
        text_lower = text.lower()
        
        for severity in severities:
            for key_severity, weight in self.severity_weights.items():
                if key_severity in severity or key_severity in text_lower:
                    modifier = max(modifier, weight - 0.5)  # Convert to modifier
        
        return modifier
    
    def check_red_flags(self, symptoms, text):
        """Check for red flag combinations"""
        text_lower = text.lower()
        all_symptoms = symptoms + [text_lower]  # Include full text for comprehensive check
        
        red_flag_score = 0.0
        
        for flag_combination in self.red_flag_combinations:
            matches = 0
            for flag_symptom in flag_combination:
                if any(flag_symptom in symptom for symptom in all_symptoms):
                    matches += 1
            
            # If all symptoms in combination are present
            if matches == len(flag_combination):
                red_flag_score += 0.3
        
        return min(red_flag_score, 0.5)  # Cap red flag contribution
    
    def check_urgent_indicators(self, text, symptoms):
        """Check for urgent indicators requiring immediate attention"""
        urgent_indicators = []
        text_lower = text.lower()
        all_text = text_lower + ' ' + ' '.join(symptoms)
        
        for urgent_keyword in self.urgent_keywords:
            if urgent_keyword in all_text:
                urgent_indicators.append(f"Urgent indicator detected: {urgent_keyword}")
        
        # Check for numeric patterns indicating severity
        severe_pain_patterns = [
            r'pain.*(?:9|10).*(?:out of|/)\s*10',
            r'(?:9|10).*(?:out of|/)\s*10.*pain',
            r'worst.*headache.*life',
            r'never.*felt.*pain.*like'
        ]
        
        for pattern in severe_pain_patterns:
            if re.search(pattern, text_lower):
                urgent_indicators.append("Severe pain intensity reported")
                break
        
        return urgent_indicators
    
    def generate_reasoning(self, symptoms, durations, severities, has_red_flags):
        """Generate human-readable reasoning for recommendation"""
        reasons = []
        
        if symptoms:
            reasons.append(f"Detected {len(symptoms)} neurological symptom(s): {', '.join(symptoms[:3])}")
            
            # High-risk symptoms
            high_risk_symptoms = [s for s in symptoms if self.symptom_weights.get(s, 0) > 0.7]
            if high_risk_symptoms:
                reasons.append(f"High-risk symptoms identified: {', '.join(high_risk_symptoms)}")
        
        if severities:
            reasons.append(f"Severity indicators mentioned: {', '.join(severities)}")
        
        if durations:
            reasons.append(f"Duration information provided: {', '.join(durations)}")
        
        if has_red_flags:
            reasons.append("Red flag symptom combinations detected")
        
        if not symptoms:
            reasons.append("No specific neurological symptoms clearly identified")
            reasons.append("Consider more detailed symptom assessment")
        
        # Add general recommendations
        if len(symptoms) >= 3:
            reasons.append("Multiple symptoms warrant investigation")
        
        return reasons
    
    def get_recommendation_text(self, score):
        """Get human-readable recommendation text"""
        if score >= 0.8:
            return "MRI scan strongly recommended - urgent medical attention advised"
        elif score >= 0.6:
            return "MRI scan recommended - schedule medical consultation"
        elif score >= 0.4:
            return "MRI scan may be beneficial - discuss with healthcare provider"
        else:
            return "MRI scan may not be immediately necessary - monitor symptoms"
