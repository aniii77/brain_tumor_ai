"""
Data models for the Medical AI System
"""
from datetime import datetime
from typing import Optional, Dict, Any, List

class PatientRecord:
    """Patient record model"""
    def __init__(self,
                 patient_age: Optional[int] = None,
                 patient_gender: Optional[str] = None,
                 symptoms: Optional[str] = None,
                 severity: Optional[str] = None,
                 duration: Optional[str] = None,
                 medical_history: Optional[str] = None):
        self.id: Optional[int] = None
        self.patient_age = patient_age
        self.patient_gender = patient_gender
        self.symptoms = symptoms
        self.severity = severity
        self.duration = duration
        self.medical_history = medical_history
        self.created_at = datetime.utcnow()

class MedicalAnalysis:
    """Medical text analysis results"""
    def __init__(self,
                 patient_record_id: Optional[int] = None,
                 extracted_entities: Optional[Dict[str, Any]] = None,
                 symptom_count: Optional[int] = None,
                 severity_indicators: Optional[int] = None,
                 duration_present: Optional[bool] = None,
                 processed_text: Optional[str] = None):
        self.id: Optional[int] = None
        self.patient_record_id = patient_record_id
        self.extracted_entities = extracted_entities
        self.symptom_count = symptom_count
        self.severity_indicators = severity_indicators
        self.duration_present = duration_present
        self.processed_text = processed_text
        self.created_at = datetime.utcnow()

class MRIRecommendation:
    """MRI scan recommendations"""
    def __init__(self,
                 analysis_id: Optional[int] = None,
                 recommendation_score: Optional[float] = None,
                 recommendation_text: Optional[str] = None,
                 urgency_level: Optional[str] = None,
                 reasons: Optional[Dict[str, Any]] = None,
                 urgent_indicators: Optional[Dict[str, Any]] = None,
                 red_flags_detected: Optional[bool] = None):
        self.id: Optional[int] = None
        self.analysis_id = analysis_id
        self.recommendation_score = recommendation_score
        self.recommendation_text = recommendation_text
        self.urgency_level = urgency_level
        self.reasons = reasons
        self.urgent_indicators = urgent_indicators
        self.red_flags_detected = red_flags_detected
        self.created_at = datetime.utcnow()

class TumorAnalysis:
    """Brain tumor detection analysis"""
    def __init__(self,
                 image_filename: Optional[str] = None,
                 tumor_probability: Optional[float] = None,
                 tumor_type: Optional[str] = None,
                 confidence_score: Optional[float] = None,
                 quality_score: Optional[float] = None,
                 analysis_method: str = 'feature-based'):
        self.id: Optional[int] = None
        self.image_filename = image_filename
        self.tumor_probability = tumor_probability
        self.tumor_type = tumor_type
        self.confidence_score = confidence_score
        self.quality_score = quality_score
        self.analysis_method = analysis_method
        self.created_at = datetime.utcnow()

class DatabaseManager:
    """In-memory database manager"""
    def __init__(self):
        self.patients = []
        self.analyses = []
        self.recommendations = []
        self.tumor_analyses = []
        self._next_id = 1
    
    def create_tables(self):
        """No-op as we're using in-memory storage"""
        pass
    
    def get_session(self):
        """Return self as we don't need sessions"""
        return self
    
    def save_patient_record(self, patient_data):
        """Save patient record to memory"""
        record = PatientRecord(
            patient_age=patient_data.get('age'),
            patient_gender=patient_data.get('gender'),
            symptoms=patient_data.get('symptoms'),
            severity=patient_data.get('severity'),
            duration=patient_data.get('duration'),
            medical_history=patient_data.get('medical_history')
        )
        record.id = self._next_id
        self._next_id += 1
        self.patients.append(record)
        return record.id
    
    def save_medical_analysis(self, analysis_data, patient_record_id):
        """Save medical analysis to memory"""
        analysis = MedicalAnalysis(
            patient_record_id=patient_record_id,
            extracted_entities=analysis_data.get('entities'),
            symptom_count=analysis_data.get('symptom_count'),
            severity_indicators=analysis_data.get('severity_indicators'),
            duration_present=analysis_data.get('duration_present'),
            processed_text=analysis_data.get('processed_text')
        )
        analysis.id = self._next_id
        self._next_id += 1
        self.analyses.append(analysis)
        return analysis.id
    
    def save_mri_recommendation(self, recommendation_data, analysis_id):
        """Save MRI recommendation to memory"""
        recommendation = MRIRecommendation(
            analysis_id=analysis_id,
            recommendation_score=recommendation_data.get('recommendation_score'),
            recommendation_text=recommendation_data.get('recommendation_text'),
            urgency_level=recommendation_data.get('urgency_level'),
            reasons=recommendation_data.get('reasons'),
            urgent_indicators=recommendation_data.get('urgent_indicators'),
            red_flags_detected=recommendation_data.get('red_flags_detected')
        )
        recommendation.id = self._next_id
        self._next_id += 1
        self.recommendations.append(recommendation)
        return recommendation.id
    
    def save_tumor_analysis(self, tumor_data):
        """Save tumor analysis to memory"""
        analysis = TumorAnalysis(
            image_filename=tumor_data.get('image_filename'),
            tumor_probability=tumor_data.get('tumor_probability'),
            tumor_type=tumor_data.get('tumor_type'),
            confidence_score=tumor_data.get('confidence'),
            quality_score=tumor_data.get('quality_score'),
            analysis_method=tumor_data.get('analysis_method', 'feature-based')
        )
        analysis.id = self._next_id
        self._next_id += 1
        self.tumor_analyses.append(analysis)
        return analysis.id
    
    def get_recent_analyses(self, limit=10):
        """Get recent analyses from memory"""
        return sorted(self.patients, key=lambda x: x.created_at, reverse=True)[:limit]
    
    def get_analysis_statistics(self):
        """Get analysis statistics from memory"""
        return {
            'total_patients': len(self.patients),
            'total_analyses': len(self.analyses),
            'total_mri_recommendations': len(self.recommendations),
            'total_tumor_analyses': len(self.tumor_analyses),
            'urgent_recommendations': sum(1 for r in self.recommendations if r.recommendation_score >= 0.7)
        }