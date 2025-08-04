import re
import spacy
from spacy import displacy
import pandas as pd
import numpy as np
from collections import defaultdict
import os

class MedicalNLP:
    """Medical Natural Language Processing for entity extraction"""
    
    def __init__(self):
        self.nlp = None
        self.medical_patterns = {}
        self.initialize_nlp_model()
        self.setup_medical_patterns()
    
    def initialize_nlp_model(self):
        """Initialize spaCy NLP model"""
        try:
            # Try to load a medical-specific model first
            try:
                self.nlp = spacy.load("en_core_sci_sm")  # ScispaCy medical model
            except OSError:
                try:
                    self.nlp = spacy.load("en_core_web_sm")  # Standard English model
                except OSError:
                    # If no model is available, create a blank one
                    self.nlp = spacy.blank("en")
                    # Add basic components
                    self.nlp.add_pipe("sentencizer")
            
            # Add custom medical entity recognition patterns
            self.add_custom_patterns()
            
        except Exception as e:
            # Fallback to rule-based approach
            self.nlp = None
            print(f"Warning: Could not load spaCy model, using rule-based approach: {e}")
    
    def add_custom_patterns(self):
        """Add custom patterns for medical entity recognition"""
        if self.nlp is None:
            return
        
        try:
            # Add EntityRuler for pattern-based entity recognition
            if "entity_ruler" not in self.nlp.pipe_names:
                ruler = self.nlp.add_pipe("entity_ruler", before="ner")
            else:
                ruler = self.nlp.get_pipe("entity_ruler")
            
            # Medical patterns
            patterns = [
                # Symptoms
                {"label": "SYMPTOM", "pattern": [{"LOWER": "headache"}]},
                {"label": "SYMPTOM", "pattern": [{"LOWER": "headaches"}]},
                {"label": "SYMPTOM", "pattern": [{"LOWER": "nausea"}]},
                {"label": "SYMPTOM", "pattern": [{"LOWER": "vomiting"}]},
                {"label": "SYMPTOM", "pattern": [{"LOWER": "dizziness"}]},
                {"label": "SYMPTOM", "pattern": [{"LOWER": "confusion"}]},
                {"label": "SYMPTOM", "pattern": [{"LOWER": "seizure"}]},
                {"label": "SYMPTOM", "pattern": [{"LOWER": "seizures"}]},
                {"label": "SYMPTOM", "pattern": [{"LOWER": "blurred"}, {"LOWER": "vision"}]},
                {"label": "SYMPTOM", "pattern": [{"LOWER": "vision"}, {"LOWER": "problems"}]},
                {"label": "SYMPTOM", "pattern": [{"LOWER": "memory"}, {"LOWER": "loss"}]},
                {"label": "SYMPTOM", "pattern": [{"LOWER": "weakness"}]},
                {"label": "SYMPTOM", "pattern": [{"LOWER": "numbness"}]},
                {"label": "SYMPTOM", "pattern": [{"LOWER": "speech"}, {"LOWER": "problems"}]},
                {"label": "SYMPTOM", "pattern": [{"LOWER": "coordination"}, {"LOWER": "problems"}]},
                
                # Duration
                {"label": "DURATION", "pattern": [{"LIKE_NUM": True}, {"LOWER": "days"}]},
                {"label": "DURATION", "pattern": [{"LIKE_NUM": True}, {"LOWER": "weeks"}]},
                {"label": "DURATION", "pattern": [{"LIKE_NUM": True}, {"LOWER": "months"}]},
                {"label": "DURATION", "pattern": [{"LIKE_NUM": True}, {"LOWER": "years"}]},
                {"label": "DURATION", "pattern": [{"LOWER": "since"}, {"LIKE_NUM": True}]},
                {"label": "DURATION", "pattern": [{"LOWER": "for"}, {"LIKE_NUM": True}]},
                
                # Severity
                {"label": "SEVERITY", "pattern": [{"LOWER": "severe"}]},
                {"label": "SEVERITY", "pattern": [{"LOWER": "mild"}]},
                {"label": "SEVERITY", "pattern": [{"LOWER": "moderate"}]},
                {"label": "SEVERITY", "pattern": [{"LOWER": "intense"}]},
                {"label": "SEVERITY", "pattern": [{"LOWER": "excruciating"}]},
                {"label": "SEVERITY", "pattern": [{"LOWER": "persistent"}]},
                {"label": "SEVERITY", "pattern": [{"LOWER": "constant"}]},
                {"label": "SEVERITY", "pattern": [{"LOWER": "intermittent"}]},
                
                # Medical tests
                {"label": "TEST", "pattern": [{"LOWER": "mri"}]},
                {"label": "TEST", "pattern": [{"LOWER": "ct"}, {"LOWER": "scan"}]},
                {"label": "TEST", "pattern": [{"LOWER": "x-ray"}]},
                {"label": "TEST", "pattern": [{"LOWER": "blood"}, {"LOWER": "test"}]},
                
                # Body parts
                {"label": "ANATOMY", "pattern": [{"LOWER": "head"}]},
                {"label": "ANATOMY", "pattern": [{"LOWER": "brain"}]},
                {"label": "ANATOMY", "pattern": [{"LOWER": "skull"}]},
                {"label": "ANATOMY", "pattern": [{"LOWER": "neck"}]},
            ]
            
            ruler.add_patterns(patterns)
            
        except Exception as e:
            print(f"Warning: Could not add custom patterns: {e}")
    
    def setup_medical_patterns(self):
        """Setup regex patterns for rule-based entity extraction"""
        self.medical_patterns = {
            'symptoms': [
                r'\b(?:headache|headaches|migraine|migraines)\b',
                r'\b(?:nausea|vomiting|dizziness|vertigo)\b',
                r'\b(?:confusion|disorientation|memory loss)\b',
                r'\b(?:seizure|seizures|convulsion|convulsions)\b',
                r'\b(?:blurred vision|vision problems|visual disturbance)\b',
                r'\b(?:weakness|numbness|tingling)\b',
                r'\b(?:speech problems|difficulty speaking|slurred speech)\b',
                r'\b(?:coordination problems|balance issues|unsteady)\b',
                r'\b(?:fatigue|tiredness|exhaustion)\b',
                r'\b(?:personality changes|behavioral changes)\b'
            ],
            'duration': [
                r'\b(?:\d+)\s*(?:day|days|week|weeks|month|months|year|years)\b',
                r'\b(?:since|for|over|about)\s*(?:\d+)\s*(?:day|days|week|weeks|month|months|year|years)\b',
                r'\b(?:recent|recently|sudden|suddenly|gradual|gradually)\b',
                r'\b(?:chronic|acute|persistent|intermittent|constant)\b'
            ],
            'severity': [
                r'\b(?:severe|severe|intense|excruciating|unbearable)\b',
                r'\b(?:moderate|moderately|medium)\b',
                r'\b(?:mild|mildly|slight|slightly|minor)\b',
                r'\b(?:worsening|getting worse|deteriorating)\b',
                r'\b(?:improving|getting better|subsiding)\b',
                r'\b(?:constant|persistent|continuous|ongoing)\b',
                r'\b(?:intermittent|occasional|sporadic|episodic)\b'
            ],
            'frequency': [
                r'\b(?:daily|every day|everyday)\b',
                r'\b(?:weekly|every week)\b',
                r'\b(?:monthly|every month)\b',
                r'\b(?:frequently|often|regularly)\b',
                r'\b(?:rarely|seldom|occasionally)\b',
                r'\b(?:\d+)\s*times?\s*(?:per|a)\s*(?:day|week|month)\b'
            ]
        }
    
    def extract_entities(self, text):
        """Extract medical entities from text"""
        entities = []
        
        if self.nlp is not None:
            # Use spaCy for entity extraction
            entities.extend(self.extract_with_spacy(text))
        
        # Add rule-based extraction as backup/supplement
        entities.extend(self.extract_with_rules(text))
        
        # Remove duplicates and return
        return self.deduplicate_entities(entities)
    
    def extract_with_spacy(self, text):
        """Extract entities using spaCy"""
        entities = []
        
        try:
            doc = self.nlp(text)
            
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': 0.8  # Default confidence for spaCy entities
                })
        
        except Exception as e:
            print(f"Warning: spaCy extraction failed: {e}")
        
        return entities
    
    def extract_with_rules(self, text):
        """Extract entities using rule-based patterns"""
        entities = []
        text_lower = text.lower()
        
        for category, patterns in self.medical_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    entities.append({
                        'text': text[match.start():match.end()],
                        'label': category.upper(),
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 0.6  # Lower confidence for rule-based
                    })
        
        return entities
    
    def deduplicate_entities(self, entities):
        """Remove duplicate entities"""
        seen = set()
        unique_entities = []
        
        for entity in entities:
            # Create a key based on text and label
            key = (entity['text'].lower(), entity['label'])
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def analyze_relationships(self, entities, text):
        """Analyze relationships between entities"""
        relationships = []
        
        # Group entities by type
        entity_groups = defaultdict(list)
        for entity in entities:
            entity_groups[entity['label']].append(entity)
        
        # Find symptom-duration relationships
        symptoms = entity_groups.get('SYMPTOM', []) + entity_groups.get('SYMPTOMS', [])
        durations = entity_groups.get('DURATION', [])
        severities = entity_groups.get('SEVERITY', [])
        
        for symptom in symptoms:
            # Find closest duration
            closest_duration = self.find_closest_entity(symptom, durations, text)
            if closest_duration:
                relationships.append({
                    'type': 'symptom_duration',
                    'entities': [symptom, closest_duration],
                    'confidence': min(symptom['confidence'], closest_duration['confidence'])
                })
            
            # Find closest severity
            closest_severity = self.find_closest_entity(symptom, severities, text)
            if closest_severity:
                relationships.append({
                    'type': 'symptom_severity',
                    'entities': [symptom, closest_severity],
                    'confidence': min(symptom['confidence'], closest_severity['confidence'])
                })
        
        return relationships
    
    def find_closest_entity(self, target_entity, candidate_entities, text):
        """Find the closest entity to a target entity in the text"""
        if not candidate_entities:
            return None
        
        target_pos = (target_entity['start'] + target_entity['end']) / 2
        closest_entity = None
        min_distance = float('inf')
        
        for candidate in candidate_entities:
            candidate_pos = (candidate['start'] + candidate['end']) / 2
            distance = abs(target_pos - candidate_pos)
            
            if distance < min_distance:
                min_distance = distance
                closest_entity = candidate
        
        # Only return if reasonably close (within 100 characters)
        return closest_entity if min_distance < 100 else None
    
    def get_entity_summary(self, entities):
        """Get a summary of extracted entities"""
        summary = defaultdict(int)
        
        for entity in entities:
            summary[entity['label']] += 1
        
        return dict(summary)
    
    def calculate_urgency_score(self, entities, text):
        """Calculate urgency score based on extracted entities"""
        urgency_score = 0.0
        urgent_indicators = []
        
        # High urgency symptoms
        high_urgency_symptoms = [
            'seizure', 'seizures', 'severe headache', 'confusion',
            'memory loss', 'vision problems', 'speech problems'
        ]
        
        # Check for urgent symptoms
        for entity in entities:
            if entity['label'] in ['SYMPTOM', 'SYMPTOMS']:
                text_lower = entity['text'].lower()
                for urgent_symptom in high_urgency_symptoms:
                    if urgent_symptom in text_lower:
                        urgency_score += 0.3
                        urgent_indicators.append(f"Urgent symptom: {entity['text']}")
        
        # Check for severity indicators
        severe_indicators = ['severe', 'intense', 'excruciating', 'unbearable']
        for entity in entities:
            if entity['label'] in ['SEVERITY']:
                text_lower = entity['text'].lower()
                for severe_indicator in severe_indicators:
                    if severe_indicator in text_lower:
                        urgency_score += 0.2
                        urgent_indicators.append(f"High severity: {entity['text']}")
        
        # Check for duration (chronic vs acute)
        for entity in entities:
            if entity['label'] in ['DURATION']:
                text_lower = entity['text'].lower()
                if any(word in text_lower for word in ['sudden', 'acute', 'recent']):
                    urgency_score += 0.15
                    urgent_indicators.append(f"Acute onset: {entity['text']}")
        
        # Cap the score at 1.0
        urgency_score = min(urgency_score, 1.0)
        
        return urgency_score, urgent_indicators
