"""
Medical entities and terminology for NLP processing
"""

# Common medical symptoms and their variations
MEDICAL_SYMPTOMS = {
    'headache': [
        'headache', 'headaches', 'head pain', 'head ache', 'cephalgia',
        'migraine', 'migraines', 'tension headache', 'cluster headache'
    ],
    'nausea': [
        'nausea', 'nauseous', 'sick', 'queasy', 'upset stomach',
        'stomach upset', 'feeling sick', 'queasiness'
    ],
    'vomiting': [
        'vomiting', 'vomit', 'throwing up', 'puking', 'emesis',
        'regurgitation', 'retching', 'heaving'
    ],
    'dizziness': [
        'dizziness', 'dizzy', 'vertigo', 'lightheaded', 'light headed',
        'spinning', 'unsteady', 'off balance', 'wobbly'
    ],
    'confusion': [
        'confusion', 'confused', 'disoriented', 'disorientation',
        'bewildered', 'muddled', 'unclear thinking', 'mental fog'
    ],
    'memory_loss': [
        'memory loss', 'memory problems', 'forgetful', 'forgetfulness',
        'amnesia', 'memory impairment', 'can\'t remember', 'memory issues'
    ],
    'seizure': [
        'seizure', 'seizures', 'convulsion', 'convulsions', 'fit', 'fits',
        'epileptic episode', 'spasm', 'tremor', 'shaking'
    ],
    'vision_problems': [
        'blurred vision', 'vision problems', 'visual disturbance', 'sight problems',
        'double vision', 'diplopia', 'visual impairment', 'eye problems',
        'can\'t see clearly', 'vision changes', 'visual symptoms'
    ],
    'weakness': [
        'weakness', 'weak', 'fatigue', 'tired', 'exhausted', 'feeble',
        'muscle weakness', 'loss of strength', 'feeling weak', 'debility'
    ],
    'numbness': [
        'numbness', 'numb', 'tingling', 'pins and needles', 'loss of feeling',
        'sensation loss', 'paresthesia', 'deadness', 'no feeling'
    ],
    'speech_problems': [
        'speech problems', 'difficulty speaking', 'slurred speech', 'dysarthria',
        'can\'t speak properly', 'speech impairment', 'trouble speaking',
        'speech difficulties', 'aphasia', 'words not coming out'
    ],
    'coordination_problems': [
        'coordination problems', 'balance issues', 'unsteady', 'clumsiness',
        'ataxia', 'loss of coordination', 'balance problems', 'stumbling',
        'difficulty walking', 'gait problems'
    ]
}

# Severity indicators
SEVERITY_INDICATORS = {
    'severe': [
        'severe', 'severely', 'intense', 'intensely', 'excruciating',
        'unbearable', 'agonizing', 'terrible', 'awful', 'horrible'
    ],
    'moderate': [
        'moderate', 'moderately', 'medium', 'fair', 'noticeable',
        'significant', 'considerable', 'substantial'
    ],
    'mild': [
        'mild', 'mildly', 'slight', 'slightly', 'minor', 'small',
        'little', 'minimal', 'low', 'weak'
    ],
    'worsening': [
        'worsening', 'getting worse', 'deteriorating', 'declining',
        'progressing', 'advancing', 'escalating', 'intensifying'
    ],
    'improving': [
        'improving', 'getting better', 'better', 'subsiding',
        'decreasing', 'lessening', 'reducing', 'diminishing'
    ]
}

# Duration and time indicators
DURATION_INDICATORS = {
    'acute': [
        'sudden', 'suddenly', 'acute', 'abrupt', 'rapid', 'quick',
        'immediate', 'instant', 'sharp onset', 'all of a sudden'
    ],
    'chronic': [
        'chronic', 'long term', 'long-term', 'ongoing', 'persistent',
        'continuous', 'constant', 'long standing', 'prolonged'
    ],
    'recent': [
        'recent', 'recently', 'new', 'just started', 'began',
        'started', 'first noticed', 'came on', 'developed'
    ],
    'intermittent': [
        'intermittent', 'occasional', 'sometimes', 'comes and goes',
        'on and off', 'episodic', 'sporadic', 'periodic'
    ]
}

# Frequency indicators
FREQUENCY_INDICATORS = {
    'daily': [
        'daily', 'every day', 'everyday', 'each day', 'day to day'
    ],
    'weekly': [
        'weekly', 'every week', 'once a week', 'per week'
    ],
    'monthly': [
        'monthly', 'every month', 'once a month', 'per month'
    ],
    'frequent': [
        'frequently', 'often', 'regularly', 'commonly', 'repeatedly'
    ],
    'rare': [
        'rarely', 'seldom', 'infrequently', 'occasionally', 'hardly ever'
    ]
}

# Medical tests and procedures
MEDICAL_TESTS = {
    'imaging': [
        'mri', 'magnetic resonance imaging', 'ct scan', 'computed tomography',
        'x-ray', 'x ray', 'radiograph', 'ultrasound', 'sonogram',
        'pet scan', 'bone scan', 'angiogram'
    ],
    'blood_tests': [
        'blood test', 'blood work', 'blood panel', 'lab work',
        'laboratory tests', 'cbc', 'complete blood count',
        'blood chemistry', 'serum tests'
    ],
    'neurological': [
        'eeg', 'electroencephalogram', 'emg', 'electromyography',
        'nerve conduction', 'lumbar puncture', 'spinal tap',
        'neurological exam', 'reflex test'
    ]
}

# Body parts and anatomy
ANATOMY_TERMS = {
    'head_neck': [
        'head', 'skull', 'brain', 'neck', 'cervical spine',
        'forehead', 'temple', 'scalp', 'cranium'
    ],
    'neurological': [
        'brain', 'spinal cord', 'nerve', 'nerves', 'nervous system',
        'central nervous system', 'peripheral nervous system',
        'neuron', 'synapse', 'cortex', 'cerebellum', 'brainstem'
    ],
    'sensory': [
        'eye', 'eyes', 'vision', 'sight', 'ear', 'ears', 'hearing',
        'nose', 'smell', 'taste', 'touch', 'sensation'
    ]
}

# Red flag symptoms that strongly suggest need for imaging
RED_FLAG_SYMPTOMS = [
    'sudden severe headache',
    'worst headache of life',
    'thunderclap headache',
    'headache with fever',
    'headache with confusion',
    'headache with seizure',
    'headache with weakness',
    'headache with vision changes',
    'headache with speech problems',
    'new onset seizure',
    'seizure in adult',
    'status epilepticus',
    'focal neurological deficit',
    'acute confusion',
    'sudden memory loss',
    'acute weakness',
    'sudden numbness',
    'acute speech problems',
    'acute vision loss',
    'acute coordination problems'
]

# Medical abbreviations commonly used in patient records
MEDICAL_ABBREVIATIONS = {
    # Vital signs and measurements
    'bp': 'blood pressure',
    'hr': 'heart rate',
    'rr': 'respiratory rate',
    'temp': 'temperature',
    'o2 sat': 'oxygen saturation',
    'bmi': 'body mass index',
    
    # Time and frequency
    'bid': 'twice daily',
    'tid': 'three times daily',
    'qid': 'four times daily',
    'qd': 'once daily',
    'prn': 'as needed',
    'q': 'every',
    'qh': 'every hour',
    'q4h': 'every 4 hours',
    
    # Medical terms
    'hx': 'history',
    'sx': 'symptoms',
    'dx': 'diagnosis',
    'tx': 'treatment',
    'rx': 'prescription',
    'pt': 'patient',
    'pts': 'patients',
    'yo': 'year old',
    'y/o': 'year old',
    
    # Clinical findings
    'sob': 'shortness of breath',
    'loc': 'loss of consciousness',
    'npo': 'nothing by mouth',
    'c/o': 'complains of',
    'w/': 'with',
    'w/o': 'without',
    'n/v': 'nausea and vomiting',
    'ha': 'headache',
    
    # Anatomical
    'cns': 'central nervous system',
    'pns': 'peripheral nervous system',
    'ue': 'upper extremity',
    'le': 'lower extremity',
    'rle': 'right lower extremity',
    'lle': 'left lower extremity',
    'rue': 'right upper extremity',
    'lue': 'left upper extremity'
}

# Urgency scores for different symptoms (0.0 to 1.0)
SYMPTOM_URGENCY_SCORES = {
    'seizure': 0.95,
    'severe headache': 0.9,
    'confusion': 0.85,
    'memory loss': 0.8,
    'speech problems': 0.8,
    'vision problems': 0.75,
    'weakness': 0.7,
    'coordination problems': 0.7,
    'headache': 0.6,
    'numbness': 0.6,
    'dizziness': 0.5,
    'nausea': 0.3,
    'vomiting': 0.4
}

# Patterns that suggest need for urgent evaluation
URGENT_PATTERNS = [
    r'worst headache.*life',
    r'sudden.*severe.*headache',
    r'thunderclap.*headache',
    r'headache.*fever.*neck.*stiff',
    r'seizure.*first.*time',
    r'new.*onset.*seizure',
    r'confusion.*disorientation',
    r'sudden.*weakness',
    r'sudden.*numbness',
    r'acute.*vision.*loss',
    r'sudden.*speech.*problems',
    r'loss.*consciousness',
    r'severe.*head.*trauma',
    r'headache.*pregnancy',
    r'headache.*cancer.*history'
]

def get_symptom_category(symptom_text):
    """Get the category of a symptom"""
    symptom_lower = symptom_text.lower()
    
    for category, variations in MEDICAL_SYMPTOMS.items():
        if any(variation in symptom_lower for variation in variations):
            return category
    
    return 'other'

def get_severity_level(text):
    """Extract severity level from text"""
    text_lower = text.lower()
    
    for severity, indicators in SEVERITY_INDICATORS.items():
        if any(indicator in text_lower for indicator in indicators):
            return severity
    
    return 'unknown'

def get_duration_type(text):
    """Extract duration type from text"""
    text_lower = text.lower()
    
    for duration_type, indicators in DURATION_INDICATORS.items():
        if any(indicator in text_lower for indicator in indicators):
            return duration_type
    
    return 'unknown'

def check_red_flags(text):
    """Check for red flag symptoms in text"""
    text_lower = text.lower()
    red_flags_found = []
    
    for red_flag in RED_FLAG_SYMPTOMS:
        if red_flag.lower() in text_lower:
            red_flags_found.append(red_flag)
    
    return red_flags_found

def get_urgency_score(symptom_text):
    """Get urgency score for a symptom"""
    symptom_lower = symptom_text.lower()
    
    # Check for exact matches first
    for symptom, score in SYMPTOM_URGENCY_SCORES.items():
        if symptom in symptom_lower:
            return score
    
    # Check for category matches
    category = get_symptom_category(symptom_text)
    return SYMPTOM_URGENCY_SCORES.get(category, 0.3)
