import re
import string
import unicodedata
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

class TextProcessor:
    """Text processing utilities for medical text analysis"""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set()
        self.medical_abbreviations = {}
        self.setup_nltk_data()
        self.setup_medical_vocabulary()
    
    def setup_nltk_data(self):
        """Setup NLTK data with error handling"""
        try:
            # Download required NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            
            # Load stopwords
            self.stop_words = set(stopwords.words('english'))
            
        except Exception as e:
            # Fallback to basic stopwords if NLTK fails
            self.stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
                'before', 'after', 'above', 'below', 'between', 'among', 'is', 'are',
                'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
                'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can'
            }
    
    def setup_medical_vocabulary(self):
        """Setup medical abbreviations and terminology"""
        self.medical_abbreviations = {
            'mri': 'magnetic resonance imaging',
            'ct': 'computed tomography',
            'bp': 'blood pressure',
            'hr': 'heart rate',
            'temp': 'temperature',
            'resp': 'respiratory',
            'o2': 'oxygen',
            'co2': 'carbon dioxide',
            'hx': 'history',
            'sx': 'symptoms',
            'dx': 'diagnosis',
            'tx': 'treatment',
            'rx': 'prescription',
            'pt': 'patient',
            'pts': 'patients',
            'yo': 'year old',
            'yrs': 'years',
            'mos': 'months',
            'wks': 'weeks',
            'q': 'every',
            'bid': 'twice daily',
            'tid': 'three times daily',
            'qid': 'four times daily',
            'prn': 'as needed',
            'npo': 'nothing by mouth',
            'sob': 'shortness of breath',
            'loc': 'loss of consciousness',
            'n/v': 'nausea and vomiting',
            'ha': 'headache',
            'c/o': 'complains of',
            'w/': 'with',
            'w/o': 'without'
        }
        
        # Medical terms that should not be considered stopwords
        self.medical_stopwords_exceptions = {
            'no', 'not', 'never', 'none', 'nothing', 'without', 'absence', 'absent',
            'severe', 'mild', 'moderate', 'chronic', 'acute', 'sudden', 'gradual'
        }
    
    def clean_text(self, text):
        """Clean and preprocess medical text"""
        if not text:
            return ""
        
        try:
            # Basic cleaning
            cleaned_text = self.basic_cleaning(text)
            
            # Expand medical abbreviations
            cleaned_text = self.expand_abbreviations(cleaned_text)
            
            # Normalize spacing and punctuation
            cleaned_text = self.normalize_spacing(cleaned_text)
            
            # Handle special medical formatting
            cleaned_text = self.handle_medical_formatting(cleaned_text)
            
            return cleaned_text
            
        except Exception as e:
            # Return original text if cleaning fails
            return text
    
    def basic_cleaning(self, text):
        """Perform basic text cleaning"""
        # Remove or replace special characters while preserving medical notation
        text = re.sub(r'[^\w\s\-\./,;:\(\)]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def expand_abbreviations(self, text):
        """Expand medical abbreviations"""
        words = text.split()
        expanded_words = []
        
        for word in words:
            word_lower = word.lower().rstrip('.,;:')
            if word_lower in self.medical_abbreviations:
                expanded_words.append(self.medical_abbreviations[word_lower])
            else:
                expanded_words.append(word)
        
        return ' '.join(expanded_words)
    
    def normalize_spacing(self, text):
        """Normalize spacing and punctuation"""
        # Fix spacing around punctuation
        text = re.sub(r'\s*([,.;:!?])\s*', r'\1 ', text)
        
        # Fix spacing around parentheses
        text = re.sub(r'\s*\(\s*', ' (', text)
        text = re.sub(r'\s*\)\s*', ') ', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def handle_medical_formatting(self, text):
        """Handle medical-specific formatting"""
        # Normalize medical measurements and units
        text = re.sub(r'(\d+)\s*(mg|ml|cc|kg|lb|cm|mm|inch|in)', r'\1\2', text)
        
        # Normalize date formats
        text = re.sub(r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})', r'\1/\2/\3', text)
        
        # Normalize time formats
        text = re.sub(r'(\d{1,2}):(\d{2})\s*(am|pm)', r'\1:\2\3', text, flags=re.IGNORECASE)
        
        return text
    
    def tokenize_text(self, text):
        """Tokenize text into sentences and words"""
        try:
            # Sentence tokenization
            sentences = sent_tokenize(text)
            
            # Word tokenization for each sentence
            tokenized_sentences = []
            for sentence in sentences:
                words = word_tokenize(sentence.lower())
                tokenized_sentences.append(words)
            
            return {
                'sentences': sentences,
                'tokenized_sentences': tokenized_sentences,
                'all_tokens': [token for sentence in tokenized_sentences for token in sentence]
            }
            
        except Exception as e:
            # Fallback tokenization
            sentences = text.split('.')
            words = re.findall(r'\b\w+\b', text.lower())
            return {
                'sentences': sentences,
                'tokenized_sentences': [words],
                'all_tokens': words
            }
    
    def remove_stopwords(self, tokens):
        """Remove stopwords while preserving medical terms"""
        filtered_tokens = []
        
        for token in tokens:
            # Keep medical exception terms even if they're typically stopwords
            if token in self.medical_stopwords_exceptions:
                filtered_tokens.append(token)
            # Remove standard stopwords
            elif token not in self.stop_words and len(token) > 1:
                filtered_tokens.append(token)
        
        return filtered_tokens
    
    def stem_and_lemmatize(self, tokens):
        """Apply stemming and lemmatization"""
        processed_tokens = []
        
        for token in tokens:
            try:
                # Apply lemmatization first (more accurate)
                lemmatized = self.lemmatizer.lemmatize(token)
                processed_tokens.append(lemmatized)
            except:
                # Fallback to stemming
                try:
                    stemmed = self.stemmer.stem(token)
                    processed_tokens.append(stemmed)
                except:
                    # Keep original token if both fail
                    processed_tokens.append(token)
        
        return processed_tokens
    
    def extract_medical_keywords(self, text):
        """Extract medical keywords and phrases"""
        medical_keywords = []
        
        # Medical symptom patterns
        symptom_patterns = [
            r'\b(?:pain|ache|aches|aching|hurt|hurting|sore|tender)\b',
            r'\b(?:headache|migraine|head pain)\b',
            r'\b(?:nausea|nauseous|sick|vomit|vomiting)\b',
            r'\b(?:dizzy|dizziness|vertigo|lightheaded)\b',
            r'\b(?:confused|confusion|disoriented|memory)\b',
            r'\b(?:seizure|seizures|convulsion|fit)\b',
            r'\b(?:weakness|weak|numbness|numb|tingling)\b',
            r'\b(?:vision|visual|sight|see|seeing|blurred|blur)\b',
            r'\b(?:speech|speak|speaking|talk|talking|slurred)\b',
            r'\b(?:balance|coordination|unsteady|stumble)\b'
        ]
        
        text_lower = text.lower()
        for pattern in symptom_patterns:
            matches = re.findall(pattern, text_lower)
            medical_keywords.extend(matches)
        
        # Medical time/duration patterns
        duration_patterns = [
            r'\b(?:\d+)\s*(?:day|days|week|weeks|month|months|year|years)\b',
            r'\b(?:since|for|over|about)\s*(?:\d+)\s*(?:day|days|week|weeks|month|months|year|years)\b',
            r'\b(?:sudden|suddenly|recent|recently|chronic|acute)\b'
        ]
        
        for pattern in duration_patterns:
            matches = re.findall(pattern, text_lower)
            medical_keywords.extend(matches)
        
        return list(set(medical_keywords))  # Remove duplicates
    
    def calculate_readability(self, text):
        """Calculate text readability metrics"""
        try:
            sentences = sent_tokenize(text)
            words = word_tokenize(text)
            
            # Basic counts
            sentence_count = len(sentences)
            word_count = len(words)
            
            if sentence_count == 0 or word_count == 0:
                return {'error': 'No sentences or words found'}
            
            # Average sentence length
            avg_sentence_length = word_count / sentence_count
            
            # Syllable count (approximation)
            syllable_count = sum(self.count_syllables(word) for word in words)
            avg_syllables_per_word = syllable_count / word_count if word_count > 0 else 0
            
            # Flesch Reading Ease (simplified)
            flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
            
            return {
                'sentence_count': sentence_count,
                'word_count': word_count,
                'avg_sentence_length': round(avg_sentence_length, 2),
                'avg_syllables_per_word': round(avg_syllables_per_word, 2),
                'flesch_score': round(flesch_score, 2),
                'readability_level': self.get_readability_level(flesch_score)
            }
            
        except Exception as e:
            return {'error': f'Error calculating readability: {str(e)}'}
    
    def count_syllables(self, word):
        """Count syllables in a word (approximation)"""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for i, char in enumerate(word):
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel
        
        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)  # Every word has at least one syllable
    
    def get_readability_level(self, flesch_score):
        """Convert Flesch score to readability level"""
        if flesch_score >= 90:
            return "Very Easy"
        elif flesch_score >= 80:
            return "Easy"
        elif flesch_score >= 70:
            return "Fairly Easy"
        elif flesch_score >= 60:
            return "Standard"
        elif flesch_score >= 50:
            return "Fairly Difficult"
        elif flesch_score >= 30:
            return "Difficult"
        else:
            return "Very Difficult"
    
    def extract_numbers_and_measurements(self, text):
        """Extract numerical values and measurements from text"""
        measurements = []
        
        # Pattern for numbers with units
        number_unit_pattern = r'(\d+(?:\.\d+)?)\s*(mg|ml|cc|kg|lb|cm|mm|inch|in|degrees?|°|bpm|mmhg|/\d+)'
        matches = re.finditer(number_unit_pattern, text.lower())
        
        for match in matches:
            measurements.append({
                'value': float(match.group(1)),
                'unit': match.group(2),
                'full_text': match.group(0)
            })
        
        # Pattern for pain scales
        pain_scale_pattern = r'(\d+)\s*(?:out of|/|of)\s*(\d+)'
        pain_matches = re.finditer(pain_scale_pattern, text)
        
        for match in pain_matches:
            measurements.append({
                'value': float(match.group(1)),
                'scale': float(match.group(2)),
                'type': 'pain_scale',
                'full_text': match.group(0)
            })
        
        return measurements
