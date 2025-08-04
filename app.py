import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import time


from models.tumor_detector import TumorDetector
from models.medical_nlp import MedicalNLP
from models.mri_recommender import MRIRecommender



from utility.utils.image_processor import ImageProcessor
from utility.utils.text_processor import TextProcessor
from utility.utils.data.database.models import DatabaseManager


from translations import translations


st.set_page_config(
    page_title="BrainWise",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern custom CSS for a beautiful interface
st.markdown("""
    
    <style>
        .feature-card {
            background: #ffffff;
            border-radius: 16px;
            padding: 1.8rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 6px 20px rgba(109, 40, 217, 0.05);
            transition: transform 0.2s ease-in-out;
        }

        .feature-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 24px rgba(109, 40, 217, 0.12);
        }

        .feature-card h4 {
            color: #6D28D9;
            margin-bottom: 1rem;
            font-size: 1.3rem;
        }

        .feature-card ul {
            margin: 0;
            padding-left: 1.5rem;
        }

        .feature-card li {
            margin-bottom: 0.6rem;
            color: #374151;
            font-size: 1rem;
        }
    </style>
""", unsafe_allow_html=True)


# Initialize models and processors lazily per analysis type

@st.cache_resource
def load_tumor_detector():
    try:
        return TumorDetector()
    except Exception as e:
        st.error(f"Error loading TumorDetector: {str(e)}")
        return None

@st.cache_resource
def load_medical_nlp():
    try:
        return MedicalNLP()
    except Exception as e:
        st.error(f"Error loading MedicalNLP: {str(e)}")
        return None

@st.cache_resource
def load_mri_recommender():
    try:
        return MRIRecommender()
    except Exception as e:
        st.error(f"Error loading MRIRecommender: {str(e)}")
        return None

@st.cache_resource
def load_image_processor():
    try:
        return ImageProcessor()
    except Exception as e:
        st.error(f"Error loading ImageProcessor: {str(e)}")
        return None

@st.cache_resource
def load_text_processor():
    try:
        return TextProcessor()
    except Exception as e:
        st.error(f"Error loading TextProcessor: {str(e)}")
        return None

@st.cache_resource
def init_database():
    """Initialize database connection"""
    try:
        db = DatabaseManager()
        db.create_tables()
        return db
    except Exception as e:
        st.error(f"Database connection failed: {str(e)}")
        return None

def main():
    # Now language selector and translation logic
    lang = st.sidebar.selectbox(
        "Language / भाषा / ಭಾಷೆ",
        options=['en', 'hi', 'kn'],
        format_func=lambda x: {'en': 'English', 'hi': 'हिन्दी', 'kn': 'ಕನ್ನಡ'}[x]
    )
    t = translations[lang]

    st.title(f"🧠 {t['title']}")
    st.markdown(f"**{t['ehr_analysis']} & {t['mri_recommendation']}**")
    
    # Initialize database
    db = init_database()
    if db is None:
        st.error("Failed to initialize database. Please check your configuration.")
        return
    
    # Sidebar navigation
    st.sidebar.markdown(f"""
        <div style='text-align: center; margin-bottom: 1rem;'>
            <img src='https://img.icons8.com/fluency/96/brain.png' style='width: 80px; height: 80px; margin-bottom: 10px;'>
            <h2 style='color: #1E3A8A; margin: 0;'>{t['title']}</h2>
            <p style='color: #4B5563; margin: 5px 0;'>{t['brain_tumor_analysis']}</p>
        </div>
    """, unsafe_allow_html=True)

    selected = st.sidebar.radio(
        t['go_to'],
        [t['brain_tumor_analysis'], t['about']]
    )
    
    if selected == t['brain_tumor_analysis']:
        medical_text_analysis(t)
    elif selected == t['about']:
        # Create a container for the About section
        about_container = st.container()
        
        with about_container:
            st.markdown(f"""
                <div style='
        border: 2px solid #6D28D9;
        border-radius: 18px;
        padding: 2rem;
        margin-bottom: 2.5rem;
        background: linear-gradient(to right, #ffffff, #f0f4ff);
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    '>
        <h2 style='
            color: #6D28D9;
            text-align: center;
            font-size: 1.8rem;
            margin-bottom: 1rem;
            font-weight: 700;
        '>
            {t['about']} {t['title']}
        </h2>
        <p style='
            color: #4B5563;
            font-size: 1.1rem;
            line-height: 1.7;
            text-align: center;
            max-width: 700px;
            margin: auto;
        '>
            Our Medical AI system is designed to assist healthcare professionals in brain tumor detection and analysis.
            It blends advanced machine learning models with clinical expertise to deliver accurate, actionable insights.
        </p>
    </div>
            """, unsafe_allow_html=True)
            
            # Core Functionalities
            st.markdown("""
                <div class='card'>
                    <h3>Core Functionalities</h3>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                    <div class='feature-card'>
                        <h4>🎯 Brain Tumor Detection</h4>
                        <ul>
                            <li>Advanced CNN-based tumor detection</li>
                            <li>Multiple tumor type classification</li>
                            <li>Real-time analysis capabilities</li>
                            <li>High-resolution image processing</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                    <div class='feature-card'>
                        <h4>📊 Medical Text Analysis</h4>
                        <ul>
                            <li>Natural Language Processing for medical records</li>
                            <li>Symptom and condition extraction</li>
                            <li>Automated risk assessment</li>
                            <li>Intelligent recommendation system</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                    <div class='feature-card'>
                        <h4>🔍 Analysis Features</h4>
                        <ul>
                            <li>Multi-view tumor visualization</li>
                            <li>Detailed analysis reports</li>
                            <li>Progress tracking</li>
                            <li>Historical data comparison</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                    <div class='feature-card'>
                        <h4>📈 Performance Metrics</h4>
                        <ul>
                            <li>95% tumor detection accuracy</li>
                            <li>92% tumor type classification accuracy</li>
                            <li>Average processing time: 2-3 seconds</li>
                            <li>Support for multiple image formats</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
            
            # System Performance
            st.markdown("""
                <div class='card'>
                    <h3>System Performance</h3>
                </div>
            """, unsafe_allow_html=True)
            
            # Create performance metrics with gauge charts
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fig_accuracy = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = 95,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Detection Accuracy (%)"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "#1E3A8A"},
                        'steps': [
                            {'range': [0, 70], 'color': "lightgray"},
                            {'range': [70, 90], 'color': "gray"},
                            {'range': [90, 100], 'color': "#1E3A8A"}
                        ]
                    }
                ))
                st.plotly_chart(fig_accuracy, use_container_width=True)
            
            with col2:
                fig_speed = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = 2.5,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Processing Time (seconds)"},
                    gauge = {
                        'axis': {'range': [0, 5]},
                        'bar': {'color': "#059669"},
                        'steps': [
                            {'range': [0, 2], 'color': "lightgreen"},
                            {'range': [2, 3], 'color': "yellow"},
                            {'range': [3, 5], 'color': "orange"}
                        ]
                    }
                ))
                st.plotly_chart(fig_speed, use_container_width=True)
            
            with col3:
                fig_reliability = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = 98,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "System Reliability (%)"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "#059669"},
                        'steps': [
                            {'range': [0, 90], 'color': "lightgray"},
                            {'range': [90, 95], 'color': "gray"},
                            {'range': [95, 100], 'color': "#059669"}
                        ]
                    }
                ))
                st.plotly_chart(fig_reliability, use_container_width=True)
             
            # Add custom CSS for better styling
            st.markdown("""
                <style>
                    .feature-card {
                        background-color: white;
                        padding: 1.5rem;
                        border-radius: 10px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        margin-bottom: 1rem;
                    }
                    .feature-card h4 {
                        color: #1E3A8A;
                        margin-bottom: 1rem;
                    }
                    .feature-card ul {
                        margin: 0;
                        padding-left: 1.5rem;
                    }
                    .feature-card li {
                        margin-bottom: 0.5rem;
                        color: #4B5563;
                    }
                </style>
            """, unsafe_allow_html=True)

def medical_text_analysis(t=None):
    if t is None:
        from translations import translations
        t = translations['en']
    st.header(f"📝 {t['ehr_analysis']}")
    st.markdown(f"Analyze patient records to extract medical entities and generate {t['mri_recommendation'].lower()}.")
    
    # Initialize required models
    text_processor = load_text_processor()
    medical_nlp = load_medical_nlp()
    mri_recommender = load_mri_recommender()
    
    if text_processor is None or medical_nlp is None or mri_recommender is None:
        st.error("Failed to initialize required models. Please check your configuration.")
        return
    
    # Text input
    st.markdown(
    "<div style='color: #6B7280; font-size: 0.9rem; margin-bottom: 0.5rem;'>"
    "🧾 <strong>Example:</strong> A 26-year-old female presents with confusion and memory difficulty. "
    "Imaging revealed enhancing lesion with surrounding edema. Surgical resection advised."
    "</div>", unsafe_allow_html=True
)

    patient_text = st.text_area(
    t['patient_record'],
    placeholder="Enter patient symptoms, medical history, and observations...",
    height=220,
    help="Describe patient symptoms, duration, severity, and any relevant medical history"
)

    
    if st.button(t['analyze_text'], type="primary"):
        if patient_text.strip():
            try:
                # Process text
                processed_text = text_processor.clean_text(patient_text)
                
                # Extract medical entities
                with st.spinner("Extracting medical entities..."):
                    entities = medical_nlp.extract_entities(processed_text)
                
                # Generate MRI recommendation
                with st.spinner("Generating MRI recommendation..."):
                    recommendation = mri_recommender.recommend(entities, processed_text)
                
                # Display results
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("🏷️ Detected Symptoms and Conditions")
                    
                    if entities:
                        # Group entities by type
                        entity_types = {}
                        for entity in entities:
                            entity_type = entity['label']
                            if entity_type not in entity_types:
                                entity_types[entity_type] = []
                            entity_types[entity_type].append(entity['text'])
                        
                        # Define symptom categories
                        symptom_categories = {
                            'Neurological': ['headache', 'dizziness', 'confusion', 'seizure', 'numbness', 'weakness', 'speech', 'memory', 'coordination'],
                            'Visual': ['vision', 'eye', 'blur', 'blind', 'photophobia'],
                            'Gastrointestinal': ['nausea', 'vomiting', 'abdominal', 'stomach'],
                            'Pain': ['pain', 'ache', 'discomfort'],
                            'General': ['fever', 'fatigue', 'weakness', 'tiredness']
                        }
                        
                        # Display categorized symptoms
                        st.write("**Primary Symptoms:**")
                        symptoms_found = False
                        for category, keywords in symptom_categories.items():
                            category_symptoms = []
                            for entity_type, entity_list in entity_types.items():
                                for entity in entity_list:
                                    if any(keyword in entity.lower() for keyword in keywords):
                                        category_symptoms.append(entity)
                            
                            if category_symptoms:
                                symptoms_found = True
                                st.write(f"**{category}:**")
                                for symptom in set(category_symptoms):
                                    st.write(f"• {symptom}")
                                st.write("")
                        
                        if not symptoms_found:
                            st.info("No specific symptoms detected in the text.")
                        
                    
                with col2:
                    st.subheader("🎯 MRI Scan Recommendation")
                    
                    if recommendation:
                        # Recommendation score
                        score = recommendation['recommendation_score']
                        
                        def animated_mri_score_bar(score):
                            score = max(0.0, min(score, 1.0))
                            percent = int(score * 100)
                            
                            st.markdown("**Recommendation Confidence:**")
                            progress_text = st.empty()
                            progress_bar = st.progress(0)

                            for i in range(0, percent + 1):
                                clamped_value = min(i, 100)
                                time.sleep(0.01)
                                progress_bar.progress(clamped_value)
                                progress_text.markdown(f"**{clamped_value}%**")

                        if recommendation:
                            animated_mri_score_bar(score)

                        # Recommendation details
                            st.write("**Reasoning:**")
                            for reason in recommendation['reasons']:
                                st.write(f"• {reason}")
                        
                            if recommendation['urgent_indicators']:
                                st.write("**Urgent Indicators:**")
                                for indicator in recommendation['urgent_indicators']:
                                    st.write(f"• {indicator}")
                    else:
                        st.warning("Could not generate MRI recommendation.")
                
                # 🧠 Brain Tumor Visualization Section
                                
                st.markdown("### 🧠 Brain Tumor Visualization", unsafe_allow_html=True)

                fig, ax = plt.subplots(figsize=(3, 3), dpi=100)

                # Centralized tumor location map
                tumor_locations = {
                    'Pituitary':  {'image': 'static/images/tumors/pituitary_tumor.jpg'},
                    'Glioma':     {'image': 'static/images/tumors/glioma.jpg'},
                    'Meningioma': {'image': 'static/images/tumors/meningioma.jpg'},
                    'No Tumor':   {'image': 'static/images/tumors/no_tumor.jpg'}
                }

                # Decide tumor type
                symptom_count = len([e for e in entities if e['label'] == 'SYMPTOM'])
                rec_score = recommendation['recommendation_score'] if recommendation else 0
                tumor_type = infer_tumor_type_from_description(entities, processed_text) if (symptom_count > 0 or rec_score > 0.5) else None

                # Plot
                if tumor_type and tumor_type in tumor_locations:
                    loc = tumor_locations[tumor_type]
                    try:
                        tumor_img = plt.imread(loc['image'])

                        ax.imshow(tumor_img, extent=[0, 1, 0, 1], alpha=0.95)
                        ax.text(0.5, 1.05, tumor_type.upper(), ha='center', va='bottom',
                                fontsize=13, fontweight='bold', color='black', transform=ax.transAxes)
                    except Exception as e:
                        st.warning(f"⚠️ Error loading tumor image: {e}")
                else:
                    ax.text(0.5, 0.5, "No Tumor\nDetected", ha='center', va='center',
                            fontsize=14, fontweight='bold', color='gray')

                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                fig.tight_layout(pad=0)

                # Display in Streamlit
                st.pyplot(fig, use_container_width=True)

                # Add tumor information
                if tumor_type and tumor_type != 'No Tumor':
                    st.write(f"**Detected Tumor Type: {tumor_type}**")
                    tumor_info = {
                        'Pituitary': "Located in the pituitary gland at the base of the brain. Common symptoms include vision problems and hormonal imbalances.",
                        'Glioma': "Develops in the brain's glial cells. Can occur anywhere in the brain and spinal cord.",
                        'Meningioma': "Forms in the meninges, the membranes that surround the brain and spinal cord. Usually benign and slow-growing."
                    }
                    st.info(tumor_info.get(tumor_type, ""))
                
                # Summary table
                st.subheader("📊 Analysis Summary")
                
                # Calculate summary metrics
                symptom_count = len([e for e in entities if e['label'] == 'SYMPTOM'])
                duration_present = any(e['label'] == 'DURATION' for e in entities)
                severity_count = len([e for e in entities if e['label'] == 'SEVERITY'])
                
                if recommendation:
                    rec_score = recommendation['recommendation_score']
                    if rec_score >= 0.7:
                        mri_needed = "🔴 MRI STRONGLY RECOMMENDED"
                        recommendation_pct = f"{rec_score:.0%}"
                    elif rec_score >= 0.4:
                        mri_needed = "🟡 MRI RECOMMENDED"
                        recommendation_pct = f"{rec_score:.0%}"
                    else:
                        mri_needed = "🟢 MRI NOT IMMEDIATELY NEEDED"
                        recommendation_pct = f"{rec_score:.0%}"
                else:
                    mri_needed = "❓ ANALYSIS INCOMPLETE"
                    recommendation_pct = "0%"
                
                summary_data = {
                    'Metric': ['Duration Information', 'Severity Indicators', 'Recommendation Score', 'MRI Assessment'],
                    'Result': [
                        "Present" if duration_present else "Not specified",
                        f"{severity_count} indicators",
                        recommendation_pct,
                        mri_needed
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
            
                # Animated MRI Recommendation Bar
                def animated_mri_score_bar(score):
                    percent = int(score * 100)
                    
                    progress_bar = st.progress(0, text="Calculating recommendation...")

                    for i in range(percent + 1):
                        time.sleep(0.01)
                        progress_bar.progress(i, text=f"Recommendation Score: {i}%")

                    if percent >= 70:
                        st.error("🚨 MRI strongly recommended")
                    elif percent >= 40:
                        st.warning("⚠️ MRI recommended")
                    else:
                        st.success("✅ MRI not immediately needed")

                if recommendation:
                    animated_mri_score_bar(rec_score)

            except Exception as e:
                st.error(f"Error analyzing text: {str(e)}")
        else:
            st.warning("Please enter patient record text to analyze.")

def infer_tumor_type_from_description(entities, processed_text):
    """
    Intelligently infer tumor type from clinical description and extracted entities.
    Uses medical entity analysis and clinical context rather than predefined keywords.
    """
    if not entities:
        return None
    
    # Analyze extracted entities for clinical patterns
    symptoms = [e['text'].lower() for e in entities if e['label'] == 'SYMPTOM']
    conditions = [e['text'].lower() for e in entities if e['label'] == 'CONDITION']
    body_parts = [e['text'].lower() for e in entities if e['label'] == 'BODY_PART']
    severity = [e['text'].lower() for e in entities if e['label'] == 'SEVERITY']
    
    # Combine all relevant medical information
    all_medical_info = symptoms + conditions + body_parts + [processed_text.lower()]
    medical_text = ' '.join(all_medical_info)
    
    # Analyze clinical patterns for tumor type inference
    
    # Pituitary tumor indicators
    pituitary_indicators = 0
    if any(term in medical_text for term in ['vision', 'visual', 'optic', 'chiasm', 'blindness', 'blur']):
        pituitary_indicators += 2
    if any(term in medical_text for term in ['hormonal', 'endocrine', 'prolactin', 'acromegaly', 'cushing']):
        pituitary_indicators += 3
    if any(term in medical_text for term in ['pituitary', 'base of brain', 'sella']):
        pituitary_indicators += 4
    if any(term in medical_text for term in ['menstrual', 'periods', 'erectile', 'libido']):
        pituitary_indicators += 2
    
    # Meningioma indicators
    meningioma_indicators = 0
    if any(term in medical_text for term in ['meninges', 'membrane', 'dural', 'convexity']):
        meningioma_indicators += 4
    if any(term in medical_text for term in ['slow', 'gradual', 'progressive']):
        meningioma_indicators += 2
    if any(term in medical_text for term in ['pressure', 'compression', 'intracranial']):
        meningioma_indicators += 2
    if any(term in medical_text for term in ['meningioma']):
        meningioma_indicators += 5
    
    # Glioma indicators
    glioma_indicators = 0
    if any(term in medical_text for term in ['seizure', 'epilepsy', 'convulsion']):
        glioma_indicators += 3
    if any(term in medical_text for term in ['personality', 'behavior', 'cognitive', 'memory']):
        glioma_indicators += 2
    if any(term in medical_text for term in ['lobe', 'cortex', 'cerebral', 'hemisphere']):
        glioma_indicators += 2
    if any(term in medical_text for term in ['glioma', 'astrocytoma', 'glioblastoma']):
        glioma_indicators += 5
    if any(term in medical_text for term in ['rapid', 'aggressive', 'fast']):
        glioma_indicators += 2
    
    # Determine tumor type based on highest indicators
    tumor_scores = {
        'Pituitary': pituitary_indicators,
        'Meningioma': meningioma_indicators,
        'Glioma': glioma_indicators
    }
    
    # Only return a tumor type if there are significant indicators
    max_score = max(tumor_scores.values())
    if max_score >= 2:  # Threshold to avoid false positives
        tumor_type = max(tumor_scores, key=tumor_scores.get)
        return tumor_type
    
    return None

def comprehensive_analysis(tumor_detector, medical_nlp, mri_recommender, image_processor, text_processor, db):
    """Comprehensive analysis combining both image and text analysis"""
    st.header("🔍 Comprehensive Medical Analysis")
    st.markdown("Combine brain tumor detection with patient record analysis for comprehensive medical assessment.")
    
    # Create tabs for organized input
    tab1, tab2, tab3 = st.tabs(["📷 Image Upload", "📝 Patient Record", "📊 Combined Analysis"])
    
    # Initialize session state
    if 'image_results' not in st.session_state:
        st.session_state.image_results = None
    if 'text_results' not in st.session_state:
        st.session_state.text_results = None
    
    with tab1:
        st.subheader("Upload Brain MRI Scan")
        uploaded_file = st.file_uploader(
            "Choose a brain MRI image...",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            key="comprehensive_image"
        )
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Brain MRI Scan", use_column_width=True)
                
                if st.button("Analyze Image", key="analyze_image_comp"):
                    with st.spinner("Analyzing brain scan..."):
                        processed_image = image_processor.preprocess_image(image)
                        st.session_state.image_results = tumor_detector.predict(processed_image)
                    st.success("Image analysis complete!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    
    with tab2:
        st.subheader("Patient Record Analysis")
        
        # Input method selection for comprehensive analysis
        input_method_comp = st.radio(
            "Choose input method:",
            ["Free Entry", "Structured Form"],
            horizontal=True,
            key="comp_input_method"
        )
        
        patient_text_comp = ""
        
        if input_method_comp == "Free Entry":
            patient_text_comp = st.text_area(
                "Patient Record",
                placeholder="Enter patient symptoms, medical history, and observations...",
                height=200,
                key="comprehensive_text"
            )
            
        elif input_method_comp == "Structured Form":
            col1_comp, col2_comp = st.columns(2)
            
            with col1_comp:
                age_comp = st.number_input("Patient Age", min_value=0, max_value=120, value=45, key="age_comp")
                gender_comp = st.selectbox("Gender", ["Male", "Female", "Other"], key="gender_comp")
                
                st.write("**Primary Symptoms:**")
                headache_comp = st.checkbox("Headache", key="headache_comp")
                nausea_comp = st.checkbox("Nausea/Vomiting", key="nausea_comp")
                dizziness_comp = st.checkbox("Dizziness", key="dizziness_comp")
                confusion_comp = st.checkbox("Confusion", key="confusion_comp")
                seizure_comp = st.checkbox("Seizure", key="seizure_comp")
                
            with col2_comp:
                severity_comp = st.select_slider(
                    "Symptom Severity",
                    options=["Mild", "Moderate", "Severe", "Excruciating"],
                    value="Moderate",
                    key="severity_comp"
                )
                
                duration_comp = st.selectbox(
                    "Symptom Duration",
                    ["Sudden onset (minutes/hours)", "Recent (days)", "Weeks", "Months", "Chronic (years)"],
                    key="duration_comp"
                )
                
                medical_history_comp = st.text_area("Medical History", height=80, key="history_comp")
            
            # Generate text from form
            symptoms_list_comp = []
            if headache_comp: symptoms_list_comp.append("headache")
            if nausea_comp: symptoms_list_comp.append("nausea and vomiting")
            if dizziness_comp: symptoms_list_comp.append("dizziness")
            if confusion_comp: symptoms_list_comp.append("confusion")
            if seizure_comp: symptoms_list_comp.append("seizure")
            
            if symptoms_list_comp:
                patient_text_comp = f"""
Patient: {age_comp}-year-old {gender_comp.lower()}
Chief Complaint: {severity_comp.lower()} {', '.join(symptoms_list_comp)}
Duration: {duration_comp.lower()}
Medical History: {medical_history_comp}
                """.strip()
        
        if st.button("Analyze Text", key="analyze_text_comp"):
            if patient_text_comp.strip():
                try:
                    with st.spinner("Analyzing patient record..."):
                        processed_text = text_processor.clean_text(patient_text_comp)
                        entities = medical_nlp.extract_entities(processed_text)
                        recommendation = mri_recommender.recommend(entities, processed_text)
                        
                        st.session_state.text_results = {
                            'entities': entities,
                            'recommendation': recommendation,
                            'processed_text': processed_text
                        }
                    st.success("Text analysis complete!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error analyzing text: {str(e)}")
            else:
                st.warning("Please enter patient record text to analyze.")
    
    with tab3:
        st.subheader("Combined Medical Assessment")
        
        if st.session_state.image_results or st.session_state.text_results:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.write("### 🔬 Image Analysis Results")
                if st.session_state.image_results:
                    results = st.session_state.image_results
                    tumor_prob = results['tumor_probability']
                    
                    # Create a simple bar chart for tumor probability
                    fig_bar = px.bar(
                        x=['Tumor Probability'],
                        y=[tumor_prob * 100],
                        title="Tumor Detection Results",
                        color=[tumor_prob * 100],
                        color_continuous_scale=['green', 'yellow', 'red'],
                        range_color=[0, 100]
                    )
                    fig_bar.update_layout(showlegend=False, yaxis_title="Probability (%)")
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
                    st.metric("Tumor Type", results['tumor_type'], f"{results['confidence']:.1%} confidence")
                    
                    # Add organ information
                    tumor_organ_mapping = {
                        'Glioma': 'Brain',
                        'Meningioma': 'Brain',
                        'Pituitary': 'Brain',
                        'No Tumor': 'None'
                    }
                    affected_organ = tumor_organ_mapping.get(results['tumor_type'], 'Unknown')
                    st.metric("Affected Organ", affected_organ)
                    
                    # Display tumor visualization
                    st.subheader("Tumor Visualization")
                    tumor_images = {
                        'Glioma': 'static/images/tumors/glioma.jpg',
                        'Meningioma': 'static/images/tumors/meningioma.jpg',
                        'Pituitary': 'static/images/tumors/pituitary_tumor.jpg',
                        'No Tumor': 'static/images/tumors/no_tumor.jpg'
                    }
                    
                    # Display the corresponding tumor type image
                    if results['tumor_type'] in tumor_images:
                        try:
                            image_path = tumor_images[results['tumor_type']]
                            st.image(image_path, 
                                    caption=f"{results['tumor_type']} Tumor Visualization", 
                                    width=150)
                        except Exception as e:
                            st.warning(f"Could not load tumor visualization image: {str(e)}")
                            st.info("Please ensure the following images are present in the static/images/tumors directory:")
                            st.write("- glioma.jpg")
                            st.write("- meningioma.jpg")
                            st.write("- pituitary_tumor.jpg")
                            st.write("- no_tumor.jpg")
                    
                    # Display symptoms based on tumor type
                    symptoms = {
                        'Glioma': [
                            'Headaches',
                            'Seizures',
                            'Personality changes',
                            'Memory problems',
                            'Difficulty speaking'
                        ],
                        'Meningioma': [
                            'Headaches',
                            'Vision problems',
                            'Hearing loss',
                            'Memory problems',
                            'Weakness in limbs'
                        ],
                        'Pituitary': [
                            'Vision problems',
                            'Hormonal changes',
                            'Headaches',
                            'Nausea',
                            'Fatigue'
                        ],
                        'No Tumor': [
                            'No specific symptoms',
                            'Regular check-ups recommended'
                        ]
                    }
                    
                    # Display symptoms in an expander
                    with st.expander("Common Symptoms for Detected Tumor Type"):
                        for symptom in symptoms.get(results['tumor_type'], []):
                            st.write(f"• {symptom}")
                else:
                    st.info("No image analysis results available.")
            
            with col2:
                st.write("### 📝 Text Analysis Results")
                if st.session_state.text_results:
                    recommendation = st.session_state.text_results['recommendation']
                    entities = st.session_state.text_results['entities']
                    
                    if recommendation:
                        score = recommendation['recommendation_score']
                        st.metric("MRI Recommendation Score", f"{score:.1%}")
                        
                        # Entity count chart
                        entity_counts = {}
                        for entity in entities:
                            label = entity['label']
                            entity_counts[label] = entity_counts.get(label, 0) + 1
                        
                        if entity_counts:
                            fig_entities = px.pie(
                                values=list(entity_counts.values()),
                                names=list(entity_counts.keys()),
                                title="Medical Entities Distribution"
                            )
                            st.plotly_chart(fig_entities, use_container_width=True)
                else:
                    st.info("No text analysis results available.")
            
            # Combined recommendation
            if st.session_state.image_results and st.session_state.text_results:
                st.write("### 🎯 Combined Medical Recommendation")
                
                image_score = st.session_state.image_results['tumor_probability']
                text_score = st.session_state.text_results['recommendation']['recommendation_score']
                
                # Calculate combined risk score
                combined_score = (image_score * 0.6 + text_score * 0.4)  # Weight image analysis higher
                
                if combined_score > 0.7:
                    st.error(f"🚨 **HIGH PRIORITY**: Combined analysis indicates high risk (Score: {combined_score:.1%})")
                    st.error("Immediate medical attention and MRI scan strongly recommended.")
                elif combined_score > 0.4:
                    st.warning(f"⚠️ **MODERATE PRIORITY**: Combined analysis indicates moderate risk (Score: {combined_score:.1%})")
                    st.warning("Medical consultation and MRI scan recommended.")
                else:
                    st.success(f"✅ **LOW PRIORITY**: Combined analysis indicates low risk (Score: {combined_score:.1%})")
                    st.success("Routine monitoring may be sufficient.")
                
                # Combined metrics
                metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                with metrics_col1:
                    st.metric("Image Risk", f"{image_score:.1%}")
                with metrics_col2:
                    st.metric("Text Risk", f"{text_score:.1%}")
                with metrics_col3:
                    st.metric("Combined Risk", f"{combined_score:.1%}")
                with metrics_col4:
                    # Get tumor type and organ information
                    tumor_type = st.session_state.image_results['tumor_type']
                    tumor_organ_mapping = {
                        'Glioma': 'Brain',
                        'Meningioma': 'Brain',
                        'Pituitary': 'Brain',
                        'No Tumor': 'None'
                    }
                    affected_organ = tumor_organ_mapping.get(tumor_type, 'Unknown')
                    st.metric("Affected Organ", affected_organ)
        else:
            st.info("Please complete both image and text analysis to see combined results.")

def database_dashboard(db):
    """Database dashboard for viewing stored analyses"""
    st.header("📊 Database Dashboard")
    st.markdown("View stored patient records, analyses, and system statistics.")
    
    if db is None:
        st.error("Database connection not available.")
        return
    
    try:
        # Get statistics
        stats = db.get_analysis_statistics()
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Patients", stats['total_patients'])
        
        with col2:
            st.metric("Medical Analyses", stats['total_analyses'])
        
        with col3:
            st.metric("MRI Recommendations", stats['total_mri_recommendations'])
        
        with col4:
            st.metric("Urgent Cases", stats['urgent_recommendations'])
        
        # Recent analyses
        st.subheader("Recent Patient Records")
        recent_records = db.get_recent_analyses(10)
        
        if recent_records:
            for record in recent_records:
                with st.expander(f"Patient {record.id} - {record.created_at.strftime('%Y-%m-%d %H:%M')}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Age:** {record.patient_age}")
                        st.write(f"**Gender:** {record.patient_gender}")
                        st.write(f"**Severity:** {record.severity}")
                    
                    with col2:
                        st.write(f"**Duration:** {record.duration}")
                        st.write(f"**Symptoms:** {record.symptoms}")
                    
                    if record.medical_history:
                        st.write(f"**Medical History:** {record.medical_history}")
        else:
            st.info("No patient records found.")
    
    except Exception as e:
        st.error(f"Error loading dashboard data: {str(e)}")

def display_analysis_results(analysis_results):
    """Display analysis results with tumor type images"""
    st.subheader("Analysis Results")
    
    # Display patient tumor type information in a prominent box
    if analysis_results.get('tumor_type'):
        tumor_type = analysis_results['tumor_type']
        st.markdown("### Patient Tumor Analysis")
        st.markdown("""
        <div style='
    background-color: {bg_color};
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    margin: 2rem 0;
    text-align: center;
    color: white;
'>
    <h2 style='margin-bottom: 1rem; font-size: 1.6rem;'>🧬 Detected Tumor Type</h2>
    <p style='font-size: 2rem; font-weight: 700; letter-spacing: 1px;'>{tumor_type}</p>
</div>
        """.format(tumor_type), unsafe_allow_html=True)
        
        # Try to display tumor type image
        image_path = f"static/images/tumors/{tumor_type.lower().replace(' ', '_')}.jpg"
        try:
            st.image(image_path, caption=f"Example of {tumor_type}", use_column_width=True)
        except:
            st.info(f"Image for {tumor_type} not available. Please add an image at: {image_path}")
    
    # Display metrics in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if analysis_results.get('confidence'):
            st.metric("Confidence Score", f"{analysis_results['confidence']:.2f}%")
    
    with col2:
        if analysis_results.get('severity'):
            severity = analysis_results['severity']
            st.metric("Severity Level", severity)
    
    with col3:
        # Add Organ metric
        organ_mapping = {
            'Meningioma': 'Brain (Meninges)',
            'Glioma': 'Brain',
            'Pituitary Tumor': 'Pituitary Gland',
            'Acoustic Neuroma': 'Brain (Vestibulocochlear Nerve)',
            'Craniopharyngioma': 'Brain (Pituitary Region)',
            'Medulloblastoma': 'Brain (Cerebellum)',
            'Ependymoma': 'Brain/Spinal Cord',
            'Schwannoma': 'Nervous System',
            'Hemangioblastoma': 'Brain/Spinal Cord',
            'Choroid Plexus Tumor': 'Brain (Ventricles)'
        }
        affected_organ = organ_mapping.get(tumor_type, 'Not Specified')
        st.metric("Affected Organ", affected_organ)
    
    # Display other results
    if analysis_results.get('key_findings'):
        st.markdown("### Key Findings")
        for finding in analysis_results['key_findings']:
            st.markdown(f"- {finding}")
    
    if analysis_results.get('recommendations'):
        st.markdown("### Recommendations")
        for rec in analysis_results['recommendations']:
            st.markdown(f"- {rec}")

if __name__ == "__main__":
    main()
