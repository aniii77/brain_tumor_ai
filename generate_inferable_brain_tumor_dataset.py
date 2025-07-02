import pandas as pd
import random
from faker import Faker
from datetime import datetime, timedelta

fake = Faker()

tumor_types = {
    'Pituitary': {
        'regions': ['optic chiasm', 'pituitary gland'],
        'symptoms': ['headaches', 'visual disturbances', 'fatigue', 'hormonal imbalance', 'irregular periods', 'erectile dysfunction'],
        'impact': ['compression of the optic chiasm', 'hormonal dysregulation']
    },
    'Glioma': {
        'regions': ['frontal lobe', 'temporal lobe', 'parietal lobe', 'occipital lobe'],
        'symptoms': ['seizures', 'cognitive decline', 'personality changes', 'speech difficulty', 'motor weakness'],
        'impact': ['edema and pressure on nearby brain tissue', 'interference with motor and cognitive function']
    },
    'Meningioma': {
        'regions': ['cerebellum', 'frontal lobe', 'parietal lobe'],
        'symptoms': ['balance issues', 'nausea', 'headaches', 'hearing loss', 'vision problems'],
        'impact': ['compression of nearby brain structures', 'increased intracranial pressure']
    }
}

treatments = ['surgical resection', 'stereotactic radiosurgery', 'chemotherapy', 'radiation therapy', 'medication management']
follow_up = [
    'regular endocrine evaluations',
    'bi-annual MRI scans to monitor for recurrence',
    'neurological assessments',
    'ongoing hormone replacement therapy'
]

def generate_inferable_summary(patient_id):
    tumor = random.choice(list(tumor_types.keys()))
    region = random.choice(tumor_types[tumor]['regions'])
    gender = random.choice(['male', 'female'])
    age = random.randint(18, 80)
    symptom1, symptom2 = random.sample(tumor_types[tumor]['symptoms'], 2)
    effect = random.choice(tumor_types[tumor]['impact'])
    treatment = random.choice(treatments)
    follow = random.choice(follow_up)
    size = round(random.uniform(1.5, 5.0), 1)
    diagnosis_date = datetime.now() - timedelta(days=random.randint(30, 1000))
    diagnosis_date = diagnosis_date.strftime("%Y-%m-%d")

    summary = (
        f"A {age}-year-old {gender} presented with {symptom1} and {symptom2}. "
        f"MRI revealed a {size} cm lesion located near the {region}, resulting in {effect}. "
        f"The patient underwent {treatment}. Long-term management includes {follow}."
    )

    return {
        "Patient ID": patient_id,
        "Age": age,
        "Gender": gender.capitalize(),
        "Tumor Type": tumor,
        "Diagnosis Date": diagnosis_date,
        "Clinical Summary": summary
    }

# Generate 10,000 records
dataset = [generate_inferable_summary(f"P{300000+i}") for i in range(10000)]
df = pd.DataFrame(dataset)

# Save to CSV
df.to_csv("brain_tumor_inferable_10000.csv", index=False)
print("Dataset generated and saved as brain_tumor_inferable_10000.csv")
