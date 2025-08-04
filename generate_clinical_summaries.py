import csv
import random
from faker import Faker

fake = Faker()
tumor_types = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']
symptoms = {
    'Glioma': [
        'progressive headaches', 'seizures', 'personality changes', 'memory loss', 'speech difficulty', 'mass in the frontal lobe', 'mass in the occipital lobe'
    ],
    'Meningioma': [
        'gradual vision loss', 'hearing difficulty', 'slow-growing mass near the meninges', 'balance issues', 'compression of nearby brain structures'
    ],
    'Pituitary': [
        'hormonal imbalance', 'blurred vision', 'lesion at the base of the brain', 'compression of the optic chiasm', 'irregular periods', 'erectile dysfunction'
    ],
    'No Tumor': [
        'mild headache', 'no neurological deficits', 'MRI was unremarkable', 'routine check-up', 'no significant findings'
    ]
}

with open('clinical_summaries_5000.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['PatientID', 'Age', 'Gender', 'TumorType', 'ClinicalSummary'])
    for i in range(1, 5001):
        tumor = random.choice(tumor_types)
        age = random.randint(18, 85)
        gender = random.choice(['Male', 'Female'])
        summary = f"A {age}-year-old {gender.lower()} {random.choice(['presented with', 'experienced', 'reported', 'had'])} {random.choice(symptoms[tumor])}."
        if tumor != 'No Tumor':
            summary += f" MRI revealed {random.choice(symptoms[tumor])}."
        writer.writerow([i, age, gender, tumor, summary])