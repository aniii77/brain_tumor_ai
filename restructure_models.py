import os
import shutil

# Root of your project (where app.py is)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FINAL_MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Create final models/ folder if not exists
os.makedirs(FINAL_MODELS_DIR, exist_ok=True)

# Deep nested folder you currently have
NESTED_MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "models", "model", "models")

# List of files to move
files_to_move = ["tumor_detector.py", "mri_recommender.py", "medical_nlp.py"]

moved = []
skipped = []

for file in files_to_move:
    found_path = None

    # Check deeply nested path
    candidate_path = os.path.join(NESTED_MODELS_DIR, file)
    if os.path.isfile(candidate_path):
        found_path = candidate_path
    else:
        # Also check intermediate levels (optional fallback)
        for root, _, files in os.walk(os.path.join(PROJECT_ROOT, "models")):
            if file in files:
                found_path = os.path.join(root, file)
                break

    if found_path:
        dest_path = os.path.join(FINAL_MODELS_DIR, file)
        shutil.move(found_path, dest_path)
        moved.append(file)
    else:
        skipped.append(file)

# Optional: clean up empty folders
def remove_empty_dirs(path):
    for root, dirs, _ in os.walk(path, topdown=False):
        for d in dirs:
            full_path = os.path.join(root, d)
            try:
                os.rmdir(full_path)
            except OSError:
                pass  # Not empty

remove_empty_dirs(os.path.join(PROJECT_ROOT, "models"))

# Report
print("✅ Moved files:", moved)
if skipped:
    print("⚠️ Skipped (not found):", skipped)
print("🎉 Done! Check your models/ folder.")
