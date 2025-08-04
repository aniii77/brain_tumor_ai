import os

# Folders where __init__.py should be added
folders_to_initialize = [
    "utility",
    "utility/utils",
    "utility/utils/data",
    "utility/utils/data/database",
]

for folder in folders_to_initialize:
    init_path = os.path.join(folder, "__init__.py")
    if not os.path.exists(init_path):
        os.makedirs(folder, exist_ok=True)
        with open(init_path, "w") as f:
            f.write("# Makes this folder a Python package\n")
        print(f"✅ Created: {init_path}")
    else:
        print(f"✔️ Already exists: {init_path}")

print("🎉 All __init__.py files are set up.")
