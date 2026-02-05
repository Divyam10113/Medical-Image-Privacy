import zipfile
import os

def zip_project():
    print("📦 Packaging project for Colab...")
    with zipfile.ZipFile('project.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add src/ folder
        for root, dirs, files in os.walk('src'):
            for file in files:
                # Skip __pycache__ and notebook files
                if '__pycache__' in root or file.endswith('.ipynb'):
                    continue
                file_path = os.path.join(root, file)
                zipf.write(file_path, arcname=file_path)
                print(f"  Added: {file_path}")
        
        # Add requirements.txt
        if os.path.exists('requirements.txt'):
            zipf.write('requirements.txt', arcname='requirements.txt')
            print("  Added: requirements.txt")
            
    print("✅ Created 'project.zip'. Upload this to Colab!")

if __name__ == "__main__":
    zip_project()
