import os

folder = r"c:\Users\Yuvaraj\Downloads\New_Projects\TNEA COLLEGE_Updation\TNEA COLLEGE_Updation"
files = ["college_details_generated.csv", "complete_engineering_colleges_dataset.csv"]

for file in files:
    filepath = os.path.join(folder, file)
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        content = content.replace("National Institute of Technology Trichy - Thuvakudi", "SRM UNIVERSITY (INSTITUTE OF SCIENCE & TECHNOLOGY)- TIRUCHIRAPPALLI")
        content = content.replace("National Institute of technology", "SRM UNIVERSITY (INSTITUTE OF SCIENCE & TECHNOLOGY)- TIRUCHIRAPPALLI")
        content = content.replace("National Institute of Technology", "SRM UNIVERSITY (INSTITUTE OF SCIENCE & TECHNOLOGY)- TIRUCHIRAPPALLI")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Replaced in {file}")
