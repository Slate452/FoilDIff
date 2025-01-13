import zipfile
import os

def unzip_all_in_directory(directory):
    # Ensure the directory exists
    if not os.path.exists(directory):
        print(f"The directory {directory} does not exist.")
        return

    # List all files in the directory
    files = os.listdir(directory)

    # Filter out only zip files
    zip_files = [file for file in files if file.endswith('.zip')]

    # Iterate through all zip files and extract them
    for zip_file in zip_files:
        zip_path = os.path.join(directory, zip_file)
        extract_path = "C:\Users\kenec\Desktop\FoilDIff\datasets\1_parameter\results"
        # Create a directory for the extracted contents
        if not os.path.exists(extract_path):
            os.makedirs(extract_path)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
            print(f"Extracted {zip_file} to {extract_path}")



directory_to_unzip = "C:\Users\kenec\Desktop\FoilDIff\datasets\1_parameter"
unzip_all_in_directory(directory_to_unzip)