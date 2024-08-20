import os
import shutil


def upload_file_to_application_directory(target_dir, default_file_path):
    """
    Prompts the user for the file location and copies it to the target directory with a fixed name 'uploaded_file.xlsx'.
    
    Args:
    target_dir (str): The directory where the file should be copied to.
    """
    
    # Ensure the target directory exists
    if not os.path.exists(target_dir):
        try:
            os.makedirs(target_dir)
            print(f"Created target directory: {target_dir}")
        except OSError as e:
            print(f"Failed to create target directory: {e}")
            return None

    max_attempts = 30000
    attempts = 0

    while attempts < max_attempts:
        # Prompt the user for the file path
        file_path = input("Enter the full path of the XLSX file you want to upload (or press Enter to use the default file, or 'q' to quit): ").strip()

        if file_path.lower() == 'q':
            print("Exiting the process.")
            return None
            
        if file_path == "":
            file_path = default_file_path
            if os.path.isfile(default_file_path):
                print(f"Using default file: {default_file_path}")
            else:
                print(f"Default file not found: {default_file_path}")
                return None
            
        # Check if the file exists and is an XLSX file
        if os.path.isfile(file_path) and file_path.lower().endswith('.xlsx'):
            try:
                # Before copying, check if we have read access to the file
                if not os.access(file_path, os.R_OK):
                    print(f"Read access denied for {file_path}. Exiting.")
                    return None
                
                # Define the fixed destination path
                destination_path = os.path.join(target_dir, 'uploaded_file.xlsx')
                
                # Copy the file to the target directory with the fixed name
                shutil.copy(file_path, destination_path)
                print(f"File uploaded successfully to {destination_path}")
                
                # Confirm read access to the copied file
                if os.access(destination_path, os.R_OK):
                    print(f"Read access confirmed for {destination_path}")
                    return destination_path
                else:
                    print(f"Read access denied for {destination_path}. Exiting.")
                    # Remove the copied file since read access failed
                    os.remove(destination_path)
                    return None
            except PermissionError as e:
                print(f"Could not read file correctly, please close the file: {e}")
                attempts += 1
            except Exception as e:
                print(f"Failed to copy the file: {e}")
                attempts += 1
        else:
            print("Invalid file path or not an XLSX file. Please try again.")
            attempts += 1

    print("Maximum attempts reached. Exiting.")
    return None
