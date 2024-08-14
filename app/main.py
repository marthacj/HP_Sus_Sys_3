import sys, os 
from sentence_transformers import SentenceTransformer
from manifest_generation import *
from LLM import *
import subprocess
from testing import *
from CSV_upload import *

questions = ["\n \n \n Can you tell me how much carbon emission is produced by machine ld71r18u44dws?\n", 
                "How much is the total carbon emissions for all the 8 machines?\n", 
                "Which machine has the GPU highest average utilisation?\n", 
                "Give me a summary of the central processing unit usage for all the machines\n",
                "Which of the machines do you recommend being moved to the next level up of compute power and why?\n",
                "What is the central processing unit average utilisation for each machine?\n",
                "What machine has the highest carbon emission value?\n"]


if __name__ == '__main__':

    # If the user does not input any file path, the default test file path will be used
    default_file_path = r"data\1038-0610-0614-day-larger-figures-test.xlsx"
    target_dir = r"data\uploaded_excel_files"
    uploaded_file_path = upload_file_to_application_directory(target_dir, default_file_path=default_file_path)

    # Check that the user hasn't quit, and/or that the file was uploaded correctly!
    if uploaded_file_path is None:
        print("Exiting the program with no successful file upload.")
        sys.exit()
    # Initial pipeline for Impact Framework
    # Define the input file path - need to work out hoow this will work if it's uploaded by the user
    excel_file = uploaded_file_path
    print(excel_file)


    # Convert the Excel file to a CSV file
    csv_file = convert_xlsx_to_csv(excel_file)

    # Define the input and output file paths
    original_CSV_filepath = csv_file
    modified_CSV_filepath = r'data\modified_CSV.csv'
    manifest_filepath = r'manifest1\z2_G4_Sci.yaml'

    # Process the CSV file and extract the duration value, start date, end date, and templates to create the manifest file
    modified_csv_path, duration, start_date, end_date, templates, analysis_window = process_csv(original_CSV_filepath, modified_CSV_filepath)
    # Generate the manifest file with the extracted duration value
    # generate_manifest(manifest_filepath, modified_csv_path, duration, templates)

    # print("\n\n★ ☆ ★ ☆ Generating manifest file with your data... ★ ☆ ★ ☆\n\n")

    # print(f"CSV file has been modified and saved as {modified_CSV_filepath}")
    # print(f"Manifest file has been created with the extracted duration value at {manifest_filepath}")
    # print(f"Extracted duration value: {duration}")

    # current_dir = os.getcwd()
    # print(f"Current working directory: {current_dir}")

    # # Construct absolute paths
    # manifest_path = os.path.join(current_dir, 'manifest1', 'z2_G4_Sci.yaml')
    # output_path = os.path.join(current_dir, 'manifest1', 'outputs', 'z2_G4_Sci_Output')
    try:
        manifest_success = safe_generate_manifest(manifest_filepath, modified_csv_path, duration, templates)

        print("\n\n★ ☆ ★ ☆ Attempting to generate manifest file with telemetry data... ★ ☆ ★ ☆\n\n")

        safe_print_file_info(modified_CSV_filepath, "Modified CSV file")
        safe_print_file_info(manifest_filepath, "Manifest file")

        if duration is not None:
            print(f"This telemetry data was observed over a period of: {duration} seconds")
        else:
            print("Warning: Duration value was not extracted successfully")

        current_dir = os.getcwd()
        print(f"Current working directory: {current_dir}")

        # Construct absolute paths
        manifest_path = os.path.abspath(os.path.join(current_dir, 'manifest1', 'z2_G4_Sci.yaml'))
        output_path = os.path.abspath(os.path.join(current_dir, 'manifest1', 'outputs', 'z2_G4_Sci_Output'))

        # Check if paths exist
        if os.path.exists(manifest_path):
            print(f"Manifest file found at: {manifest_path}")
        else:
            print(f"Warning: Manifest file not found at: {manifest_path}")

        if os.path.exists(output_path):
            print(f"Output directory found at: {output_path}")

        # If manifest generation was successful, try to read and print some info
        if manifest_success:
            try:
                with open(manifest_filepath, 'r') as f:
                    manifest_data = yaml.safe_load(f)
                print(f"Manifest file successfully read. Contains {len(manifest_data)} top-level keys.")
            except Exception as e:
                print(f"Warning: Could not read manifest file: {str(e)}")

    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        print("The program will continue running...")
        

    # Construct the command with absolute paths
    command = f'ie --manifest "{manifest_path}" --output "{output_path}"'
    # Run the terminal command
    # command = r"ie --manifest '\manifest1\NEW_z2_G4_Sci.yaml' --output '\manifest1\outputs\z2_G4_Sci_Output'"

    print("\n\n★ ☆ ★ ☆ Running Impact Framework command... ★ ☆ ★ ☆\n\n")
    try:
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        print("Command output:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the command: {e}")
        print("Error output:")
        print(e.stderr)
        sys.exit(1)

    # Pipeline to run after running the termimnal commmand to run the Impact Framework
    # taking in our raw 'uploaded.xlsx' file
    excel_file = excel_file
    # taking in the output yaml file with the carbon emissions data from IF
    yaml_file = r'manifest1\outputs\z2_G4_Sci_Output.yaml'

    # cleaned up exel file into a datafrarme ready to add relevant carbon emissions data
    prepared_df = prepare_excel_file(excel_file)
   
    # load the outputs from the manifest file
    emissions_reference_data = load_data_files(yaml_file)
    # extract only the data we intend to add to the dataframe for the LLM
    machine_emissions_list, machine_id_dict = extract_data_from_yaml(emissions_reference_data)

    # now add the carbon emissions data to the prepared dataframe
    merged_df, machine_ids = merge_data_into_one_df(prepared_df, machine_emissions_list, machine_id_dict)
    # Append the total carbon emissions row
    # merged_df = append_sum_row(merged_df, 'carbon emissions (gCO2eq)')
    # print(merged_df.columns)
    # Save the merged DataFrame to a CSV file 
    merged_df.to_csv(r'embeddings\merged_df.csv', index=False)
    # def round_floats(x):
    #     return round(x, 2) if isinstance(x, float) else x
    # merged_df = merged_df.applymap(round_floats)
    csv_filename = r'embeddings\merged_df.csv'

    # convert the csv file to a json file
    data_dict_json = csv_to_json(csv_filename, as_json=False)

    # Flatten the dictionary and stringify it for our sentences    
    flat_dict = flatten(data_dict_json)
    dict_string = stringify(flat_dict)

    with open(r'embeddings\data.txt', 'w') as f:
        f.write(dict_string)

    # Read the stringified flat dictionary from the file
    with open(r'embeddings\data.txt', 'r') as f:
        read_back_string = f.read()

    print("Stringified flat dictionary read back from the file:")
    # print(read_back_string)

    # Path to data.txt file
    sentences_file_path = r'embeddings\data.txt'

    # Read sentences from file
    sentences = read_sentences_from_file(sentences_file_path)
    add_context_to_sentences(sentences, duration, start_date, end_date, analysis_window, num_of_machines=str(len(machine_ids)))
    print(sentences)

    # Load the pre-trained model for embedding with SentenceTransformer
    model = SentenceTransformer('multi-qa-mpnet-base-cos-v1')

    # Embed the sentences using the model
    index, embeddings = embed_sentences(sentences, model)

    generate_question(index, embeddings, model, sentences, questions, machine_ids, model_name)
    

    

