import sys, os 
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from manifest_generation import *
from LLM import *
import subprocess

questions = ["\n \n \n [0] Can you tell me how much carbon emission is produced by machine ld71r18u44dws?\n", 
                "[1] How much is the total carbon emissions for all the machines?\n", 
                "[2] Which is the machine that uses GPU most intensively on average?\n", 
                "[3] Give me a summary of the central processing unit usage for all the machines\n",
                "[4] Which of the machines do you recommend being moved to the next level up of compute power and why?\n",
                "[5] What is the central processing unit average utilisation for each machine?\n",
                "[6] What machine has the highest carbon emission value?\n"]


if __name__ == '__main__':

    # Initial pipeline for Impact Framework
    # Define the input file path - need to work out hoow this will work if it's uploaded by the user
    excel_file = r'data\1038-0610-0614-evening.xlsx'

    # Convert the Excel file to a CSV file
    csv_file = convert_xlsx_to_csv(excel_file)

    # Define the input and output file paths
    original_CSV_filepath = csv_file
    modified_CSV_filepath = r'data\modified_CSV1038-0610-0614-day.csv'
    manifest_filepath = r'manifest1\NEW_z2_G4_Sci.yaml'

    # Process the CSV file and extract the duration value, start date, end date, and templates to create the manifest file
    modified_csv_path, duration, start_date, end_date, templates = process_csv(original_CSV_filepath, modified_CSV_filepath)
    # Generate the manifest file with the extracted duration value
    generate_manifest(manifest_filepath, modified_csv_path, duration, templates)

    print("\n\n★ ☆ ★ ☆ Generating manifest file with your data... ★ ☆ ★ ☆\n\n")

    print(f"CSV file has been modified and saved as {modified_CSV_filepath}")
    print(f"Manifest file has been created with the extracted duration value at {manifest_filepath}")
    print(f"Extracted duration value: {duration}")

    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")

    # Construct absolute paths
    manifest_path = os.path.join(current_dir, 'manifest1', 'NEW_z2_G4_Sci.yaml')
    output_path = os.path.join(current_dir, 'manifest1', 'outputs', 'z2_G4_Sci_Output')

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
    model_path = r"models\Meta-Llama-3-8B-Instruct.Q5_0.gguf"

    llm = Llama(
        model_path=model_path,
#         temperature=0.1,
          n_ctx=16000,
          n_gpu_layers=-1,
          verbose=True,
)
    llm.close() 
    # taking in our raw 'uploaded xlsx file
    excel_file = r'data\1038-0610-0614-day-larger-figures-test.xlsx'
    # taking in the output yaml file with the carbon emissions data from IF
    yaml_file = r'manifest1\outputs\z2_G4_Sci_Output.yaml'

    # cleaned up exel file into a datafrarme ready to add relevant carbon emissions data
    prepared_df = prepare_excel_file(excel_file)
   
    # load the outputs from the manifest file
    emissions_reference_data = load_data_files(yaml_file)
    # extract only the data we intend to add to the dataframe for the LLM
    machine_emissions_list, machine_id_dict = extract_data_from_yaml(emissions_reference_data)

    # now add the carbon emissions data to the prepared dataframe
    merged_df = merge_data_into_one_df(prepared_df, machine_emissions_list, machine_id_dict)
    # Append the total carbon emissions row
    # merged_df = append_sum_row(merged_df, 'carbon emissions (gCO2eq)')
    print(merged_df.columns)
    # Save the merged DataFrame to a CSV file 
    merged_df.to_csv('embeddings\merged_df.csv', index=False)
    # def round_floats(x):
    #     return round(x, 2) if isinstance(x, float) else x
    # merged_df = merged_df.applymap(round_floats)
    csv_filename = r'embeddings\merged_df.csv'

    # convert the csv file to a json file
    data_dict_json = csv_to_json(csv_filename, as_json=False)

    # Flatten the dictionary and stringify it for our sentences    
    flat_dict = flatten(data_dict_json)
    dict_string = stringify(flat_dict)

    with open('embeddings\data.txt', 'w') as f:
        f.write(dict_string)

    # Read the stringified flat dictionary from the file
    with open('embeddings\data.txt', 'r') as f:
        read_back_string = f.read()

    print("Stringified flat dictionary read back from the file:")
    # print(read_back_string)

    # Path to data.txt file
    sentences_file_path = r'embeddings\data.txt'

    # Read sentences from file
    sentences = read_sentences_from_file(sentences_file_path)
    # print(sentences)

    # Load the pre-trained model for embedding with SentenceTransformer
    model = SentenceTransformer('multi-qa-mpnet-base-cos-v1')

    # Embed the sentences using the model
    index, embeddings = embed_sentences(sentences, model)

    generate_question(index, embeddings, model, sentences, questions)
