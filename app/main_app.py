import pandas as pd
import yaml
import json
import csv
import sys, os 
import faiss
import numpy as np
import logging
from sentence_transformers import SentenceTransformer
import pickle
from llama_cpp import Llama
import ollama

questions = ["Can you tell me how much carbon emission is produced by machine ld71r18u44dws?", 
                "How much is the total carbon emissions for all the machines?", 
                "Which is the machine that uses GPU most intensively on average?", 
                "Give me a summary of the central processing unit usage for all the machines",
                "Which of the machines do you recommend being moved to the next level up of compute power and why?",
                "What is the central processing unit average utilisation for each machine?",
                "What machine has the highest carbon emission value?"]


def send_prompt(llm, prompt: str, interface: str = "ollama", 
                max_tokens: int = 1024) -> str:
    if interface == "ollama":
        response = ollama.generate(model="llama3",
                                   prompt=prompt, keep_alive='24h', options={'num_ctx': 16000})
        response = response['response']
    else: 
        # assume llama_cpp
        response = llm(prompt=prompt, max_tokens=max_tokens)
        response = response['choices'][0]['text']
    return response


def prepare_excel_file(excel_file):
    """funciton to take in excel file and preppare it for llm, adding empty carbon emissions column, filling model column, renaming columns and dropping unnecessary columns"""
    df = pd.read_excel(excel_file, sheet_name='WS-Data', skiprows=2)
    first_column_header = df.iloc[0, 0]
    second_column_header = df.iloc[0, 1]
    remaining_headers = df.iloc[1].tolist()
    # Combine headers
    headers = [first_column_header, second_column_header] + remaining_headers[2:]
    # Check if the length of headers matches the number of columns
    num_columns = len(df.columns)
    if len(headers) != num_columns:
        raise ValueError(f"Length mismatch: Expected {num_columns} columns, but got {len(headers)} headers.")
    df.insert(2, 'carbon emissions (g/co2)', pd.NA)  # You can initialize with pd.NA or any default value
    print(df['carbon emissions (g/co2)'])
    # Update headers to reflect the new column
    headers.insert(2, 'carbon emissions (g/Co2)')  # Insert 'carbon' into the correct position
    # Set the new headers
    df.columns = headers
    # Drop the first two rows which were used for headers
    df = df.drop([0, 1]).reset_index(drop=True)

    replace_dict = {
        '#Cores': 'number of cores (central processing unit % utilization)',
        'CPU\nHighest\navg': 'central processing unit average (central processing unit % utilization)',
        'GPU\navg': 'graphics processing unit average (NVIDIA % Utilization)',
        'Total MB\nSent': 'total MB sent (All Network Traffic)',
        'Total MB\nReceived': 'total MB received (All Network Traffic)',
        'GPU\n#oc > 80%': 'graphics processing unit number of occurrences over 80% (NVIDIA % utilization)',
        'Core\nHighest\nmax': 'core maximum (central processing unit % utilization)',
        'Core\nHighest\navg': 'core average (central processing unit % utilization)',
        'Core\n# oc > 80%': 'core number of occurrences over 80% (central processing unit % Utilization)',
        'Core \nTotal Seconds > 80%': 'core total seconds over 80% (central processing unit % utilization)',
        '\nCPU\nTotal Seconds > 80%': 'central processing unit total seconds over 80% (central processing unit % utilization)',
        'CPU\nHighest\nmax': 'central processing unit Maximum (central processing unit % utilization)',
        '\nCPU# oc > 80%': 'central processing unit number of occurrences over 80% (central processing unit % utilization)',
        'Total RAM\n(GB)': 'total RAM GB (memory % utilization)',
        'max': 'maximum (memory % utilization)',
        'avg': 'average (memory % utilization)',
        '#oc > 80%': 'number of occurrences over 80% (memory % utilization)',
        'GPU\nmin': 'graphics processing unit minimum (NVIDIA % utilization)', 
        'GPU\nmax': 'graphics processing unit maximum (NVIDIA % utilization)',
        'Host Name': 'Machine',
        'MEM\nmax': 'graphics processing unit memory maximum (NVIDIA % utilization)', 
        'MEM\navg': 'graphics processing unit memory average (NVIDIA % utilization)', 
        'MEM\n#oc > 80%': 'GPU memory number of occurrences over 80% (NVIDIA % utilization)',
        'Model': 'model',
        'Machine': 'machine'
    }

    df.rename(columns=replace_dict, inplace=True)
    df.loc[df['number of cores (central processing unit % utilization)'] == 24, 'model'] = 'z2 mini'
    df.loc[df['number of cores (central processing unit % utilization)'] == 28, 'model'] = 'Z4R G4'
    drop_names = [
        'Core\nHighest\nmin', 'Core \nTotal Time > 80%', 'CPU\nHighest\nmin', '\nCPU\n% Interval > 80%',
        '\nCPU\nTotal Time > 80%', 'send min\nMB/Sec', 'send max\nMB/Sec', 'send avg\nMB/Sec', 'receive min\nMB/Sec',
        'receive max\nMB/Sec', 'receive avg\nMB/Sec', 'tx min\nMB/Sec',
        'tx max\nMB/Sec', 'tx avg \nMB/Sec', 'Total MB\nsent', 'rx min\nMB/Sec', 'rx max\nMB/Sec', 'rx avg\nMB/Sec',
        'Total MB\nreceived', 'rx \n% packet loss\nmin', 'rx \n% packet loss\nmax', 'rx \n% packet loss\navg',
        'tx \n% packet loss\nmin', 'tx \n% packet loss\nmax', 'tx \n% packet loss\navg', 'Read MB\nmin', 'Read MB\nmax',
        'Read MB\navg', 'Write MB\nmin', 'Write MB\nmax', 'Write MB\navg', 'Read IOPs\nmin', 'Read IOPs\nmax',
        'Read IOPs\navg', 'Write IOPs\nmin', 'Write IOPs\nmax', 'Write IOPs\navg', 'Free MB\nmin', 'Free MB\nmax',
        'Free MB\navg', 'min', 'MEM\nmin'
    ]


    df.drop(columns=drop_names, inplace=True)
    # now drop the last three rows in the df
    df.drop(df.tail(3).index, inplace=True)


    """round the values in the column GPU average (NVIDIA % Utilization) to 3 decimal places"""
    df['graphics processing unit average (NVIDIA % Utilization)'] = df['graphics processing unit average (NVIDIA % Utilization)'].apply(lambda x: round(x, 3))
    df['central processing unit average (central processing unit % utilization)'] = df['central processing unit average (central processing unit % utilization)'].apply(lambda x: round(x, 3))
    df['core average (central processing unit % utilization)'] = df['core average (central processing unit % utilization)'].apply(lambda x: round(x, 3))
    df['core maximum (central processing unit % utilization)'] = df['core maximum (central processing unit % utilization)'].apply(lambda x: round(x, 3))
    df['central processing unit Maximum (central processing unit % utilization)'] = df['central processing unit Maximum (central processing unit % utilization)'].apply(lambda x: round(x, 3))
    
    return df


def load_data_files(yaml_file, return_yaml: bool = False) -> tuple:
    # Load yaml file
    with open(yaml_file, 'r') as f:
        """this data is what is put in <BACKGROUND> tag in the prompt"""
        emissions_reference_data = yaml.load(f, Loader=yaml.SafeLoader)
        emissions_reference_data_str = yaml.dump(emissions_reference_data)
        # split by first occureance of word defaults and take [1]
        emissions_reference_data_str = emissions_reference_data_str.split('pipeline', 1)[1]

    return emissions_reference_data
    # return emissions_reference_data_str


def extract_data_from_yaml(yaml_data: yaml) -> tuple[dict, dict]:
    """
    yaml structure is:
    tree:
        children:
            child:
                children:
    """
    """iterate through the bottom children and extract the data"""
    machine_emissions_list = []
    machine_id_dict = {}
    """machine_dict = {}"""
    lowest_children_level = yaml_data['tree']['children']['child']['children']
    lowest_children_level.update(yaml_data['tree']['children']['child']['children'])
    """dump the yaml to file for debug"""
    with open('yaml_dump.txt', 'w') as f:
        yaml.dump(lowest_children_level, f)
    for i, machine in enumerate(lowest_children_level):
        for child in lowest_children_level[machine]['outputs']:
            """convert child to a dictionary"""
            child = dict(child)
            """only pull out values for keys timestamp, instance-type, sci """
            child = {k: v for k, v in child.items() if k in ['timestamp', 'instance-type', 'sci', 'carbon-embodied', 'carbon-operational', 'duration', 'carbon']}
            """ convert timestamp to a UTC datetime not an object"""
            child['timestamp'] = pd.to_datetime(child['timestamp'], utc=True)
            """convert that to a string"""
            child['timestamp'] = child['timestamp'].strftime('%Y-%m-%d')
            """round sci to 6 dp"""
            child['sci'] = round(child['sci'], 3)
            child['carbon'] = round(child['carbon'], 3)
            """pull out value for carbon embodied, carbon operational, and duration"""
            child['carbon-embodied'] = round(child['carbon-embodied'], 3)
            child['carbon-operational'] = round(child['carbon-operational'], 3)
            child['duration'] = child['duration']
            """letters 7 to 10 are unique to each machine"""
            # child['machine-id'] = str(i)
            # machine_id_dict[machine[6:]] = str(i)
            machine_id_dict[machine] = str(i)
            """replace instance-type with machine-family, and sci with machine-carbon-emission-value"""
            child['machine-family'] = child.pop('instance-type')
            # child['machine-carbon-emission-value'] = child.pop('carbon')
            """machine_dict['machine-id-'+machine] = child"""
            machine_emissions_list.append(child)
        
    return machine_emissions_list, machine_id_dict


def merge_data_into_one_df(prepared_df, machine_emissions_list, machine_id_dict):
    machine_ids = list(machine_id_dict.keys())

    # Update machine_emissions_list to include machine IDs
    for idx, item in enumerate(machine_emissions_list):
        if idx < len(machine_ids):
            item['machine'] = machine_ids[idx]
    for machine in machine_emissions_list:
        print(machine_emissions_list)
        for i in range(len(machine_ids)):
            if prepared_df.loc[i, 'Machine'] == machine['machine']:
                prepared_df.loc[i, 'carbon emissions (g/Co2)'] = machine['carbon']
                print(prepared_df.loc[i, 'carbon emissions (g/Co2)'])
    """if there is no column called duration, add it to the dataframe  and fill it with the duration value"""
    if 'duration' not in prepared_df.columns:
        prepared_df.insert(2, 'duration (seconds)', pd.NA)
    for machine in machine_emissions_list:
        for i in range(len(machine_ids)):
            if prepared_df.loc[i, 'Machine'] == machine['machine']:
                prepared_df.at[i, 'duration (seconds)'] = machine['duration']
    if 'embodied carbon (gCO2)' not in prepared_df.columns:
        prepared_df.insert(2, 'embodied carbon (gCO2)', pd.NA)
    for machine in machine_emissions_list:
        for i in range(len(machine_ids)):
            if prepared_df.loc[i, 'Machine'] == machine['machine']:
                prepared_df.loc[i, 'embodied carbon (gCO2)'] = machine['carbon-embodied']  
                print('HERE:', prepared_df.loc[i, 'embodied carbon (gCO2)']) 
    if 'operational carbon (gCO2)' not in prepared_df.columns:
        prepared_df.insert(2, 'operational carbon (gCO2)', pd.NA)
    for machine in machine_emissions_list:
        for i in range(len(machine_ids)):
            if prepared_df.loc[i, 'Machine'] == machine['machine']:
                prepared_df.loc[i, 'operational carbon (gCO2)'] = machine['carbon-operational'] 
    """do the same for timestamp"""
    if 'timestamp' not in prepared_df.columns:
        prepared_df.insert(2, 'timestamp', pd.NA)
    for machine in machine_emissions_list:
        for i in range(len(machine_ids)):
            if prepared_df.loc[i, 'Machine'] == machine['machine']:
                prepared_df.loc[i, 'timestamp'] = machine['timestamp']

    return prepared_df


def append_sum_row(df, column_name, label='total carbon emissions in g/co2'):
    """
    Sum the values in the specified column and append a new row to the DataFrame
    with the sum and a label.

    Parameters:
    df (pd.DataFrame): The DataFrame to modify.
    column_name (str): The name of the column to sum.
    label (str): The label to use for the new row. Default is 'total'.

    Returns:
    pd.DataFrame: The DataFrame with the new row appended.
    """
    # Calculate the sum of the specified column
    total_value = df[column_name].sum()

    # Create a new row with None values for all columns
    new_row = {col: None for col in df.columns}

    # Set the label in the first column and the sum in the specified column
    new_row[df.columns[0]] = label
    new_row[column_name] = total_value

    # Append the new row to the original DataFrame
    df = df._append(new_row, ignore_index=True)

    return df


def csv_to_json(csv_filename, as_json=True):
    """Convert the csv filenmae into a json file"""
    data_dict = {}
    with open(csv_filename, mode='r') as file:
        csvFile = csv.reader(file)
        headers = next(csvFile)
        for row in csvFile:
            data_dict[row[0]] = {headers[i]: row[i] for i in range(1, len(headers))}
    if as_json:
        with open('data_dict.json', 'w') as f:
            json.dump(data_dict, f)
    print('data_dict:', data_dict)
    return data_dict


def flatten(data_dict):
    flat_dict = {}
    for machine_key, machine_data in data_dict.items():
        machine_key = machine_key[6:]
        for top_level_key, top_level_data in machine_data.items():
            if isinstance(top_level_data, dict):
                for lower_level_key, lower_level_data in top_level_data.items():
                    flat_dict[machine_key + '_' + top_level_key + '_' + lower_level_key] = lower_level_data
            else:
                flat_dict[machine_key + '_' + top_level_key] = top_level_data
    return flat_dict


def stringify(flat_dict):
    dict_string = ""
    for k,v in flat_dict.items():
        dict_string += k.replace('_',' ') + " = " + v + "\n"
    # Changed to dict_string return not flat_dict
    
    dict_string = deabbreviate(dict_string)
    print('dict_string:', dict_string)
    return dict_string


def deabbreviate(sentence: str) -> str:
    abbr_list = {' CPU ': ' central processing unit ', 'GPU': 'graphics processing unit', '%':'Percent ', 'Mem ':'Memory ', 'min ':'minimum', 'max ':'maximum ', 'avg ':'average ', '#': ' number of ', 'mb ':'megabytes ',
                 'oc ': 'occurrences '} 
    sentence = sentence.lower()
    for abbr, full in abbr_list.items():
        sentence = sentence.replace(abbr.lower(), full.lower())
    return sentence


def read_sentences_from_file(sentences_file_path):
    with open(sentences_file_path, 'r') as file:
        sentences = file.readlines()
    # Strip any surrounding whitespace from each sentence
    sentences = [sentence.strip() for sentence in sentences]
    return sentences


def embed_sentences(sentences, model):
    try:
    # sentences += ['CPU','cpu','CPU CPU','cpu cpu','something cpu','something cpu something','CPU CPU CPU','cpu cpu cpu','CPU CPU CPU CPU','cpu cpu cpu cpu','CPU CPU CPU CPU CPU something','something cpu cpu cpu cpu cpu','CPU CPU CPU CPU CPU CPU','cpu cpu cpu cpu cpu cpu','CPU CPU CPU CPU CPU CPU CPU','cpu cpu cpu cpu cpu cpu cpu','CPU CPU CPU CPU CPU CPU CPU CPU','cpu cpu cpu cpu cpu cpu cpu cpu','CPU CPU CPU CPU CPU CPU CPU CPU CPU','cpu cpu cpu cpu cpu cpu cpu cpu cpu']
        """if embeddings.pickle exists, load the embeddings from file  and skip the encoding step"""
        if os.path.exists('embeddings.pickle'):   
            # Load the embeddings from file
            with open('embeddings.pickle', 'rb') as file:
                embeddings = pickle.load(file)
                rebuild_faiss_index = False
        else:
            # Encode sentences to get their embeddings
            embeddings = model.encode(sentences)
            rebuild_faiss_index = True
            
            """save the encodings to pickle file"""
            with open('embeddings.pickle', 'wb') as file:
                pickle.dump(embeddings, file)
    # Convert embeddings to a numpy array
        embeddings = np.array(embeddings)

        # Create a FAISS index
        d = embeddings.shape[1]  # Dimension of embeddings (768)
        index = faiss.IndexFlatL2(d)
        
        """if faiss_index.bin exists, load the index from file and skip the add step"""
        if os.path.exists('faiss_index.bin') and not rebuild_faiss_index:
            index = faiss.read_index('faiss_index.bin')
            print("FAISS index loaded from 'faiss_index.bin'")
        else:
            # Add embeddings to the index
            index.add(embeddings)
            print(f"Number of sentences indexed: {index.ntotal}")
            # Save the FAISS index to disk
            faiss.write_index(index, 'faiss_index.bin')
            print("FAISS index saved to 'faiss_index.bin'")
        # Print the embeddings
        # for sentence, embedding in zip(sentences, embeddings):
        #     print(f"Sentence: {sentence}")
        #     print(f"Embedding: {embedding}\n")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    return index, embeddings


def generate_question(index, embeddings, model, sentences, questions):
    try:
        q = questions[6]
        q_embedding = model.encode(q)
        q_embedding = q_embedding.reshape(1, -1)
        
        # Calculate top_k based on 25% of the number of sentences
        top_k = int(0.25 * len(sentences))
        distances, indices = index.search(q_embedding, top_k)
        
        prompt = "Here is your context for a question I will ask you:\n"
        for ind in indices[0]:
            prompt += f"{sentences[ind]}\n"
        
        prompt += f"Here is a new question for you to answer:\n{q}"
        print("prompt:", prompt)
        
        response = send_prompt(llm, prompt, interface="ollama")
        print(response)
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")


if __name__ == '__main__':

    model_path = r"C:\Users\martha.calder-jones\OneDrive - University College London\UCL_comp_sci\Sustainable_Systems_3\HP_Sus_Sys_3\models\Meta-Llama-3-8B-Instruct.Q5_0.gguf"

    llm = Llama(
        model_path=model_path,
#         temperature=0.1,
          n_ctx=16000,
          n_gpu_layers=-1,
          verbose=True,
)
    
    # taking in our raw 'uploaded xlsx file
    excel_file = r'data\1038-0610-0614-evening.xlsx'
    # taking in the output yaml file with the carbon emissions data from IF
    yaml_file = r'C:\Users\martha.calder-jones\OneDrive - University College London\UCL_comp_sci\Sustainable_Systems_3\HP_Sus_Sys_3\manifest1\outputs\z2_G4_Sci_Output.yaml'

    # cleaned up exel file into a datafrarme ready to add relevant carbon emissions data
    prepared_df = prepare_excel_file(excel_file)

    # load the outputs from the manifest file
    emissions_reference_data = load_data_files(yaml_file)
    # extract only the data we intend to add to the dataframe for the LLM
    machine_emissions_list, machine_id_dict = extract_data_from_yaml(emissions_reference_data)

    # now add the carbon emissions data to the prepared dataframe
    merged_df = merge_data_into_one_df(prepared_df, machine_emissions_list, machine_id_dict)
    
    # Append the total carbon emissions row
    merged_df = append_sum_row(merged_df, 'carbon emissions (g/Co2)')

    # Save the merged DataFrame to a CSV file 
    merged_df.to_csv('merged_df.csv', index=False)

    csv_filename = r'C:\Users\martha.calder-jones\OneDrive - University College London\UCL_comp_sci\Sustainable_Systems_3\HP_Sus_Sys_3\merged_df.csv'

    # convert the csv file to a json file
    data_dict_json = csv_to_json(csv_filename, as_json=False)

    # Flatten the dictionary and stringify it for our sentences    
    flat_dict = flatten(data_dict_json)
    dict_string = stringify(flat_dict)

    with open('data.txt', 'w') as f:
        f.write(dict_string)

    # Read the stringified flat dictionary from the file
    with open('data.txt', 'r') as f:
        read_back_string = f.read()

    print("Stringified flat dictionary read back from the file:")
    print(read_back_string)

    # Path to data.txt file
    sentences_file_path = r'C:\Users\martha.calder-jones\OneDrive - University College London\UCL_comp_sci\Sustainable_Systems_3\HP_Sus_Sys_3\data.txt'

    # Read sentences from file
    sentences = read_sentences_from_file(sentences_file_path)
    print(sentences)

    # Load the pre-trained model for embedding with SentenceTransformer
    model = SentenceTransformer('multi-qa-mpnet-base-cos-v1')

    # Embed the sentences using the model
    index, embeddings = embed_sentences(sentences, model)

    generate_question(index, embeddings, model, sentences, questions)
