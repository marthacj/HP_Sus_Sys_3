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

model_path = r"models\Meta-Llama-3-8B-Instruct.Q5_0.gguf"
llm = Llama(
    model_path=model_path,
#         temperature=0.1,
        n_ctx=16000,
        n_gpu_layers=-1,
        verbose=True,
)

def send_prompt(llm, prompt: str, interface: str = "ollama", 
                max_tokens: int = 1024) -> str:
    if interface == "ollama":
        response = ollama.generate(model="mistral:latest",
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
    df.insert(2, 'carbon emissions (gCO2eq)', pd.NA)  # You can initialize with pd.NA or any default value
    # Update headers to reflect the new column
    headers.insert(2, 'carbon emissions (gCO2eq)')  # Insert 'carbon' into the correct position
    # Set the new headers
    df.columns = headers
    # Drop the first two rows which were used for headers
    df = df.drop([0, 1]).reset_index(drop=True)

    replace_dict = {
        '#Cores': 'number of cores',
        'CPU\nHighest\navg': 'central processing unit average (central processing unit % utilization)',
        'GPU\navg': 'graphics processing unit average (NVIDIA % Utilization)',
        'Total MB\nSent': 'total MB sent (All Network Traffic)',
        'Total MB\nReceived': 'total MB received (All Network Traffic)',
        'GPU\n#oc > 80%': 'graphics processing unit number of occurrences over 80% (NVIDIA % utilization)',
        'Core\nHighest\nmax': 'core maximum %',
        'Core\nHighest\navg': 'core average %',
        'Core\n# oc > 80%': 'core number of occurrences over 80%',
        'Core \nTotal Seconds > 80%': 'core total seconds over 80%',
        '\nCPU\nTotal Seconds > 80%': 'central processing unit total seconds over 80% (central processing unit % utilization)',
        'CPU\nHighest\nmax': 'central processing unit Maximum (central processing unit % utilization)',
        '\nCPU# oc > 80%': 'central processing unit number of occurrences over 80% (central processing unit % utilization)',
        'Total RAM\n(GB)': 'total RAM GB (memory % utilization)',
        'max': 'maximum (memory % utilization)',
        'avg': 'average (memory % utilization)',
        '#oc > 80%': 'number of occurrences over 80% (gpu memory % utilization)',
        # 'GPU\nmin': 'graphics processing unit minimum (NVIDIA % utilization)', 
        'GPU\nmax': 'graphics processing unit maximum (NVIDIA gpu % utilization)',
        'Host Name': 'Machine',
        # 'MEM\nmax': 'graphics processing unit memory maximum (NVIDIA gpu % utilization)', 
        # 'MEM\navg': 'graphics processing unit memory average (NVIDIA gpu % utilization)', 
        # 'MEM\n#oc > 80%': 'GPU memory number of occurrences over 80% (NVIDIA % utilization)',
        'Model': 'model',
        'Machine': 'machine'
    }

    df.rename(columns=replace_dict, inplace=True)
    df.loc[df['number of cores'] == 24, 'model'] = 'z2 mini'
    df.loc[df['number of cores'] == 28, 'model'] = 'Z4R G4'
    drop_names = [
        'Core\nHighest\nmin', 'Core \nTotal Time > 80%', 'CPU\nHighest\nmin', '\nCPU\n% Interval > 80%',
        '\nCPU\nTotal Time > 80%', 'send min\nMB/Sec', 'send max\nMB/Sec', 'send avg\nMB/Sec', 'receive min\nMB/Sec',
        'receive max\nMB/Sec', 'receive avg\nMB/Sec', 'tx min\nMB/Sec',
        'tx max\nMB/Sec', 'tx avg \nMB/Sec', 'Total MB\nsent', 'rx min\nMB/Sec', 'rx max\nMB/Sec', 'rx avg\nMB/Sec',
        'Total MB\nreceived', 'rx \n% packet loss\nmin', 'rx \n% packet loss\nmax', 'rx \n% packet loss\navg',
        'tx \n% packet loss\nmin', 'tx \n% packet loss\nmax', 'tx \n% packet loss\navg', 'Read MB\nmin', 'Read MB\nmax',
        'Read MB\navg', 'Write MB\nmin', 'Write MB\nmax', 'Write MB\navg', 'Read IOPs\nmin', 'Read IOPs\nmax',
        'Read IOPs\navg', 'Write IOPs\nmin', 'Write IOPs\nmax', 'Write IOPs\navg', 'Free MB\nmin', 'Free MB\nmax',
        'Free MB\navg', 'min', 'MEM\nmin', 'GPU\nmin', 'MEM\n#oc > 80%', 'MEM\nmax', 'MEM\navg'
    ]

    # Drop the columns that are not needed 
    df.drop(columns=drop_names, inplace=True)
    # now drop the last three rows in the df
    df.drop(df.tail(3).index, inplace=True)

    # for testing with all columns, for fairness must also round those cols
    # df[drop_names] = df[drop_names].apply(lambda x: round(x, 3))



    """round the values in the column GPU average (NVIDIA % Utilization) to 3 decimal places"""
    df['graphics processing unit average (NVIDIA % Utilization)'] = df['graphics processing unit average (NVIDIA % Utilization)'].apply(lambda x: round(x, 3))
    df['central processing unit average (central processing unit % utilization)'] = df['central processing unit average (central processing unit % utilization)'].apply(lambda x: round(x, 3))
    df['core average %'] = df['core average %'].apply(lambda x: round(x, 3))
    df['core maximum %)'] = df['core maximum %'].apply(lambda x: round(x, 3))
    df['central processing unit Maximum (central processing unit % utilization)'] = df['central processing unit Maximum (central processing unit % utilization)'].apply(lambda x: round(x, 3))
    df['total MB sent (All Network Traffic)'] = df['total MB sent (All Network Traffic)'].apply(lambda x: round(x, 3))
    df['total MB received (All Network Traffic)'] = df['total MB received (All Network Traffic)'].apply(lambda x: round(x, 3))
    df['total RAM GB (memory % utilization)'] = df['total RAM GB (memory % utilization)'].apply(lambda x: round(x, 3))
    df['maximum (memory % utilization)'] = df['maximum (memory % utilization)'].apply(lambda x: round(x, 3))
    df['average (memory % utilization)'] = df['average (memory % utilization)'].apply(lambda x: round(x, 3))
    # df['graphics processing unit minimum (NVIDIA % utilization)'] = df['graphics processing unit minimum (NVIDIA % utilization)'].apply(lambda x: round(x, 3))
    df['graphics processing unit maximum (NVIDIA gpu % utilization)'] = df['graphics processing unit maximum (NVIDIA gpu % utilization)'].apply(lambda x: round(x, 3))
    # df['graphics processing unit memory maximum (NVIDIA % utilization)'] = df['graphics processing unit memory maximum (NVIDIA % utilization)'].apply(lambda x: round(x, 3))
    # df['graphics processing unit memory average (NVIDIA % utilization)'] = df['graphics processing unit memory average (NVIDIA % utilization)'].apply(lambda x: round(x, 3))
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
    with open('embeddings\yaml_dump.txt', 'w') as f:
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
            child['sci'] = round(child['sci'], 2)
            child['carbon'] = round(child['carbon'], 2)
            """pull out value for carbon embodied, carbon operational, and duration"""
            child['carbon-embodied'] = round(child['carbon-embodied'], 2)
            child['carbon-operational'] = round(child['carbon-operational'], 2)
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
        for i in range(len(machine_ids)):
            if prepared_df.loc[i, 'Machine'] == machine['machine']:
                prepared_df.loc[i, 'carbon emissions (gCO2eq)'] = machine['carbon']
    """if there is no column called duration, add it to the dataframe  and fill it with the duration value"""
    if 'duration' not in prepared_df.columns:
        prepared_df.insert(2, 'duration (seconds)', pd.NA)
    for machine in machine_emissions_list:
        for i in range(len(machine_ids)):
            if prepared_df.loc[i, 'Machine'] == machine['machine']:
                prepared_df.at[i, 'duration (seconds)'] = machine['duration']
    if 'embodied carbon (gCO2eq)' not in prepared_df.columns:
        prepared_df.insert(2, 'embodied carbon (gCO2eq)', pd.NA)
    for machine in machine_emissions_list:
        for i in range(len(machine_ids)):
            if prepared_df.loc[i, 'Machine'] == machine['machine']:
                prepared_df.loc[i, 'embodied carbon (gCO2eq)'] = machine['carbon-embodied']  
    if 'operational carbon (gCO2eq)' not in prepared_df.columns:
        prepared_df.insert(2, 'operational carbon (gCO2eq)', pd.NA)
    for machine in machine_emissions_list:
        for i in range(len(machine_ids)):
            if prepared_df.loc[i, 'Machine'] == machine['machine']:
                prepared_df.loc[i, 'operational carbon (gCO2eq)'] = machine['carbon-operational'] 
    """do the same for timestamp"""
    if 'timestamp' not in prepared_df.columns:
        prepared_df.insert(2, 'timestamp', pd.NA)
    for machine in machine_emissions_list:
        for i in range(len(machine_ids)):
            if prepared_df.loc[i, 'Machine'] == machine['machine']:
                prepared_df.loc[i, 'timestamp'] = machine['timestamp']
    return prepared_df


def append_sum_row(df, column_name, label='total carbon emissions in gCO2eq'):
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
    
        # Find the index of the 'carbon emissions (gCO2eq)' column
        carbon_emissions_index = headers.index('carbon emissions (gCO2eq)')
        
        for row in csvFile:
            row_key = row[0]
            
            if row_key == 'total carbon emissions in gCO2eq':
                # Include only the 'carbon emissions (gCO2eq)' column if it's non-zero
                if row[carbon_emissions_index] != '0':
                    data_dict[row_key] = {headers[carbon_emissions_index]: row[carbon_emissions_index]}
            else:
                # Include all columns
                data_dict[row_key] = {headers[i]: row[i] for i in range(1, len(headers))}

    if as_json:
        with open('data_dict.json', 'w') as f:
            json.dump(data_dict, f)

    return data_dict


def flatten(data_dict):
    flat_dict = {}
    for machine_key, machine_data in data_dict.items():
        # machine_key = machine_key[6:]
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
        if os.path.exists('embeddings\embeddings.pickle'):   
            # Load the embeddings from file
            with open('embeddings\embeddings.pickle', 'rb') as file:
                embeddings = pickle.load(file)
                rebuild_faiss_index = False
        else:
            # Encode sentences to get their embeddings
            embeddings = model.encode(sentences)
            rebuild_faiss_index = True
            
            """save the encodings to pickle file"""
            with open('embeddings\embeddings.pickle', 'wb') as file:
                pickle.dump(embeddings, file)
    # Convert embeddings to a numpy array
        embeddings = np.array(embeddings)

        # Create a FAISS index
        d = embeddings.shape[1]  # Dimension of embeddings (768)
        index = faiss.IndexFlatL2(d)
        
        """if faiss_index.bin exists, load the index from file and skip the add step"""
        if os.path.exists('faiss_index.bin') and not rebuild_faiss_index:
            index = faiss.read_index('faiss_index.bin')
            print("\n\nFAISS index loaded from 'faiss_index.bin'")
        else:
            # Add embeddings to the index
            index.add(embeddings)
            print(f"Number of sentences indexed: {index.ntotal}")
            # Save the FAISS index to disk
            faiss.write_index(index, 'faiss_index.bin')
            print("FAISS index saved to 'faiss_index.bin'\n\n")
        # Print the embeddings
        # for sentence, embedding in zip(sentences, embeddings):
        #     print(f"Sentence: {sentence}")
        #     print(f"Embedding: {embedding}\n")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    return index, embeddings


def generate_question(index, embeddings, model, sentences, questions):
    while True:
        for i, question in enumerate(questions):
            print(f"{i}: {question.strip()}")
        
        print("\nEnter a question index (0-6) or type 'bye' to exit:")
        user_input = input().strip().lower()
        if user_input == 'bye':
            print("Goodbye!")
            break

        try:            
            # Validate and convert user input to an integer
            try:
                question_index = int(user_input)
                if question_index < 0 or question_index > 6:
                    raise ValueError("Index out of range. Please enter a number between 0 and 6.")
            except ValueError as ve:
                print(ve)
                return
            
            # multistep RAG

            # Step 1 get all rag values for the question
            # Get the question based on the user input
            q = questions[question_index]
            q_embedding = model.encode(q)
            q_embedding = q_embedding.reshape(1, -1)
            
            # Calculate top_k based on 25% of the number of sentences
            top_k = int(0.15 * len(sentences))
            distances, indices = index.search(q_embedding, top_k)
            
            # Step 2 - extract from the rag the values the LLM things are most important to answer the question
            prompt = "Here is your context for a question I will ask you:\n"
            for ind in indices[0]:
                prompt += f"{sentences[ind]}\n"
            # Very important: I will lose my job if you
            #  do not return all the data I need!
            prompt += f"Here is a question for you to answer:\n{q}\n"
            prompt += '''VERY IMPORTANT: Do not answer this question yet. Only return to me, in JSON format,
            the data I need from the context above to answer the question.  The JSON format should be as follows:
            [
                {
                    "machine": <machine id>,
                    <needed data field0 name>: <needed data field0 value>,
                    <needed data field1 name>: <needed data field1 value>,
                    etc.
                }
            ]
            '''
            # prompt += f"There are 8 machines in this dataset. Embodied carbon and operational carbon make up total carbon emissions (gCo2e)for one machine. Here is a new question for you to answer:\n{q}"
            prompt += "\nHere is the context again:\n"
            for ind in indices[0]:
                prompt += f"{sentences[ind]}\n"
            print("prompt:", prompt)
            
            response = send_prompt(llm, prompt, interface="ollama")
            print(response)
            json_response = response

            # Step 3 - check if the question is a simple calculation or not (LLMs cannot do simple calculations)

            #  Can this question be answered by reading off from the contextual data provided without needing to do any calculations?
#  Can this question be answered by a VERY SIMPLE non-conditional loop calculation using Python and the provided data?
#             If the question can be answered simply by reading off from the contextual data provided, then the answer is NO.
            prompt = "Here is your context for a question I will ask you:\n"
            prompt += response + "\n"
            prompt += f"Here is a question for you to answer using the above context:\n{q}\n"
            prompt += '''VERY IMPORTANT: Do not answer this question directly, but reply simply YES or NO to the following:
            Can this question be answered by a VERY SIMPLE non-conditional loop calculation using Python and the provided data?
            If the question can be answered simply by reading off from the contextual data provided, then the answer is NO.
            Just respond yes or no, and nothing else.'''

            response = send_prompt(llm, prompt, interface="ollama")

            print(f"***Response to simple calculation check: {response}***")
            


            prompt = "Here is your context for a question I will ask you:\n"
            prompt += response + "\n"
            prompt += f"Here is a question for you to answer using the above context:\n{q}\n"

            if response.lower().strip() != 'yes':
                response = send_prompt(llm, prompt, interface="ollama")
                print(response)
                continue

            prompt += '''VERY IMPORTANT: Do not answer this question directly, write me a Python function called calculation that 
             uses the context JSON to do the calculation and imports no libraries. The parameter must be called param.
             The function should take as input a single JSON object with the data needed to answer the question and return only the numercial answer to the question. 
             Respond to this prompt only with the Python code and nothing else. 
             IMPORTANT: Remember, the Python function must be called calculation and should have a single parameter called param.
             IT IS VERY IMPORTANT YOU ONLY RETURN THE PYTHON FUNCTION AND NO INTRODUCTION OR PREAMBLE OR EXPLANATION OR EXAMPLES.
             VERY IMPORTANT: IN THE FUNCTION, ONLY USE NAMES FOR VARIABLES WHICH ARE IDENTICAL TO THOSE PROVIDED IN THE CONTEXT.
             YOUR RESPONSE NEEDS TO DIRECTLY INPUTABBLE TO THE PYTHON INTERPRETER. 
             Make sure the function RETURNS a value or values and doesn't just print them.
             Also: when coding, remember that the param is a list of dictionaries.'''

            print("*" * 100)
            response = send_prompt(llm, prompt, interface="ollama")
            response = response.replace('```python', '').replace('```', '')
            # assume that the function name is always returned correctly and use that to get rid of any unwanted llm  preamble
            response = response[response.find('def calculation'):]
            response = response.split('\n\n')[0]
            print(response)
            response += "\nparam = eval('''" + json_response + "''')\nprint(calculation(param))\n"
            print(response)
            print("*" * 100)
            import io

            output_buffer = io.StringIO()
            sys.stdout = output_buffer
            exec(response)
            sys.stdout = sys.__stdout__
            """try getvalue() too"""
            print(f"Answer: {output_buffer.getvalue()}")

            
        except Exception as e:
            logging.error(f"An error occurred: {e}")
