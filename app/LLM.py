import pandas as pd
import yaml
import json
import csv
import sys, os 
import faiss
import numpy as np
import logging
import pickle
import ollama
import io
import re
from detect_os import detect_os
user_os = detect_os()


# model_path = r"models\Meta-Llama-3-8B-Instruct.Q5_0.gguf"
# llm = Llama(
#     model_path=model_path,
# #         temperature=0.1,
#         n_ctx=16000,
#         n_gpu_layers=-1,
#         verbose=True,
# )

model_name = "llama3"
def send_prompt(prompt: str, interface: str = "ollama",
                max_tokens: int = 1024, temperature: float = 0):
    if interface == "ollama":
        if user_os == 'Windows':
            # Checking that the user has pulled the model and moved the folders to this directory before running the code
            if model_name == "llama3qa" and not is_model_pulled("llama3"):
                print("You need to pull 'llama3-chatqa' from Ollama first and move the blobs and manifests folders to the models folder in the app/models folder of this application.")
                sys.exit(1)
            elif model_name == "llama3" and not is_model_pulled("llama3"):
                print("You need to pull 'llama3' from Ollama first and move the blobs and manifests folders to the models folder in the app/models folder of this application.")
                sys.exit(1)
        
        # Generating the response with streaming enabled
        if model_name == "llama3qa":
            response_stream = ollama.generate(
                model="llama3-chatqa",
                prompt=prompt,
                keep_alive='24h',
                options={'num_ctx': 16000, 'temperature': temperature},
                stream=True  
            )
        elif model_name == "llama3":
            response_stream = ollama.generate(
                model="llama3",
                prompt=prompt,
                keep_alive='24h',
                options={'num_ctx': 16000, 'temperature': temperature},
                stream=True 
            )
        else:
            raise ValueError("Unsupported model name provided.")

        # Streaming the response
        for chunk in response_stream:
            yield chunk['response']  
    else:
        raise ValueError("Unsupported interface provided. Exiting the system.")


def is_model_pulled(model_name: str) -> bool:
    # checking the model is there 
    model_path = os.path.join("app", "models", "manifests", "registry.ollama.ai", "library", model_name)

    return os.path.exists(model_path)
  
 

def prepare_excel_file(excel_file):
    """function to take in excel file and preppare it for llm, adding empty carbon emissions column, filling model column, renaming columns and dropping unnecessary columns"""
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
    df.insert(2, 'carbon emissions (gCO2eq) - use this for questions about CARBON EMISSIONS', pd.NA)  # You can initialize with pd.NA or any default value
    # Update headers to reflect the new column
    headers.insert(2, 'carbon emissions (gCO2eq) - use this for questions about CARBON EMISSIONS')  # Insert 'carbon' into the correct position
    # Set the new headers
    df.columns = headers
    # Drop the first two rows which were used for headers
    df = df.drop([0, 1]).reset_index(drop=True)

    replace_dict = {
        '#Cores': 'number of cores',
        'CPU\nHighest\navg': 'central processing unit average utilisation percent',
        'GPU\navg': 'graphics processing unit average utilisation percent',
        'Total MB\nSent': 'MB sent across network traffic',
        'Total MB\nReceived': 'MB received across network traffic',
        'GPU\n#oc > 80%': 'number of occurrences graphics processing unit went over 80%',
        'Core\nHighest\nmax': 'core maximum utilisation percent (single core of highest usage)',
        'Core\nHighest\navg': 'core average utilisation percent (single core of highest usage)',
        'Core\n# oc > 80%': 'core number of occurrences over 80%',
        'Core \nTotal Seconds > 80%': 'core total seconds over 80%',
        '\nCPU\nTotal Seconds > 80%': 'central processing unit total seconds over 80%',
        'CPU\nHighest\nmax': 'central processing unit maximum utilisation percent',
        '\nCPU# oc > 80%': 'number of occurrences central processing unit went over 80%',
        'Total RAM\n(GB)': 'total RAM capacity in GB',
        'max': 'maximum memory utilisation percent',
        'avg': 'average memory utilisation percent',
        '#oc > 80%': 'number of occurrences GPU memory went over 80%',
        # 'GPU\nmin': 'graphics processing unit minimum (NVIDIA % utilization)', 
        'GPU\nmax': 'graphics processing unit maximum utilisation percent',
        'Host Name': 'Machine',
        'MEM\nmax': 'graphics processing unit maximum memory utilisation percent', 
        'MEM\navg': 'graphics processing unit average memory utilisation percent', 
        'MEM\n#oc > 80%': 'number of occurrences GPU memory went over 80%',
        'Model': 'model',
        'Machine': 'machine'
    }

    df.rename(columns=replace_dict, inplace=True)
    required_columns = list(replace_dict.keys())
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns == []:
        raise ValueError(f"The following required columns are missing after processing the file: {', '.join(missing_columns)}")

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
        'Free MB\navg', 'min', 'MEM\nmin', 'GPU\nmin', 
    ]

    # Drop the columns that are not needed 
    df.drop(columns=drop_names, inplace=True)
    # now drop the last three rows in the df
    df.drop(df.tail(3).index, inplace=True)

    # for testing with all columns, for fairness must also round those cols
    # df[drop_names] = df[drop_names].apply(lambda x: round(x, 3))


    """round the values in the column GPU average (NVIDIA % Utilization) to 3 decimal places"""
    df['graphics processing unit average utilisation percent'] = df['graphics processing unit average utilisation percent'].apply(lambda x: round(x, 2))
    df['central processing unit average utilisation percent'] = df['central processing unit average utilisation percent'].apply(lambda x: round(x, 2))
    df['core average utilisation percent (single core of highest usage)'] = df['core average utilisation percent (single core of highest usage)'].apply(lambda x: round(x, 2))
    df['core maximum utilisation percent (single core of highest usage)'] = df['core maximum utilisation percent (single core of highest usage)'].apply(lambda x: round(x, 2))
    df['central processing unit maximum utilisation percent'] = df['central processing unit maximum utilisation percent'].apply(lambda x: round(x, 2))
    df['MB sent across network traffic'] = df['MB sent across network traffic'].apply(lambda x: round(x, 2))
    df['MB received across network traffic'] = df['MB received across network traffic'].apply(lambda x: round(x, 2))
    df['total RAM capacity in GB'] = df['total RAM capacity in GB'].apply(lambda x: round(x, 2))
    df['average memory utilisation percent'] = df['average memory utilisation percent'].apply(lambda x: round(x, 2))
    df['maximum memory utilisation percent'] = df['maximum memory utilisation percent'].apply(lambda x: round(x, 2))
    # df['graphics processing unit minimum (NVIDIA % utilization)'] = df['graphics processing unit minimum (NVIDIA % utilization)'].apply(lambda x: round(x, 3))
    df['graphics processing unit maximum utilisation percent'] = df['graphics processing unit maximum utilisation percent'].apply(lambda x: round(x, 2))
    df['graphics processing unit maximum memory utilisation percent'] = df['graphics processing unit maximum memory utilisation percent'].apply(lambda x: round(x, 2))
    df['graphics processing unit average memory utilisation percent'] = df['graphics processing unit average memory utilisation percent'].apply(lambda x: round(x, 2))
    return df
 


def load_data_files(yaml_file, return_yaml: bool = False) -> tuple:
    # Load yaml file
    with open(yaml_file, 'r') as f:
        # this data is what is put in <BACKGROUND> tag in the prompt - before RAG was implemented
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
    lowest_children_level = yaml_data['tree']['children']['child']['children']
    lowest_children_level.update(yaml_data['tree']['children']['child']['children'])
    # dump the yaml to file for debug
    with open(r'embeddings/yaml_dump.txt', 'w') as f:
        yaml.dump(lowest_children_level, f)
    for i, machine in enumerate(lowest_children_level):
        for child in lowest_children_level[machine]['outputs']:
            # convert child to a dictionary
            child = dict(child)
            # only pull out values for keys timestamp, instance-type, sci
            child = {k: v for k, v in child.items() if k in ['timestamp', 'instance-type', 'sci', 'carbon-embodied', 'carbon-operational', 'duration', 'carbon']}
            # convert timestamp to a UTC datetime not an object
            child['timestamp'] = pd.to_datetime(child['timestamp'], utc=True)
            # convert that to a string
            child['timestamp'] = child['timestamp'].strftime('%Y-%m-%d')
            # round sci to 6 dp
            child['sci'] = round(child['sci'], 2)
            child['carbon'] = round(child['carbon'], 2)
            # pull out value for carbon embodied, carbon operational, and duration
            child['carbon-embodied'] = round(child['carbon-embodied'], 2)
            child['carbon-operational'] = round(child['carbon-operational'], 2)
            child['duration'] = child['duration']
            # letters 7 to 10 are unique to each machine
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
    for idx, item in enumerate(machine_emissions_list):
        if idx < len(machine_ids):
            item['machine'] = machine_ids[idx]
    for machine in machine_emissions_list:
        for i in range(len(machine_ids)):
            if prepared_df.loc[i, 'Machine'] == machine['machine']:
                prepared_df.loc[i, 'carbon emissions (gCO2eq) - use this for questions about CARBON EMISSIONS'] = machine['carbon']
    # if there is no column called duration, add it to the dataframe  and fill it with the duration value
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
    #do the same for timestamp
    for machine in machine_emissions_list:
        for i in range(len(machine_ids)):
            if prepared_df.loc[i, 'Machine'] == machine['machine']:
                prepared_df.loc[i, 'timestamp'] = machine['timestamp']
    return prepared_df, machine_ids

def label_max_min(col):
    if pd.api.types.is_numeric_dtype(col):
        max_val = col.max()
        min_val = col.min()
        return col.apply(lambda x: f"{x} (Highest)" if x == max_val else (f"{x} (Lowest)" if x == min_val else str(x)))
    else:
        return col  # Return the column as-is if it's not numeric

def append_sum_row(df, column_name, label='total carbon emissions in gCO2eq'):
    """
    IMPORTANT: this function is not used in the current implementation due to the labels needed in the columns for max and min qs rendering this function now invalid 
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


def extract_and_sum_numeric(df, column_name):
    """Extract numeric values from each cell in the specified column and calculate their sum."""
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
    
    def extract_numeric(value):
        # Use regular expressions to extract numeric parts from a string
        numbers = re.findall(r'\d+\.?\d*', str(value))
        return float(numbers[0]) if numbers else 0

    # Apply the function to each cell in the column and sum the results
    total_sum = df[column_name].apply(extract_numeric).sum()

    return total_sum


def csv_to_json(csv_filename, as_json=True):
    """Convert the csv filenmae into a json file"""
    data_dict = {}
    with open(csv_filename, mode='r') as file:
        csvFile = csv.reader(file)
        headers = next(csvFile)
    
        # Find the index of the 'carbon emissions (gCO2eq)' column
        carbon_emissions_index = headers.index('carbon emissions (gCO2eq) - use this for questions about CARBON EMISSIONS')
        
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
        # if embeddings.pickle exists, load the embeddings from file  and skip the encoding step - removed as system needs to rebuild embeddings for each start updepending on file upload
        # if os.path.exists(r'embeddings\embeddings.pickle'):   
        #     # Load the embeddings from file
        #     with open(r'embeddings\embeddings.pickle', 'rb') as file:
        #         embeddings = pickle.load(file)
        #         rebuild_faiss_index = False
        # else:
            # Encode sentences to get their embeddings
        embeddings = model.encode(sentences)
        rebuild_faiss_index = True
        
        """save the encodings to pickle file"""
        with open(r'embeddings/embeddings.pickle', 'wb') as file:
            pickle.dump(embeddings, file)
    # Convert embeddings to a numpy array
        embeddings = np.array(embeddings)

        # Create a FAISS index
        d = embeddings.shape[1]  # Dimension of embeddings (768)
        index = faiss.IndexFlatL2(d)
        
        """if faiss_index.bin exists, load the index from file and skip the add step"""
        # if os.path.exists('faiss_index.bin') and not rebuild_faiss_index:
        #     index = faiss.read_index('faiss_index.bin')
        #     print("\n\nFAISS index loaded from 'faiss_index.bin'")
        # else:
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

def extract_json_from_response(response):
    # Use regex to find the first JSON array in the response
    json_match = re.search(r'\[.*?\]', response, re.DOTALL)
    if json_match:
        return json_match.group(0)
    return None

def generate_question(index, embeddings, model, sentences, questions, machine_ids, model_name):
    num_of_machines = str(len(machine_ids))
    # This is where the different personas can be catered to, however removed due to being more complex than anticipateed in terms of its effect on the accuracy of the model
    # =================================================================================================================================================================
    # print(f"Number of machines: {num_of_machines}")     
    # while True:
    #     profile_input = input("""\n\n\n What is your job role? Enter 1, 2, or 3:\n
    #                         (1) Director of IT \n\n
    #                         (2) IT Admin\n\n
    #                         (3) N/A 
    #                         Type 'exit' to quit.\n""")
        
    #     if profile_input == '1' or profile_input == '2' or profile_input == '3':
    #         print("Adjusting my outputs...")
    #         if profile_input == '1':
    #             prompt_appendix = """The user you are speaking with is the Director of IT. 
    #                             Their goal is to optimise resource allocation and utilisation, and to ensure cost-effectiveness. 
    #                             If asking for summary of compute, you must give a high overview and whether or not any of the machines should move up or down compute power.
    #                             """
    #         elif profile_input == '2':
    #             prompt_appendix = """The user you are speaking with is the IT Admin.
    #                                 Their goal is to manage and maintain the machines and their storage. 
    #                                 If asking for a summary, you must give a detailed breakdown of each machine. 
    #                                 """
    #         elif profile_input == '3':
    #             prompt_appendix = ''
    #         break
    #     elif profile_input.lower() == 'exit':
    #         print("Sorry to see you go so soon.")
    #         break
    #     else:
    #         print("Invalid input, please try again.")
    # =================================================================================================================================================================
    while True:
        # Display the list of questions with indices
        for i, question in enumerate(questions):
            print(f"{i}: {question.strip()}")
        
        print("\nEnter a question index (0-7), type your own question, or type 'bye' to exit:")
        user_input = input().strip()

        if user_input.lower() == 'bye':
            print("Goodbye!")
            break
    
        question_index = None
        # Check if the input is a digit and within the valid range
        if user_input.isdigit():
            question_index = int(user_input)
            if 0 <= question_index < len(questions):
                # User selected a question from the list
                q = questions[question_index]
                print(f"You selected question {question_index}: {q}")
            else:
                print("Index out of range. Please enter a number between 0 and 6.")
                continue
        else:
            # Treat the input as a custom question
            q = user_input
            print(f"You entered a custom question: {q}")
        # Step 1 get all rag values for the question
        # Get the question based on the user input
        q_embedding = model.encode(q)
        q_embedding = q_embedding.reshape(1, -1)
        
        # Calculate top_k based on 25% of the number of sentences
        top_k = int(0.25 * len(sentences))
        distances, indices = index.search(q_embedding, top_k)
       
        # Step 2 - extract from the rag the values the LLM thinks are most important to answer the question
        prompt = "Here is your context for a question I will ask you:\n"
        for ind in indices[0]:
            if 0 <= ind < len(sentences):
                prompt += f"{sentences[ind]}\n"
            else:
                print(f"Warning: Index {ind} is out of range.")
        # print('prompt:', prompt)
        prompt += f"Use the above context to answer this question:\n{q}\n"
        # print("prompt:", prompt)
        if model_name == 'llama3':
            print("RUNNING Llama3")
            if question_index is not None and question_index == 1:
                prompt += '''VERY IMPORTANT:  Return to me, in JSON format,
                the data I need from the context above to answer the question.  The JSON format should be as follows:
                [
                    {
                        "machine": <machine id>,
                        <data-field0>: <data-field0 value>,
                        <data-field1>: <data-field1 value>,
                        etc.
                    }
                ]
                The data field keys should ONLY be an exactly copied label (not an abbreviation or reduction) from the context I provided and the values should be the actual values from the context I provided.
                VERY IMPORTANT: There are ''' + num_of_machines + ' machines in this total. Check the context properly. Do not leave any out. ' + num_of_machines + ' MACHINES therefore ' + num_of_machines + ' dictionaries in the list.'
                prompt += "\nHere is that context again:\n"
                for ind in indices[0]:
                    prompt += f"{sentences[ind]}\n"
                # print("prompt:", prompt)
                # response = send_prompt(prompt, interface="ollama")
                # print(response)
                response = ""
                for chunk in send_prompt(prompt=prompt, interface="ollama", temperature=0):
                    response += chunk
                    print(chunk, end='', flush=True)
                json_response = response
                # remove any pre-amble or post comment from llm by getting location of first [ and last ]
                json_response = json_response[json_response.find('['):json_response.rfind(']')+1]
# =================================================================================================================================================================
# Would be where llm judges the type of question it has been asked (see archive: archive\arhive-llm-getting-llm-to-judge-question-type.py)
# =================================================================================================================================================================
                prompt = "Here is your context for a question I will ask you:\n"
                prompt += json_response + "\n"
                prompt += f"Here is a question for you to answer using the above context:\n{q}\n"
                prompt += '''VERY IMPORTANT: Do not answer this question directly, write me a Python function called calculation that 
                uses the context JSON to do the calculation and imports no libraries. The parameter must be called param.
                The function should take as 
                input a single JSON object with the data needed to answer the question and return only the numercial answer to the 
                    question. 
                Respond to this prompt only with the Python code and nothing else. 
                IMPORTANT: Remember, the Python function must be called calculation and should have a single parameter called param.
                IT IS VERY IMPORTANT YOU ONLY RETURN THE PYTHON FUNCTION AND NO INTRODUCTION OR PREAMBLE OR EXPLANATION OR EXAMPLES.
                YOUR RESPONSE NEEDS TO DIRECTLY INPUTABLE TO THE PYTHON INTERPRETER. 
                Make sure the function RETURNS a value or values and doesn't just print them.
                Also: when coding, remember that the param is a list of dictionaries.
                VERY IMPORTANT: Only use the precise data field labels from the context I provided in the Python code you return.
                Here's the context again:'''
                prompt += json_response + "\n"
                # print("*" * 100)
                response = ""
                for chunk in send_prompt(prompt=prompt, interface="ollama", temperature=0):
                    response += chunk
                    # print(chunk, end='', flush=True)
                # response = send_prompt(prompt, interface="ollama")
                response = response.replace('```python', '').replace('```', '')
                # assume that the function name is always returned correctly and use that to get rid of any unwanted llm  preamble
                response = response[response.find('def calculation'):]
                response = response.split('\n\n')[0]
                print("\n\n\n")
                print("*" * 100)
                response += "\nparam = eval('''" + json_response + "''')\nprint(calculation(param))\n"
                # print(response)

                # print("*" * 100)
                output_buffer = io.StringIO()
                sys.stdout = output_buffer
                exec(response)
                sys.stdout = sys.__stdout__
                """try getvalue() too"""
                print(f"Answer in gcO2eq: {output_buffer.getvalue()}")
                prompt = f'Here is your answer to the question {q}: {output_buffer.getvalue()}\n'
                # prompt += output_buffer.getvalue()
                prompt += f"If there is any additional relevant data in the following context which you think is important to add to answer the question {q}, enhance your answer with it.  IMPORTANT: If there is none, your next response should be empty.: "
                for ind in indices[0]:
                    if 0 <= ind < len(sentences):
                        prompt += f"{sentences[ind]}\n"
                    else:
                        print(f"Warning: Index {ind} is out of range.")
                prompt += f"Your response must be in plain English, including the value {output_buffer.getvalue()} in gCO2eq. Do not include any code in your response."
                # response = send_prompt(prompt, interface="ollama", temperature=0)
                for chunk in send_prompt(prompt=prompt, interface="ollama", temperature=0):
                    print(chunk, end='', flush=True)
                # print(f'\n\n\n', response)
                continue
            else: 
                prompt += f"VERY IMPORTANT: you must take into account all {num_of_machines} machines and their respective data in the context OTHERWISE I WILL LOSE MY JOB"
                prompt += 'DO NOT MIX UP THE VALUES ACROSS THE MACHINES! \n\n'
                # response = send_prompt(prompt, interface="ollama", temperature=0)
                for chunk in send_prompt(prompt=prompt, interface="ollama", temperature=0):
                    print(chunk, end='', flush=True)
                # print(f'\n\n\n', response)
                print("\n\n\n")
                continue
        
        elif model_name == 'llama3qa': 
            print("RUNNING LLAMA3QA")
            prompt += 'VERY IMPORTANT: There are ' + num_of_machines + ' machines in total so your observations must take into account ' + num_of_machines + ' machines - Check the context properly. Do not leave any out or I will LOSE MY JOB if not all ' + num_of_machines + ' are included.'
            prompt += "You are exceptional at mathematics and must perform addition perfectly. To do so, only add two numbers at a time and find a total value by applying this method."
            # prompt += f"Here is a question for you to answer using the above context:\n{q}\n"
            # print("prompt:", prompt)
            # response = send_prompt(prompt, interface="ollama", temperature=0.5)
            # print(response)
            for chunk in send_prompt(prompt=prompt, interface="ollama", temperature=0):
                print(chunk, end='', flush=True)
            print("\n\n\n")
            #prompt += prompt_appendix
            continue
        else:
            print("No model available. Check the server.")
            continue


def add_context_to_sentences(sentences, duration, start_date, end_date, analysis_window, num_of_machines, merged_df):
    # Prepare the duration and date of data collection sentences
    date_of_data_collection = 'DATE OF DATA COLLECTION: Data was collected between ' + start_date[:10] + ' and ' + end_date[:10] + '.'
    CPU_average_context = 'If the CPU average utilisation percent is above 85%, it indicates that the machine is under heavy load and may need more resources, like being moved to a higher compute power with more cores.  Around 75% utlilisation is very efficient as it is making the most of the resources (good for sustainability), yet work is not slowed down by compute limitations.  If the value is below 15%, it indicates that the machine is underutilised and it it would be more cost effective to reduce resources, like a lower compute power. This saves money and is more environmentally friendly which is important to the business.'
    CPU_maximum_high_context = 'If the CPU maximum utilisation percent is above 95%, it indicates that the machine is close to performing at maximum compute power which means it may be causing a bottleneck. However, if the average is low, then by itself it is not a reason to change the compute power.'
    # CPU_average_low_context = 'If the central processing unit average utilisation percent is below 15%, it indicates that the machine is underutilised and it it would be more cost effective to reduce resources, like a lower compute power. This saves money and is more environmentally friendly which is important to the business.'
    GPU_average_context = 'If the graphics processing unit average utilisation percent is above 85%, it indicates that the machine is under heavy load and may need more resources, like a higher compute power with more cores. Around 75% utlilisation is very efficient as it is making the most of the resources (good for sustainability), yet work is not slowed down by compute limitations. If the value is below 15%, it indicates that the machine is underutilised and it it would be more cost effective to reduce resources, like a lower compute power. This saves money and is more environmentally friendly which is important to the business.'
    GPU_maximum_high_context = 'If the graphics processing unit maximum utilisation percent is above 95%, it indicates that the machine is close to performing at maximum compute power which means it may be causing a bottleneck. However, if the average is low, then by itself it is not a reason to change the compute power.'
    # GPU_average_low_context = 'If the graphics processing unit average utilisation percent is below 15%, it indicates that the machine is underutilised and it it would be more cost effective to reduce resources, like a lower compute power. This saves money and is more environmentally friendly which is important to the business.'
    sustainability_beliefs = 'Sustainable Impact is a business imperative to HP. Reducing waste and carbon footprint is very important. Making sure machines run efficiently is important to reducing carbon emissions.'
    carbon_explanation = "The embodied carbon for each machine is produced during the manufacturing and have already been proportionately distributed over the observed duration, using the Green Software Foundation's Impact Framework to calculate this. Carbon Emissions already includes embodied and operational. We must NEVER add embodied AND operational AND carbon as this will give us double the actual value."
    # Initialize collection_period with a default value
    collection_period = "Data collection time is unknown."
    wording = "If a user asks about 'the pool', they are referring to all the " + num_of_machines + " machines in the dataset. A machine can also be referred to as a 'unit'"
    # Split the analysis_window into words and clean up
    words = analysis_window.split()
    cleaned_words = []
    total_sum = extract_and_sum_numeric(merged_df, 'carbon emissions (gCO2eq) - use this for questions about CARBON EMISSIONS').round(2)
    total_sum_operational = extract_and_sum_numeric(merged_df, 'operational carbon (gCO2eq)').round(2)
    total_sum_embodied = extract_and_sum_numeric(merged_df, 'embodied carbon (gCO2eq)').round(2)

    total_sum_of_carbon_emissions = "The total carbon emissions across all " + num_of_machines + " machines is " + str(total_sum) + " gCO2eq. Answer with this value if asked to give the total sum for all machines of carbon emissions."
    total_sum_of_operational_emissions = "The total operational carbon emissions across all " + num_of_machines + " machines is " + str(total_sum_operational) + " gCO2eq. Answer with this value if asked to give the total OPERATIONAL carbon emissions."
    total_sum_of_embodied_emissions = "The total embodied emissions across all " + num_of_machines + " machines is " + str(total_sum_embodied) + " gCO2eq. Answer with this value if asked to give the total EMBODIED carbon emissions."    
    # Replace day abbreviations with full names and remove any unwanted characters
    for word in words:
        cleaned_word = word.strip(',."')
        if cleaned_word == "Mon":
            cleaned_words.append("Monday")
        elif cleaned_word == "Tue":
            cleaned_words.append("Tuesday")
        elif cleaned_word == "Wed":
            cleaned_words.append("Wednesday")
        elif cleaned_word == "Th":
            cleaned_words.append("Thursday")
        elif cleaned_word == "Fri":
            cleaned_words.append("Friday")
        elif cleaned_word == "Sat":
            cleaned_words.append("Saturday")
        elif cleaned_word == "Sun":
            cleaned_words.append("Sunday")
        else:
            cleaned_words.append(word)

    # Iterate through the cleaned words to find the first occurrence of a time
    for i in range(len(cleaned_words)):
        word = cleaned_words[i]
        if word == '8:00':
            collection_period = "This telemetry data was collected during working days (daytime), meaning numbers reflect a period of time where the machines are most likley to be used, so are likley to have higher values than the night time figures."
            break
        elif word == '20:00':
            collection_period = "This telemetry was collected at night, meaning numbers more likely reflect downtime. However, this is not guaranteed, as some teams train models at night."
            break
    duration_of_data_collection = 'DURATION: This data was collected over a period of ' + str(duration) + ' seconds, equivalent to ' + str((duration / 60) / 60) + ' hours.' + collection_period 
    duration_of_operational_carbon = 'Operational carbon was produced over a total of ' + str((duration / 60) / 60) + ' hours. Each day was 12 hours.'
    # Rebuild the updated analysis_window sentence
    analysis_window = ' '.join(cleaned_words)
    days_of_collection = 'This data was collected on the following days: ' + analysis_window + '.'
 
    # Append the sentences to the list
    sentences += [duration_of_operational_carbon, duration_of_data_collection, date_of_data_collection, days_of_collection, CPU_average_context, GPU_average_context, CPU_maximum_high_context, GPU_maximum_high_context, sustainability_beliefs, carbon_explanation, total_sum_of_carbon_emissions, total_sum_of_embodied_emissions, total_sum_of_operational_emissions]
    

def extract_metric_and_machines(question, machine_ids):
    # Example: "What is the total CPU average for machines ld71r18u44dws, ld71r16u15ws?" where CPU average is the metric key mapped to the headers 
    metric_keywords = {
    "operational carbon": "operational carbon (gco2eq)",
    "embodied carbon": "embodied carbon (gco2eq)",
    'embodied emissions': "embodied carbon (gco2eq)",
    "operational emissions": "operational carbon (gco2eq)",
    "carbon emissions": "carbon emissions (gco2eq) - use this for questions about carbon emissions",
    "number of cores": "number of cores",
    "maximum core utilisation": "core maximum utilisation percent (single core of highest usage)",
    "average core utilisation": "core average utilisation percent (single core of highest usage)",
    "core occurrences over 80%": "core number of occurrences over 80percent",
    "core total seconds over 80%": "core total seconds over 80percent",
    "cpu max utilisation": "central processing unit maximum utilisation percent",
    "cpu avg utilisation": "central processing unit average utilisation percent",
    "cpu occurrences over 80%": "number of occurrences central processing unit went over 80percent",
    "cpu total seconds over 80%": "central processing unit total seconds over 80percent",
    "total ram": "total ram capacity in gb",
    "max memory utilisation": "maximum memory utilisation percent",
    "average memory utilisation": "average memory utilisation percent",
    "gpu memory occurrences over 80%": "number of occurrences graphics processing unit memory went over 80percent",
    "network traffic sent": "megabytes sent across network traffic",
    "network traffic received": "megabytes received across network traffic",
    "gpu max utilisation": "graphics processing unit maximum utilisation percent",
    "gpu avg utilisation": "graphics processing unit average utilisation percent",
    "gpu occurrences over 80%": "number of occurrences graphics processing unit went over 80percent",
    "gpu max memory utilisation": "graphics processing unit maximum memory utilisation percent",
    "gpu average memory utilisation": "graphics processing unit average memory utilisation percent"
}
    
    # Identify the metric
    metric = None
    user_friendy_metric = None
    for key in metric_keywords:
        if key.lower() in question.lower():
            metric = metric_keywords[key]
            user_friendy_metric = key.lower()
            break
    
    # Extract machine identifiers
    machines = [machine_id for machine_id in machine_ids if machine_id in question]
    
    return metric, machines, user_friendy_metric


def perform_calculation(metric, machines, operation="sum", sentences=None):
    if sentences is None:
        raise ValueError("Sentences list cannot be None")
    
    values = []
    
    for sentence in sentences:
        machine_id = sentence.split(" ")[0]
        if machine_id in machines:
            if metric in sentence:
                value_str = sentence.split('=')[-1].strip()
                value_str = re.sub(r'[^\d.]+', '', value_str)
                try:
                    value = float(value_str)
                    values.append(value)
                except ValueError:
                    print(f"Could not convert '{value_str}' to float.")
    # Perform the specified operation
    if operation == "sum":
        return sum(values)
    else:
        raise ValueError("Unsupported operation")


def response_decorator(func):
    def wrapper(question, machine_ids, sentences):
        # Determine if the question is asking for a calculation
        if any(keyword in question.lower() for keyword in ["total", "sum", "add"]):
            # Extract metric and machines
            metric, machines, user_friendy_metric = extract_metric_and_machines(question, machine_ids)
            if metric and machines:
                operation = "sum" if "total" in question.lower() or "sum" in question.lower() or "add" in question.lower() else "average"
                result = perform_calculation(metric, machines, operation, sentences)
                print('resultllm', result)
                if result is not None:
                    return f"The total {user_friendy_metric} for machines {', '.join(machines)} is {result:.2f}.\n\n"
        return func(question, machine_ids, sentences)
    return wrapper


@response_decorator
def model_response(question, machine_ids, sentences):
    return "blah hello blah."


def parse_user_input(question, term_mapping):
    question_lower = question.lower()
    for term, standardised in term_mapping.items():
        if term in question_lower:
            return standardised
    return None


def process_user_input(machine_ids, model, index, sentences, send_prompt, questions):
    TERM_MAPPING = {
    'embodied carbon': 'embodied carbon (gCO2eq)',
    'embodied emissions': 'embodied carbon (gCO2eq)',
    'embodied': 'embodied carbon (gCO2eq)',
    'operational carbon': 'operational carbon (gco2eq)',
    'operational emissions': 'operational carbon (gco2eq)',
    'operational': 'operational carbon (gco2eq)',
    'carbon emissions': 'carbon emissions (gCO2eq)'
}
    num_of_machines = str(len(machine_ids))
    while True:
        # Display the list of questions with indices
        for i, question in enumerate(questions):
            print(f"{i}: {question.strip()}")
        
        print("\nEnter a question index (0-7), type your own question, or type 'bye' to exit:")
        user_input = input().strip()

        if user_input.lower() == 'bye':
            print("Goodbye!")
            break
    
        question_index = None
        # Check if the input is a digit and within the valid range
        if user_input.isdigit():
            question_index = int(user_input)
            if 0 <= question_index < len(questions):
                # User selected a question from the list
                q = questions[question_index]
                print(f"You selected question {question_index}: {q}")
            else:
                print("Index out of range. Please enter a number between 0 and 7.")
                continue
        else:
            # Processing the custom question
            q = user_input
            print(f"You entered a custom question: {q}")
        standardised_metric = parse_user_input(q, TERM_MAPPING)
        # Step 1: Get all RAG values for the question
        q_embedding = model.encode(q)
        q_embedding = q_embedding.reshape(1, -1)

        # Calculate top_k based on 25% of the number of sentences
        top_k = int(0.25 * len(sentences))
        distances, indices = index.search(q_embedding, top_k)

        # Step 2: Extract from the RAG the values the LLM thinks are most important to answer the question
        prompt = "Here is your context for a question I will ask you:\n"
        rag_sentences = []
        for ind in indices[0]:
            if 0 <= ind < len(sentences):
                rag_sentences  += [sentences[ind]]
                prompt += f"{sentences[ind]}\n"
            else:
                print(f"Warning: Index {ind} is out of range.")
        if question_index is not None and question_index == 1:
                prompt += f"Use the above context to answer this question:\n{q}\n"
                prompt += '''VERY IMPORTANT:  Return to me, in JSON format,
                the data I need from the context above to answer the question.  The JSON format should be as follows:
                [
                    {
                        "machine": <machine id>,
                        <data-field0>: <data-field0 value>,
                        <data-field1>: <data-field1 value>,
                        etc.
                    }
                ]
                The data field keys should ONLY be an exactly copied label (not an abbreviation or reduction) from the context I provided and the values should be the actual values from the context I provided.
                VERY IMPORTANT: There are ''' + num_of_machines + ' machines in this total. Check the context properly. Do not leave any out. ' + num_of_machines + ' MACHINES therefore ' + num_of_machines + ' dictionaries in the list.'
                prompt += "\nHere is that context again:\n"
                for ind in indices[0]:
                    prompt += f"{sentences[ind]}\n"
                # print("prompt:", prompt)
                # response = send_prompt(prompt, interface="ollama")
                # print(response)
                response = ""
                for chunk in send_prompt(prompt=prompt, interface="ollama", temperature=0):
                    response += chunk
                    print(chunk, end='', flush=True)
                json_response = response
                # remove any pre-amble or post comment from llm by getting location of first [ and last ]
                json_response = json_response[json_response.find('['):json_response.rfind(']')+1]
# =================================================================================================================================================================
# Would be where llm judges the type of question it has been asked (see archive: archive\arhive-llm-getting-llm-to-judge-question-type.py)
# =================================================================================================================================================================
                prompt = "Here is your context for a question I will ask you:\n"
                prompt += json_response + "\n"
                prompt += f"Here is a question for you to answer using the above context:\n{q}\n"
                prompt += '''VERY IMPORTANT: Do not answer this question directly, write me a Python function called calculation that 
                uses the context JSON to do the calculation and imports no libraries. The parameter must be called param.
                The function should take as 
                input a single JSON object with the data needed to answer the question and return only the numercial answer to the 
                    question. 
                Respond to this prompt only with the Python code and nothing else. 
                IMPORTANT: Remember, the Python function must be called calculation and should have a single parameter called param.
                IT IS VERY IMPORTANT YOU ONLY RETURN THE PYTHON FUNCTION AND NO INTRODUCTION OR PREAMBLE OR EXPLANATION OR EXAMPLES.
                YOUR RESPONSE NEEDS TO DIRECTLY INPUTABLE TO THE PYTHON INTERPRETER. 
                Make sure the function RETURNS a value or values and doesn't just print them.
                Also: when coding, remember that the param is a list of dictionaries.
                VERY IMPORTANT: Only use the precise data field labels from the context I provided in the Python code you return.
                Here's the context again:'''
                prompt += json_response + "\n"
                # print("*" * 100)
                response = ""
                for chunk in send_prompt(prompt=prompt, interface="ollama", temperature=0):
                    response += chunk
                    print(chunk, end='', flush=True)
                # response = send_prompt(prompt, interface="ollama")
                response = response.replace('```python', '').replace('```', '')
                # assume that the function name is always returned correctly and use that to get rid of any unwanted llm  preamble
                response = response[response.find('def calculation'):]
                response = response.split('\n\n')[0]
                print("\n\n\n")
                print("*" * 100)
                response += "\nparam = eval('''" + json_response + "''')\nprint(calculation(param))\n"
                print(response)

                # print("*" * 100)
                output_buffer = io.StringIO()
                sys.stdout = output_buffer
                exec(response)
                sys.stdout = sys.__stdout__
                """try getvalue() too"""
                print(f"Answer in gcO2eq: {output_buffer.getvalue()}")
                prompt = f'Here is your answer to the question {q}: {output_buffer.getvalue()}\n'
                # prompt += output_buffer.getvalue()
                prompt += f"If there is any additional relevant data in the following context which you think is important to add to answer the question {q}, enhance your answer with it.  IMPORTANT: If there is none, your next response should be empty.: "
                for ind in indices[0]:
                    if 0 <= ind < len(sentences):
                        prompt += f"{sentences[ind]}\n"
                    else:
                        print(f"Warning: Index {ind} is out of range.")
                prompt += f"Your response must be in plain English, including the value {output_buffer.getvalue()} in gCO2eq. Do not include any code in your response."
                # response = send_prompt(prompt, interface="ollama", temperature=0)
                for chunk in send_prompt(prompt=prompt, interface="ollama", temperature=0):
                    print(chunk, end='', flush=True)
                # print(f'\n\n\n', response)
                print("\n\n\n")
                continue
        prompt += f"Use the above context to answer this question:\n{q}\n"
        prompt += 'DO NOT MIX UP THE VALUES ACROSS THE MACHINES! \n\n\n' 
        if standardised_metric:
            prompt += f"Use the above context to answer this question based on the metric: {standardised_metric}.\n"
        else:
            prompt += f"Use the above context to answer this question:\n{q}\n"
        # Use the decorator logic to check if the question requires a special calculation
        response = model_response(q, machine_ids, rag_sentences)
        # print('response', response)
        if "total" in q.lower() or "sum" in q.lower() or "add" in q.lower():
            if response.startswith("The"):
                # Skip further process if calculation is done
                print(response)
                continue

        # Append additional instructions
        prompt += f"VERY IMPORTANT: you must take into account all {num_of_machines} machines and their respective data in the context OTHERWISE I WILL LOSE MY JOB"
        prompt += 'DO NOT MIX UP THE VALUES ACROSS THE MACHINES! \n\n'

        # Get the response from the LLM
        for chunk in send_prompt(prompt=prompt, interface="ollama", temperature=0):
            print(chunk, end='', flush=True)
        print("\n\n\n")
