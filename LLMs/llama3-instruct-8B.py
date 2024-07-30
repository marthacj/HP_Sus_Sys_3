# This file is for testing Llama3-Instruct-8B model with different prompts, contexts, RAGS.
from llama_cpp import Llama
import ollama
import pandas as pd
import yaml
import sys
from collections import OrderedDict
# from data.csv_to_json import csv_to_json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
# from langchain_core.tools import tool
# from langchain import hub
# from langchain.agents import AgentExecutor, create_openai_tools_agent
# from langchain_openai import ChatOpenAI
# from ollama import OllamaModel, OllamaAgent, OllamaAgentExecutor

# pushed to git so need new key:
# KEY = """sk-proj-oAyTGcVCbFbpWLFhHsPST3BlbkFJy11eIi8CryUucNeXqvZg"""
# from langchain_core.prompts import PromptTemplate


# Model path for llama-3-8B-instruct gguf model
model_path = r"C:\Users\martha.calder-jones\OneDrive - University College London\UCL_comp_sci\Sustainable_Systems_3\HP_Sus_Sys_3\models\Meta-Llama-3-8B-Instruct.Q5_0.gguf"

# Initialize the LlamaCPP model, removing the max_new_tokens parameter and adjusting the context window
llm = Llama(
        model_path=model_path,
#         temperature=0.1,
          n_ctx=16000,
          n_gpu_layers=-1,
          verbose=True,
)
# "Can you tell me how much carbon emission is produced by a 24 core machine, using on average 0.08% central processing unit utilisation?"
# Define the standard set of questions which will be used across all tests
questions = ["Can you tell me how much carbon emission is produced by machine ld71r18u44dws?", 
                "How much is the total carbon emissions for all the machines?", 
                "Which is the machine that uses GPU most intensively on average?", 
                "Give me a summary of the central processing unit usage for all the machines",
                "Which of the machines do you recommend being moved to the next level up of compute power and why?",
                "What is the central processing unit average utilisation for each machine?",
                "What machine has the highest carbon emission value?"]

few_shots = [ 

              ("How much is the carbon footprint of all the machines?", 
              "Look in <BACKGROUND> and add together all the 'machine-carbon-emission-value' values for each 'machine-id'"),
                ("Which is the machine that uses GPU most intensively on average?", 
            "Look in <USAGE> and find the 'machine-id' with the highest 'GPU average usage'"),
                ("Give me a summary of the compute usage for all the machines",
                "Look in <USAGE> and find the 'CPU average usage' and 'GPU average usage' for each 'machine-id'")
                
                ]

def send_prompt(prompt: str, interface: str = "ollama", 
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

# ============================================================================================
"""code for rag experiment"""
# Function to read sentences from a text file
def read_sentences_from_file(file_path):
    with open(file_path, 'r') as file:
        sentences = file.readlines()
    # Strip any surrounding whitespace from each sentence
    sentences = [sentence.strip() for sentence in sentences]
    return sentences

"""agent experiments"""
# prompt = hub.pull("hwchase17/openai-tools-agent")

try:
    # Load the pre-trained BERT-based model
    # model = SentenceTransformer('bert-base-nli-mean-tokens')
    # model = SentenceTransformer('all-mpnet-base-v2')
    model = SentenceTransformer('multi-qa-mpnet-base-cos-v1')

    # Path to your data.txt file
    file_path = r'C:\Users\martha.calder-jones\OneDrive - University College London\UCL_comp_sci\Sustainable_Systems_3\HP_Sus_Sys_3\data.txt'

    # Read sentences from the file
    sentences = read_sentences_from_file(file_path)
    
    # sentences += ['CPU','cpu','CPU CPU','cpu cpu','something cpu','something cpu something','CPU CPU CPU','cpu cpu cpu','CPU CPU CPU CPU','cpu cpu cpu cpu','CPU CPU CPU CPU CPU something','something cpu cpu cpu cpu cpu','CPU CPU CPU CPU CPU CPU','cpu cpu cpu cpu cpu cpu','CPU CPU CPU CPU CPU CPU CPU','cpu cpu cpu cpu cpu cpu cpu','CPU CPU CPU CPU CPU CPU CPU CPU','cpu cpu cpu cpu cpu cpu cpu cpu','CPU CPU CPU CPU CPU CPU CPU CPU CPU','cpu cpu cpu cpu cpu cpu cpu cpu cpu']
    import os, pickle
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

q = questions[0]
"""questions[5]"""
q_embedding = model.encode(q)
print(q_embedding.shape)
q_embedding = q_embedding.reshape(1, -1)
""" want to reduce out prompt context by 75%"""
top_k = int(0.25 * len(sentences))
distances, indices = index.search(q_embedding, top_k)
print("Question:", q    )
"""for ind, distance in zip(indices[0], distances[0]):
    print(f"Distance: {distance}")
    print(f"Sentence: {sentences[ind]}\n")"""
prompt = """Here is your context for a question I will ask you:\n"""
returns = [[sentences[i], d] for i, d in zip(indices[0], distances[0])]
for r in returns:
    print(r)
input("Press Enter to continue...")
for ind, distance in zip(indices[0], distances[0]):
    prompt += f"{sentences[ind]}\n"
prompt += f"Here is a new question for you to answer:\n{q}"
print("prompt:", prompt)

response = send_prompt(prompt, interface="ollama")
print(response)

sys.exit()

# ============================================================================================

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
            child = {k: v for k, v in child.items() if k in ['timestamp', 'instance-type', 'sci']}
            """ convert timestamp to a UTC datetime not an object"""
            child['timestamp'] = pd.to_datetime(child['timestamp'], utc=True)
            """convert that to a string"""
            child['timestamp'] = child['timestamp'].strftime('%Y-%m-%d')
            """round sci to 6 dp"""
            child['sci'] = round(child['sci'], 6)
            """letters 7 to 10 are unique to each machine"""
            child['machine-id'] = str(i)
            machine_id_dict[machine[6:]] = str(i)
            """replace instance-type with machine-family, and sci with machine-carbon-emission-value"""
            child['machine-family'] = child.pop('instance-type')
            child['machine-carbon-emission-value'] = child.pop('sci')
            """machine_dict['machine-id-'+machine] = child"""
            machine_emissions_list.append(child)
        
    return machine_emissions_list, machine_id_dict


def rewrite_csv_input(cleaned_machine_usage_data: dict) -> dict:
    """rewrite the csv data to be more readable - with machine names instead of index"""
    machine_id_list = cleaned_machine_usage_data['machine-id-list']
    print("machine_id_list:", machine_id_list)
    num_machines = len(machine_id_list)
    cleaned_machine_usage_data.pop('machine-id-list')
    """remove any empty subdict"""
    for k, v in cleaned_machine_usage_data.copy().items():
        if not v:
            cleaned_machine_usage_data.pop(k)
    """ now insert machine name instead of index"""
    new_machine_usage_data = {}
    for k, v_subdict in cleaned_machine_usage_data.items():
        new_machine_usage_data[k] = {}
        for machine_index, v in v_subdict.items():
            """important: this excludes some rows from the csv - TODO: check they're not useful or important to LLM"""
            if machine_index < num_machines:
                new_machine_usage_data[k][machine_id_list[machine_index][6:]] = v
    return new_machine_usage_data
     
        
def flip_data_dict(data_dict):
    # Initialize an empty dictionary to hold the flipped data
    flipped_data = {}

    # Get the length of the data by checking the length of the first nested OrderedDict
    data_length = len(next(iter(data_dict.values())))

    # Iterate over the indices of the data points
    for idx in range(data_length):
        flipped_data[idx] = {}
        for category, values in data_dict.items():
            # Some values might be single keys instead of OrderedDicts
            if isinstance(values, OrderedDict):
                if idx in values:
                    flipped_data[idx][category] = values[idx]
            else:
                if idx == 0:
                    flipped_data[idx][category] = values

    return flipped_data

def load_data_files(return_yaml: bool = False) -> tuple:
    # Load the YAML file which contains the carbon emissions data for the machines
    yaml_file = r'C:\Users\martha.calder-jones\OneDrive - University College London\UCL_comp_sci\Sustainable_Systems_3\HP_Sus_Sys_3\manifest1\outputs\NEW_z2_G4_Sci_Scores_Output.yaml'
    # yaml_file = r'C:\Users\martha.calder-jones\OneDrive - University College London\UCL_comp_sci\Sustainable_Systems_3\HP_Sus_Sys_3\manifest1\outputs\z2_G4_Sci_Scores_Output.yaml'
    # Load yaml file
    with open(yaml_file, 'r') as f:
        """this data is what is put in <BACKGROUND> tag in the prompt"""
        emissions_reference_data = yaml.load(f, Loader=yaml.SafeLoader)
        emissions_reference_data_str = yaml.dump(emissions_reference_data)
        # split by first occureance of word defaults and take [1]
        emissions_reference_data_str = emissions_reference_data_str.split('pipeline', 1)[1]
        
        
    # Load the Excel file of actual cpu usage (cpu / gpu etc average utilisation / data transfer over the course of X time). File here is stored in codebase 
    file_path = r'C:\Users\martha.calder-jones\OneDrive - University College London\UCL_comp_sci\Sustainable_Systems_3\HP_Sus_Sys_3\data\1038-0610-0614-day.xlsx'
    try:
        machine_usage_data = pd.read_excel(file_path, skiprows=4)
        

    except Exception as e:
        print(f"Error reading the Excel file: {e}")
        exit()
    # convert to dictionary
    machine_usage_data = machine_usage_data.to_dict()
    
    """remove all keys and values from machine_usage_data dict where the value is nan"""
    cleaned_machine_usage_data = {}
    for mud_k, mud_v in machine_usage_data.items():
        cleaned_machine_usage_data[mud_k] = {k: v for k, v in mud_v.items() if str(v) != 'nan'}
        """sort by key"""
        cleaned_machine_usage_data[mud_k] = OrderedDict(sorted(cleaned_machine_usage_data[mud_k].items()))
    """machine list key is 'Unnamed: 0'"""
    cleaned_machine_usage_data['machine-id-list'] = cleaned_machine_usage_data.pop('Unnamed: 0')
    """select only chars 7 onwards for machine id list value using dict comprehension"""
    cleaned_machine_usage_data['machine-id-list'] = {k: v[6:] for k, v in cleaned_machine_usage_data['machine-id-list'].items()}
    

    """go through all levels of the dict and if an element is numerical round it to 6 dp"""
    for mud_k, mud_v in cleaned_machine_usage_data.items():
        for k, v in mud_v.items():
            if isinstance(v, float):
                cleaned_machine_usage_data[mud_k][k] = round(v, 6)
    """remove any empty subdict"""
    for k, v in cleaned_machine_usage_data.copy().items():
        if not v:
            cleaned_machine_usage_data.pop(k)
    """move machine names into each dict"""
    """    cleaned_machine_usage_data = rewrite_csv_input(cleaned_machine_usage_data)               
    """  
    machine_usage_data = str(cleaned_machine_usage_data)
    # machine_usage_data = flip_data_dict(cleaned_machine_usage_data)
  

    # machine_usage_data = str(machine_usage_data)
    """replace all ravw string \n with a space"""
    machine_usage_data = machine_usage_data.replace(r'\n', ' ')
    """replace 'avg' with 'average usage'"""
    machine_usage_data = machine_usage_data.replace('avg', 'average usage')
    

  
    if return_yaml:
        return emissions_reference_data, machine_usage_data
    return emissions_reference_data_str, machine_usage_data


def answer_question_with_data(question: int) -> str:
    emissions_reference_data, machine_usage_data = load_data_files(return_yaml=True)
    emissions_reference_data, machine_id_dict = extract_data_from_yaml(emissions_reference_data)

    emissions_reference_data = str(emissions_reference_data)

    for k in machine_id_dict:
        print(k, machine_id_dict[k])
        machine_usage_data = machine_usage_data.replace(k, machine_id_dict[k])
    print("machine_usage_data:", machine_usage_data)
    input("Press Enter to continue...")



    print("emissions_reference_data:", emissions_reference_data)
    input("Press Enter to continue...")
    print("machine_usage_data:", machine_usage_data)
    print("machine_id_dict:", machine_id_dict)
    input("Press Enter to continue...")

    
    prompt = f"""Here is some background information on machines and their carbon emissions:
                <BACKGROUND> {emissions_reference_data} </BACKGROUND>
                Here are details on usage for a number of machines:
                <USAGE DETAILS> {machine_usage_data} </USAGE DETAILS>
    """
    prompt += """Here are some methodologies you can use to answer questions using this data:
    """
    for i,eg in enumerate(few_shots):
        prompt += f"""Example Question {i}: {eg[0]} 
        """
        prompt += f"""Answer methodology: {eg[1]}
        """
    prompt += f"""
    Here is a new question for you to answer:
      {questions[question]}"""
    prompt += """ Break your answer down into steps and explain your reasoning. """
    print("prompt:", prompt)
    input("Press Enter to continue...")
    response = send_prompt(prompt, interface="ollama")
    return response

q = 3
print(f"ANSWERING QUESTION {q}:")
print(answer_question_with_data(q-1))
sys.exit()
     

def benchmark_no_data():
    # ============================================================================================
    # BENCHMARKING FOR LLAMA-INSTRUCT-8B MODEL, NO INPUT DATA & ONLY RECOMMENDED PROMPT FOR NO CONTEXT
    # ============================================================================================
    # Define the prompt template: this is the recommended, basic template for llama3-instruct-8B model for true benchmarking 
    
    template = """
            <|begin_of_text|>
            <|start_header_id|>system<|end_header_id|>
            {system_prompt}
            <|eot_id|>
            <|start_header_id|>user<|end_header_id|>
            {user_prompt}
            <|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>
            """
    # Construct the PromptTemplate object
    prompt_template = PromptTemplate(template=template)

    # this is the basic system prompt
    system_prompt = """This is a chat between a user and an artificial intelligence assistant. 
                    The assistant gives helpful, detailed, and polite answers to the user's questions. 
                    The assistant should also indicate when the answer is not known."""

    for question in questions:
        # Format the prompt with the question - but not sure what prompt object we are going for?
        formatted_prompt = prompt_template.format(user_prompt=question, system_prompt=system_prompt)

        # Generate the response from the model (removed complete())
        print('formatted prompt:', formatted_prompt)
        response = send_prompt(formatted_prompt, interface="ollama")
        """response = llm(prompt=formatted_prompt, max_tokens=1024)"""
        # Print the response
        print(response)


# The below functions are for benchmark_csv_yaml_data and are used to generate sentences from the csv and yaml data
def generate_list_of_machines(row):
            machine = row['Unnamed: 0']
            return machine

def generate_list_of_carbon_per_z2_mini(machines_list):
            list_of_carbon_data_per_z2_mini = []
            for machine in machines_list:
                if machine in emissions_reference_data['tree']['children']['child']['children']['z2 mini']['children']:
                    print(machine)
                    for child in emissions_reference_data['tree']['children']['child']['children']['z2 mini']['children'][machine]['outputs']:
                        list_of_carbon_data_per_z2_mini.append(child)
                        # print(child)
            return list_of_carbon_data_per_z2_mini
        

def generate_list_of_carbon_per_Z4R_G4(machines_list):
    list_of_carbon_data_per_Z4R_G4 = []
    for machine in machines_list:
        if machine in emissions_reference_data['tree']['children']['child']['children']['Z4R G4']['children']:
            print(machine)
            for child in emissions_reference_data['tree']['children']['child']['children']['Z4R G4']['children'][machine]['outputs']:
                list_of_carbon_data_per_Z4R_G4.append(child)
                # print(child)
    return list_of_carbon_data_per_Z4R_G4


def generate_sentences_for_Z4R_G4(list_of_carbon_data_per_Z4R_G4):
    """Function to generate sentences for Z4R G4 machines' carbon data."""
    sentences = []
    for machine in list_of_carbon_data_per_Z4R_G4:
        sentence_with_tag = f"""
                    <DURATION-IN-SECONDS>{machine['duration']}</DURATION-IN-SECONDS>.
                    <MACHINE>{machine['machine-code']}</MACHINE><MACHINE-FAMILY>{machine['instance-type']}</MACHINE-FAMILY>.
                    <INFO>All of the z2 mini machines and the Z4R G4 machines make up one pool of units</INFO>.
                    <SCI-CARBON-EMISSION-VALUE>{machine['sci']}</SCI-CARBON-EMISSION-VALUE>."""
        sentences.append(sentence_with_tag)
    print(sentences)
    return sentences


def generate_sentences_for_z2_mini_carbon_data(list_of_carbon_data_per_z2_mini):
    """Function to generate sentences for z2 mini machines' carbon data."""
    sentences = []
    for machine in list_of_carbon_data_per_z2_mini:
        sentence_with_tag = f"""
                    <DURATION-IN-SECONDS>{machine['duration']}</DURATION-IN-SECONDS>.
                    <MACHINE>{machine['machine-code']}</MACHINE><MACHINE-FAMILY>{machine['instance-type']}</MACHINE-FAMILY>.
                    <INFO>All of the z2 mini machines and the Z4R G4 machines make up one pool of units</INFO>.
                    <SCI-CARBON-EMISSION-VALUE>{machine['sci']}</SCI-CARBON-EMISSION-VALUE>."""
        sentences.append(sentence_with_tag)
    return sentences


def generate_sentences_for_all_machines_carbon_data(machines_list):
    """Function to generate sentences for all machines' carbon data."""
    carbon_data_for_all_machines = []
    list_of_carbon_data_per_Z4R_G4 = generate_list_of_carbon_per_Z4R_G4(machines_list)
    list_of_carbon_data_per_z2_mini = generate_list_of_carbon_per_z2_mini(machines_list)
    sentences_Z4R_G4 = generate_sentences_for_Z4R_G4(list_of_carbon_data_per_Z4R_G4)
    sentences_z2_mini = generate_sentences_for_z2_mini_carbon_data(list_of_carbon_data_per_z2_mini)
    carbon_data_for_all_machines.extend(sentences_Z4R_G4)
    carbon_data_for_all_machines.extend(sentences_z2_mini)
    return carbon_data_for_all_machines


def generate_sentence(row):
    """Function to generate sentences from the CSV DataFrame rows - this is on CPU utilisation etc NOT carbon emissions"""
    cpu_avg = row['CPU\nHighest\navg']
    highest_core_average = row['Core\nHighest\navg']
    core_over_80_seconds = row['Core \nTotal Seconds > 80%']
    core_division_result = float(core_over_80_seconds / 216000 * 100)
    cpu_over_80_seconds = row['\nCPU\nTotal Seconds > 80%']
    cpu_division_result = float(cpu_over_80_seconds / 216000 * 100)
    total_memory_capacity = row['Total RAM\n(GB)']
    MB_sent_per_sec = row['send avg\nMB/Sec']
    MB_received_per_sec = row['receive avg\nMB/Sec']
    total_MB_sent = row['Total MB\nSent']
    total_MB_received = row['Total MB\nReceived']
    GPU_min_percentage = row['GPU\nmin']
    GPU_max_percentage = row['GPU\nmax']
    GPU_average_percentage = row['GPU\navg']
    GPU_over_80_occurrences = row['GPU\n#oc > 80%']
    GPU_mem_min = row['MEM\nmin']
    GPU_mem_max = row['MEM\nmax']
    GPU_mem_avg = row['MEM\navg']
    GPU_mem_over_80_occurrences = row['MEM\n#oc > 80%']
    sentence_with_tag = f"""<DURATION-SECONDS>216000</DURATION-SECONDS>.
                    <MACHINE>{row['Unnamed: 0']}</MACHINE>.
                    <CPU-AVERAGE-UTILISATION-PERCENTAGE>{cpu_avg}</CPU-AVGERAGE-UTILISATION-PERCENTAGE>.
                    <CPU-PERCENTAGE-OF-DURATION-OVER-80-PERCENT-UTILISATION>{cpu_division_result}</CPU-PERCENTAGE-OF-DURATION-OVER-80-PERCENT-UTILISATION>.
                    <CORE-AVERAGE-UTILISATION-PERCENTAGE>{highest_core_average}</CORE-AVERAGE-UTILISATION-PERCENTAGE>.
                    <CORE-PERCENTAGE-OF-DURATION-OVER-80-PERCENT-UTILISATION>{core_division_result}</CORE-PERCENTAGE-OF-DURATION-OVER-80-PERCENT-UTILISATION>.
                    <AVERAGE-MEMORY-UTILISATION>{row['avg']}</AVERAGE-MEMORY-UTILISATION>.
                    <MEMORY-PERCENTAGE-OF-DURATION-OVER-80-PERCENT-UTILISATION>{row['#oc > 80%']}</MEMORY-PERCENTAGE-OF-DURATION-OVER-80-PERCENT-UTILISATION>.
                    <NETWORK-TRAFFIC-TOTAL-MB-SENT>{total_MB_sent}</NETWORK-TRAFFIC-TOTAL-MB-SENT>.
                    <NETWORK-TRAFFIC-TOTAL-MB-RECEIVED>{total_MB_received}</NETWORK-TRAFFIC-TOTAL-MB-RECEIVED>.
                    <GPU-AVERAGE-UTILISATION-PERCENTAGE>{GPU_average_percentage}</GPU-AVERAGE-UTILISATION-PERCENTAGE>.
                    <GPU-MEMORY-UTILISATION-AVERAGE-PERCENTAGE>{GPU_mem_avg}</GPU-MEMORY-UTILISATION-AVERAGE-PERCENTAGE>.
                    <GPU-MEMORY-UTILISATION-MAXIMUM-PERCENTAGE>{GPU_mem_max}</GPU-MEMORY-UTILISATION-MAXIMUM-PERCENTAGE>."""
    return sentence_with_tag

# calling functions and generating global variable sentences for the data on carbon emissions and machine usage 
machines_list = machine_usage_data.apply(generate_list_of_machines, axis=1)

sentences_for_all_machines_carbon = generate_sentences_for_all_machines_carbon_data(machines_list)

sentences_for_all_machines_usage = machine_usage_data.apply(generate_sentence, axis=1)


def benchmark_csv_yaml_data():
        # ================================================================================================================
        # CSV AND YAML DATA FOR LLAMA3 MODEL, SLIGHT TWEAK OF PROMPT BUT NOTHING RESEARCHED. Reason for this is that 
        # there may be some tweaking that can be done to the data initially to make it more readable for the model.
        # ================================================================================================================
        # One method for including the csv and yaml data is to generate tagged 'sentences' from the data, another is to use the raw data.

        # generate all written context which can be added to prompt
        context = (sentences_for_all_machines_carbon, sentences_for_all_machines_usage)

        template = """
            This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers 
            to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context.
            You should look across both sources and combine the information that relates to the same machine. 
            E.g.  ld71r18u44ews might appear as a listed machine in both sources and should be treated as referrring to the same machine across both sources. 
            You should consider the carbon emissions produced by the machines in the context of their compute usage.
            You should be able to recognise which machine IDs make up a family of the same type of machine 
            e.g. z2 mini family or Z4R G4 family. There can be a mixture of these machines in the same pool or workstation.
            Context: {context}
            Question: {question}  
            Answer:  """
        
        # Again, are we using a langchain object? How best to actually get the prompt out / interact with the model? 
        prompt_template = PromptTemplate(template=template)

        # No idea if the following is best way to go about it? 
        for question in questions:
        # Format the prompt with the question
            formatted_prompt = prompt_template.format(question=question, context=context)
            print("formatted prompt:", formatted_prompt)
            print(len(formatted_prompt.split()))
            # Generate the response from the model
            response = send_prompt(formatted_prompt, interface="ollama")
            """response = llm(prompt=formatted_prompt, max_tokens=1024)"""
            # Print the response
            print(response)


def prompt_engineered():
    template_input = input("""Choose the template for the prompt:
                            Template 1
                            Template 2
                            Template 3
                            """)
    if template_input.lower() == 'template 1':
        system_prompt = """
                    You are a helpful assistant with expertise in carbon emissions and compute usage."""
        pass
    if template_input.lower() == 'template 2':
        pass
    if template_input.lower() == 'template 3':
        pass
    pass


def rag():
    pass

"""manually load data files"""
emissions_reference_data, machine_usage_data = load_data_files(return_yaml=True)
"convert yaml to dictionary"
emissions_reference_data = extract_data_from_yaml(emissions_reference_data)
import pprint
pprint.pprint(emissions_reference_data)
sys.exit()

# Allowing the user to choose testing of the model with two optimisation methdos.
while True:
    user_input = input("This is Llama3-Instruct-8B Model. Enter\n [1] Benchmark no data\n [2] Benchmark CSV & YAML data \n [3] Prompt Engineered \n [4] RAG\n\n")
    if user_input == '1':
        # General behaviour of the model out of the box
        benchmark_no_data()

    elif user_input == '2':
        # cleaned data with only relevant data, correct labelling, removed NaNs etc.
        benchmark_csv_yaml_data()

    elif user_input == '3':
        prompt_engineered()

    elif user_input == '4':
        rag()

    elif user_input.lower() == 'exit':
        break

    else:
        print("Invalid input. Please enter '1', '2', or '3', '4', or 'Exit.")

print("Bye.")