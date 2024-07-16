# This file is for testing Llama3-Instruct-8B model with different prompts, contexts, RAGS.
from llama_cpp import Llama
import ollama
import pandas as pd
import yaml
import sys
from langchain_core.prompts import PromptTemplate


# Model path
model_path = r"C:\Users\martha.calder-jones\OneDrive - University College London\UCL_comp_sci\Sustainable_Systems_3\HP_Sus_Sys_3\models\Meta-Llama-3-8B-Instruct.Q5_0.gguf"

# Initialize the LlamaCPP model, removing the max_new_tokens parameter and adjusting the context window
llm = Llama(
        model_path=model_path,
#         temperature=0.1,
          n_ctx=16000,
          n_gpu_layers=-1,
          verbose=True,
)

# Define the standard set of questions which will be used across all tests
questions = ["Can you tell me how much carbon emission is produced by a 24 core machine which is using on average 0.0807847747114039% CPU utilisation?", 
                "How much is the carbon footprint of the pool or a single unit within the pool over the last month/week?", 
                "Which is the unit that uses GPU most intensively?", 
                "Give me a summary of the compute usage within the unit's pool over the last day/week/month?",
                    "Which of the machines in the pool do you recommend being moved to the next level up of compute power and why  based on CPU usage?"]


def send_prompt(prompt: str, interface: str = "ollama", 
                max_tokens: int = 1024) -> str:
    if interface == "ollama":
        response = ollama.generate(model="llama3",
                                   prompt=prompt, keep_alive='24h')
        response = response['response']
    else: 
        # assume llama_cpp
        response = llm(prompt=prompt, max_tokens=max_tokens)
        response = response['choices'][0]['text']
    return response


def load_data_files() -> tuple[str,str]:
    # Load the YAML file which contains the carbon emissions data for the machines
    yaml_file = r'C:\Users\martha.calder-jones\OneDrive - University College London\UCL_comp_sci\Sustainable_Systems_3\HP_Sus_Sys_3\manifest\outputs\z2_G4_Sci_Scores_Output.yaml'
    # Load yaml file
    with open(yaml_file, 'r') as f:
        emissions_reference_data = yaml.load(f, Loader=yaml.SafeLoader)
        emissions_reference_data_str = yaml.dump(emissions_reference_data)
        # split by first occureance of word defaults and take [1]
        emissions_reference_data_str = emissions_reference_data_str.split('defaults', 1)[1]
        print(emissions_reference_data_str)
        
    # Load the Excel file of actual cpu usage (cpu / gpu etc average utilisation / data transfer over the course of X time). File here is stored in codebase 
    file_path = r'C:\Users\martha.calder-jones\OneDrive - University College London\UCL_comp_sci\Sustainable_Systems_3\HP_Sus_Sys_3\data\1038-0610-0614-day.xlsx'
    try:
        machine_usage_data = pd.read_excel(file_path, skiprows=4)
        # convert to dictionary
        machine_usage_data = str(machine_usage_data.to_dict())

    except Exception as e:
        print(f"Error reading the Excel file: {e}")
        exit()
    return emissions_reference_data_str, machine_usage_data


# Calling the function to load the data files and setting the global variables
emissions_reference_data, machine_usage_data = load_data_files()

def answer_question_with_data(question: int) -> str:
    prompt = f"""Here is some background information on how processer usage and carbon emissions are related:
                <BACKGROUND> {emissions_reference_data} </BACKGROUND>
                Here are details on usage for a number of machines:
                <USAGE DETAILS> {machine_usage_data} </USAGE DETAILS>
    """
    prompt += f"""Question: {questions[question]}"""
    response = send_prompt(prompt, interface="ollama")
    return response

q = 5
print(f"ANSWERING QUESTION {q}:")
print(answer_question_with_data(q-1))
input("Press Enter to continue...")
     

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
        sentence = f"""
                    This machine is the {machine['machine-code']} and is a part of the {machine['instance-type']} machine family. All of the z2 mini machines and the Z4R G4 machines make up one pool of units.
                    This {machine['machine-code']} machine's CPU used {machine['cpu-wattage-times-duration']} Watts over the course of {machine['duration']} seconds, which is 2.5 days.
                    This {machine['machine-code']} machine's GPU used {machine['gpu-wattage-times-duration']} Watts over the course of {machine['duration']} seconds, which is 2.5 days.
                    This {machine['machine-code']} machine's embodied emissions (carbon emissions produced during the manufacturing of the machine) are {machine['device/emissions-embodied']} gCO2e.
                    The amount of embodied carbon emissions allocated for the {machine['duration']} seconds that this machine was running is {machine['carbon-embodied']} gCO2.
                    This {machine['machine-code']} machine's operational emissions (carbon emissions produced during the operation of the machine) are {machine['carbon-operational']} gCO2e.
                    In total, the carbon emissions produced by machine {machine['machine-code']} is {machine['carbon']} gCO2e over {machine['duration']} seconds."""
        # different approach to formatting the sentence
        sentence = f"""
                    <MACHINE>{machine['machine-code']}</MACHINE><MACHINE-FAMILY>{machine['instance-type']}</MACHINE-FAMILY>. 
                    <<INFO>All of the z2 mini machines and the Z4R G4 machines make up one pool of unitS.</INFO>
                    <MACHINE-CPU-USAGE-WATTS>{machine['cpu-wattage-times-duration']}</MACHINE-CPU-USAGE-WATTS>
                    {machine['duration']} seconds, which is 2.5 days.
                    This {machine['machine-code']} machine's GPU used {machine['gpu-wattage-times-duration']} Watts over the course of {machine['duration']} seconds, which is 2.5 days.
                    This {machine['machine-code']} machine's embodied emissions (carbon emissions produced during the manufacturing of the machine) are {machine['device/emissions-embodied']} gCO2e.
                    The amount of embodied carbon emissions allocated for the {machine['duration']} seconds that this machine was running is {machine['carbon-embodied']} gCO2.
                    This {machine['machine-code']} machine's operational emissions (carbon emissions produced during the operation of the machine) are {machine['carbon-operational']} gCO2e.
                    In total, the carbon emissions produced by machine {machine['machine-code']} is {machine['carbon']} gCO2e over {machine['duration']} seconds."""
        
        
        sentences.append(sentence)
    return sentences


def generate_sentences_for_z2_mini_carbon_data(list_of_carbon_data_per_z2_mini):
    """Function to generate sentences for z2 mini machines' carbon data."""
    sentences = []
    for machine in list_of_carbon_data_per_z2_mini:
        sentence = f"""
                    This machine is the {machine['machine-code']} and is a part of the {machine['instance-type']} machine family. All of the z2 mini machines and the Z4R G4 machines make up one pool of units.
                    This {machine['machine-code']} machine's CPU used {machine['cpu-wattage-times-duration']} Watts over the course of {machine['duration']} seconds, which is 2.5 days.
                    This {machine['machine-code']} machine's GPU used {machine['gpu-wattage-times-duration']} Watts over the course of {machine['duration']} seconds, which is 2.5 days.
                    This {machine['machine-code']} machine's embodied emissions (carbon emissions produced during the manufacturing of the machine) are {machine['device/emissions-embodied']} gCO2e.
                    The amount of embodied carbon emissions allocated for the {machine['duration']} seconds that this machine was running is {machine['carbon-embodied']} gCO2.
                    This {machine['machine-code']} machine's operational emissions (carbon emissions produced during the operation of the machine) are {machine['carbon-operational']} gCO2e.
                    In total, the carbon emissions produced by machine {machine['machine-code']} is {machine['carbon']} gCO2e over {machine['duration']} seconds."""
        sentences.append(sentence)
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
    # for sentence in carbon_data_for_all_machines:
        # print(sentence)
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
    sentence = f"""This data was taken over a timespan of 216000 seconds, which is 2.5 days.
                    Host machine {row['Unnamed: 0']} with {row['#Cores']} cores has {cpu_avg} average CPU utilisation.
                    The core in machine {row['Unnamed: 0']} with the highest utilisation rate was averaging {highest_core_average}%, 
                    and went above 80% core utilisation for a total of {core_over_80_seconds} seconds. 
                    The percentage {core_division_result}% ({core_over_80_seconds} / 216000) tells you whether this is a high percentage of the time or not.
                    The CPU in machine {row['Unnamed: 0']} which had the highest usage had an average of {cpu_over_80_seconds} seconds above 80% utilisation. 
                    The percentage {cpu_division_result}% ({cpu_over_80_seconds} / 216000) tells you whether this is a high percentage of the time or not.
                    The total memory capacity for machine {row['Unnamed: 0']} is {total_memory_capacity} GB, and its average memory utilisation is {row['avg']}. 
                    The amount of time machine {row['Unnamed: 0']} went above 80% memory utilisation was {row['#oc > 80%']}.
                    The network traffic statistics for the machine {row['Unnamed: 0']} is known: 
                    The average MB sent per second was {MB_sent_per_sec}, the average MB received per second was {MB_received_per_sec},
                    the total MB sent was {total_MB_sent}, the total MB received was {total_MB_received}. 
                    The GPU utilisation rate for machine {row['Unnamed: 0']} had a minimum of {GPU_min_percentage}%, a maximum of {GPU_max_percentage}%, and an average of {GPU_average_percentage}%,
                    the number of times machine  {row['Unnamed: 0']} went over a GPU utilisation rate of 80% was {GPU_over_80_occurrences}. 
                    The GPU memory utilisation for machine {row['Unnamed: 0']} had a minimum of {GPU_mem_min}%, a maximum of {GPU_mem_max}%, and an average of {GPU_mem_avg}%, 
                    and the number of times it went above an average of 80% was {GPU_mem_over_80_occurrences}."""
    return sentence

# calling functions and generating global variable sentences for the data on carbon emissions and machine usage 
machines_list = machine_usage_data.apply(generate_list_of_machines, axis=1)

sentences_for_all_machines_carbon = generate_sentences_for_all_machines_carbon_data(machines_list)

sentences_for_all_machines_usage = machine_usage_data.apply(generate_sentence, axis=1)

def benchmark_csv_yaml_data():
        # ================================================================================================================
        # CSV AND YAML DATA FOR LLAMA3 MODEL, SLIGHT TWEAK OF PROMPT BUT NOTHING RESEARCHED. Reason for this is that 
        # there may be some tweaking that can be done to the data initially to make it more readable for the model.
        # ================================================================================================================
        # One method for including the csv and yaml data is to generate sentences from the data and then include these in the prompt, another is to use the raw data.

  
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

# Allowing the user to choose testing of the model with three optimisation levels.
while True:
    user_input = input("This is Llama3-Instruct-8B Model. Enter\n [1] Benchmark no data\n [2] Benchmark CSV & YAML data \n [3] Prompt Engineered \n [4] RAG\n\n")
    if user_input == '1':
        benchmark_no_data()

    elif user_input == '2':
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