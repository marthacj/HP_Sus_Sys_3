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
        # sentence = f"""
        #             This machine is the {machine['machine-code']} and is a part of the {machine['instance-type']} machine family. All of the z2 mini machines and the Z4R G4 machines make up one pool of units.
        #             This {machine['machine-code']} machine's CPU used {machine['cpu-wattage-times-duration']} Watts over the course of {machine['duration']} seconds, which is 2.5 days.
        #             This {machine['machine-code']} machine's GPU used {machine['gpu-wattage-times-duration']} Watts over the course of {machine['duration']} seconds, which is 2.5 days.
        #             This {machine['machine-code']} machine's embodied emissions (carbon emissions produced during the manufacturing of the machine) are {machine['device/emissions-embodied']} gCO2e.
        #             The amount of embodied carbon emissions allocated for the {machine['duration']} seconds that this machine was running is {machine['carbon-embodied']} gCO2.
        #             This {machine['machine-code']} machine's operational emissions (carbon emissions produced during the operation of the machine) are {machine['carbon-operational']} gCO2e.
        #             In total, the carbon emissions produced by machine {machine['machine-code']} is {machine['carbon']} gCO2e over {machine['duration']} seconds."""
        # different approach to formatting the sentence with tags, and including less info
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
        # sentence = f"""
        #             This machine is the {machine['machine-code']} and is a part of the {machine['instance-type']} machine family. All of the z2 mini machines and the Z4R G4 machines make up one pool of units.
        #             This {machine['machine-code']} machine's CPU used {machine['cpu-wattage-times-duration']} Watts over the course of {machine['duration']} seconds, which is 2.5 days.
        #             This {machine['machine-code']} machine's GPU used {machine['gpu-wattage-times-duration']} Watts over the course of {machine['duration']} seconds, which is 2.5 days.
        #             This {machine['machine-code']} machine's embodied emissions (carbon emissions produced during the manufacturing of the machine) are {machine['device/emissions-embodied']} gCO2e.
        #             The amount of embodied carbon emissions allocated for the {machine['duration']} seconds that this machine was running is {machine['carbon-embodied']} gCO2.
        #             This {machine['machine-code']} machine's operational emissions (carbon emissions produced during the operation of the machine) are {machine['carbon-operational']} gCO2e.
        #             In total, the carbon emissions produced by machine {machine['machine-code']} is {machine['carbon']} gCO2e over {machine['duration']} seconds."""
        # different approach to formatting the sentence with tags, and including less info
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