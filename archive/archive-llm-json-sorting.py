 #     prompt += '''VERY IMPORTANT:  Return to me, in JSON format, all the data from the context above to answer the question. 
            #     The JSON format should be as follows:
            #     [
            #         {
            #             "machine": <machine id>,
            #             <data-field0>: <data-field0 value>,
            #             <data-field1>: <data-field1 value>,
            #             etc.
            #         }
            #     ]
            #     The data field keys should ONLY be an exactly copied label (not an abbreviation or reduction) from the context I provided and the values should be the actual values from the context I provided.
            #     VERY IMPORTANT: There are ''' + num_of_machines + ' machines in total so your list must have ' + num_of_machines + ' dictionaries - Check the context properly. Do not leave any out or I will LOSE MY JOB if not all ' + num_of_machines + ''' are included.
            #    \n\n THIS IS VERY IMPORTANT: If a value is not in the context, do not include it in the JSON. DO NOT INCLUDE NULL VALUES IN THE JSON.'''
                # prompt += 'DO NOT MIX UP THE VALUES ACROSS THE MACHINES! \n\n'
            #    
                # prompt += "\nHere is that context again:\n"
                # for ind in indices[0]:
                #     if 0 <= ind < len(sentences):
                #         prompt += f"{sentences[ind]}\n"
                #     else:
                #         print(f"Warning: Index {ind} is out of range.")
                # prompt += "\n\n THIS IS VERY IMPORTANT: DO NOT ALTER ANY ZERO VALUES. If a value is '0', keep it as 0. If a value is not in the context, do not include it in the JSON."
                # prompt += '\n\n Reminder: VERY IMPORTANT: There are ' + num_of_machines + ' machines in total so your list must have ' + num_of_machines + ' dictionaries - Check the context properly. Do not leave any out or I will LOSE MY JOB if not all ' + num_of_machines + ' are included.'
            #     print("prompt:", prompt)
            #     # sys.exit()
            #     response = send_prompt(prompt, interface="ollama")
            #     # print(response)
            #     json_response = response    
            #     # remove any pre-amble or post comment from llm by getting location of first [ and last ]
            #     # json_response = json_response[json_response.find('['):json_response.rfind(']')+1]
            #     json_response = extract_json_from_response(response)
            #     if json_response is None:
            #         raise ValueError("No valid JSON found in the response.")
            #     # prompt+= json_response
            #     prompt = "Here is your context for a question I will ask you:\n"
            #     prompt+= json_response
            #     print("prompt:", prompt)
            #     j_dict_list = json.loads(json_response)
                
            #     # Remove null values from the parsed JSON
            #     for d in j_dict_list:
            #         keys_to_remove = [k for k, v in d.items() if v is None]
            #         for k in keys_to_remove:
            #             del d[k]

            # #   shuffling the values in the final context so that 1) maybe smiilar figures wont be next to each other and 2) e.g. the highest value wont always be in same place for a dataset 
            #     d_list = []
            #     for d in j_dict_list:
            #         d_sub_list = []
            #         for k, v in d.items():
            #             d_sub_list.append((k,v))
            #         d_list.append(d_sub_list)
            #     random.shuffle(d_list)
            #     for d in d_list:
            #         d_dict = dict(d)
            #         for k, v in d_dict.items():
            #             prompt += f"{k}: {v}\n"
            #         prompt += "\n"
                # prompt += f"Here is a question for you to answer using the above context:\n{q}\n"

