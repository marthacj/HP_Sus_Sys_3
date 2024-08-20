# ================================================================================================================================================================= 
                # The following step was removed as the prior if statement takes away the need for it
                # # Step 3 - check if the question is a simple calculation or not (LLMs cannot do simple calculations)
                #prompt = """I have a problem I need to solve and I need your advice on how best to solve it.
                # Should I use only a set of database lookups on the context I provided to answer the question, or instead write python code to do a calculation?
                # The database contains the following information:"""
                # prompt += response + "\n"
                # prompt += f"Here is a problem I need to solve:\n{q}\n"
                # prompt += '''Should I use a database lookup or write python code to do a simple calculation to solve the problem?
                # Please respond only with 'database lookup' or 'simple calculation'. Do not include Python code or any 
                # other information in your response.'''

                # # print(f"***Prompt for simple calculation check: {prompt}***")
                # response = send_prompt(llm, prompt, interface="ollama")
                # # print(f"***Response to simple calculation check: {response}***")
                
                # if response.lower().strip() == 'database lookup':
                #     prompt = "Here is your context for a question I will ask you:\n"
                #     j_dict_list = eval(json_response)
                    
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

                #     prompt += f"Here is a question for you to answer using the above context:\n{q}\n"
                #     #prompt += appendix_prompt
                #     print(f"***Prompt for database lookup: {prompt}***")
                #     response = send_prompt(llm, prompt, interface="ollama", temperature=0.5)
                #     print(response)
                #     continue