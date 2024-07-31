
import os
from eldar import Query, Index
import inspect
from typing import get_type_hints
import ollama
from autogen import AssistantAgent, UserProxyAgent
from autogen.coding import LocalCommandLineCodeExecutor
import sys

def generate_function_metadata(func):
    """
    Takes as input a function and generates json metadata to be used 
    for an autogent function calling agent"""
    signature = inspect.signature(func)
    parameters = signature.parameters
    hints = get_type_hints(func)
    
    properties = {}
    required = []

    for name, param in parameters.items():
        param_type = hints.get(name, str)
        properties[name] = {
            'type': param_type.__name__,
            'description': f"The {name} parameter",
        }
        required.append(name)

     # or f"{func.__name__} function",

    """do not leave out docstring!"""
    metadata = {
        'name': func.__name__,
        'description': func.__doc__,
        'parameters': {
            "type": "object",
            "properties": properties,
            "required": required
        }
    }

    return metadata



def boolean_search_text(query: str) -> list[str]:
    """
    This function allows the searching of the text data using
    a standard boolean search query. The function returns a list
    of results that match the query.
    For example, a query of "carbon AND (emissions OR cats)" will return
    all documents that contain the word "carbon" and either "emissions" or "cats".
    """
    if not os.path.exists('data_index.p'):
        index.save('data_index.p')
    else:
        with open('data.txt', 'r') as f:
            data = f.read()
        documents = data.split('\n')
        index = Index(ignore_case=True)
        index.build(documents)
    results = index.search(query)
    return results

    
md = generate_function_metadata(boolean_search_text)

llm_config = {
    "model": "gpt-4o",
    "api_key": "", 
    "functions": [md]
}

assistant = AssistantAgent(
    name="assistant",
    llm_config=llm_config,
    system_message="You are a helpful assistant that can construct multiple advanced boolean search queries for text data."
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode = "NEVER",
    max_consecutive_auto_reply=5,
    is_termination_msg=lambda x: x.get("content","").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "executor": LocalCommandLineCodeExecutor()
    }
)
functions = [boolean_search_text]
for func in functions:
    assistant.register_for_llm(description=func.__doc__)(func)


task = "Can you find all CPU info for machine 8u44dws?"
user_proxy.initiate_chat(
    assistant=assistant,
    task=task,
    recipient=assistant
  
)
