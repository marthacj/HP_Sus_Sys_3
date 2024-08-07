# import unittest
# from unittest.mock import patch
# import logging
# from LLM import *

# import pandas as pd

# # Example DataFrame structure
# data = {
#     'question': [
#         'Can you tell me how much carbon emission is produced by machine ld71r18u44dws?', 
#                 "How much is the total carbon emissions for all the machines?", 
#                 "Which is the machine that uses GPU most intensively on average?", 
#                 "Give me a summary of the central processing unit usage for all the machines",
#                 "Which of the machines do you recommend being moved to the next level up of compute power and why?\n",
#                 "What is the central processing unit average utilisation for each machine?",
#                 "What machine has the highest carbon emission value?" 
#     ],
#     'expected_answer': [
#         '5000 USD', 
#         '3000 gCO2e', 
#         # ... other answers
#     ]
# }

# df = pd.DataFrame(data)


# def extract_json_from_response(response):
#     try:
#         json_response = response[response.find('['):response.rfind(']') + 1]
#         return json_response
#     except Exception as e:
#         logging.error(f"Error extracting JSON: {e}")
#         return None

# class TestGenerateQuestion(unittest.TestCase):

#     @patch('builtins.input', side_effect=['1', '2', '1', '0', 'bye'])
#     def test_generate_question(self, mock_input):
#         index = None  # Replace with actual index
#         embeddings = None  # Replace with actual embeddings
#         model = None  # Replace with actual model
#         sentences = [
#             "Machine ld71r18u44dws has carbon emissions of 513.44 gCO2eq.",
#             "Machine ld71r16u14ws has carbon emissions of 432.24 gCO2eq.",
#             "Machine ld71r16u13ws has carbon emissions of 436.22 gCO2eq.",
#             "Machine ld71r16u15ws has carbon emissions of 435.24 gCO2eq.",
#             "Machine ld71r18u44ews has carbon emissions of 515.85 gCO2eq."
#         ]
#         questions = [
#             "\n \n \n [0] Can you tell me how much carbon emission is produced by machine ld71r18u44dws?\n",
#             "[1] How much is the total carbon emissions for all the machines?\n",
#             "[2] Which is the machine that uses GPU most intensively on average?\n",
#             "[3] Give me a summary of the central processing unit usage for all the machines\n",
#             "[4] Which of the machines do you recommend being moved to the next level up of compute power and why?\n",
#             "[5] What is the central processing unit average utilisation for each machine?\n",
#             "[6] What machine has the highest carbon emission value?\n"
#         ]
        
#         generate_question(index, embeddings, model, sentences, questions)
        
# if __name__ == '__main__':
#     unittest.main()
