import unittest
from unittest.mock import patch
import os
from CSV_upload import upload_file_to_application_directory
from manifest_generation import process_csv
import pandas as pd
import pytest
import yaml
from LLM import extract_data_from_yaml

class TestFileUpload(unittest.TestCase):
    """This class is used to test the functionality (TEST SCENARIO) of uploading a file to the application directory.
        It is made up of 7 test cases:
        Test Case 1: Default File Not Found 
        Test Case 2: Custom File Not Found
        Test Case 3: Copy Failure
        Test Case 4: Invalid File Path
        Test Case 5: Read Access Denied
        Test Case 6: Successful Upload and Read Access
        Test Case 7: Use Default File
    """

    @patch('builtins.input', return_value='')  # Simulate pressing Enter to use the default file
    @patch('os.path.isfile', return_value=True)
    @patch('os.makedirs')
    @patch('shutil.copy')
    @patch('os.access', return_value=True)
    def test_upload_default_file(self, mock_access, mock_copy, mock_makedirs, mock_isfile, mock_input):
        target_dir = 'data/uploaded_excel_files'
        default_file_path = 'data/1038-0610-0614-day-larger-figures-test.xlsx'
        expected_destination = os.path.join(target_dir, 'uploaded_file.xlsx')
        
        result = upload_file_to_application_directory(target_dir, default_file_path)
        self.assertEqual(result, expected_destination)
        mock_copy.assert_called_once_with(default_file_path, expected_destination)
        mock_makedirs.assert_not_called()

    @patch('builtins.input', return_value='custom_file.xlsx')
    @patch('os.path.isfile', side_effect=lambda x: x == 'custom_file.xlsx')
    @patch('os.makedirs')
    @patch('shutil.copy')
    @patch('os.access', return_value=True)
    def test_upload_custom_file(self, mock_access, mock_copy, mock_makedirs, mock_isfile, mock_input):
        target_dir = 'data/uploaded_excel_files'
        default_file_path = 'data/1038-0610-0614-day-larger-figures-test.xlsx'
        expected_destination = os.path.join(target_dir, 'uploaded_file.xlsx')
        
        result = upload_file_to_application_directory(target_dir, default_file_path)
        self.assertEqual(result, expected_destination)
        mock_copy.assert_called_once_with('custom_file.xlsx', expected_destination)

    @patch('builtins.input', return_value='')
    @patch('os.path.isfile', return_value=False)
    def test_default_file_not_found(self, mock_isfile, mock_input):
        target_dir = 'data/uploaded_excel_files'
        default_file_path = 'data/non_existent_file.xlsx'
        
        result = upload_file_to_application_directory(target_dir, default_file_path)
        self.assertIsNone(result)

    @patch('builtins.input', return_value='invalid_file.txt')
    @patch('os.path.isfile', return_value=False)
    def test_invalid_custom_file(self, mock_isfile, mock_input):
        target_dir = 'data/uploaded_excel_files'
        default_file_path = 'data/1038-0610-0614-day-larger-figures-test.xlsx'
        
        result = upload_file_to_application_directory(target_dir, default_file_path)
        self.assertIsNone(result)

    @patch('builtins.input', return_value='custom_file.xlsx')
    @patch('os.path.isfile', side_effect=lambda x: x == 'custom_file.xlsx')
    @patch('os.makedirs', side_effect=OSError("Permission denied"))
    def test_directory_creation_failure(self, mock_makedirs, mock_isfile, mock_input):
        target_dir = 'data/uploaded_excel_files'
        default_file_path = 'data/1038-0610-0614-day-larger-figures-test.xlsx'
        
        result = upload_file_to_application_directory(target_dir, default_file_path)
        self.assertIsNone(result)

    @patch('builtins.input', return_value='custom_file.xlsx')
    @patch('os.path.isfile', side_effect=lambda x: x == 'custom_file.xlsx')
    @patch('os.makedirs')
    @patch('shutil.copy', side_effect=Exception("Copy failed"))
    def test_file_copy_failure(self, mock_copy, mock_makedirs, mock_isfile, mock_input):
        target_dir = 'data/uploaded_excel_files'
        default_file_path = 'data/1038-0610-0614-day-larger-figures-test.xlsx'
        
        result = upload_file_to_application_directory(target_dir, default_file_path)
        self.assertIsNone(result)

    @patch('builtins.input', return_value='custom_file.xlsx')
    @patch('os.path.isfile', side_effect=lambda x: x == 'custom_file.xlsx')
    @patch('os.makedirs')
    @patch('shutil.copy')
    @patch('os.access', return_value=False)
    def test_read_access_denied(self, mock_access, mock_copy, mock_makedirs, mock_isfile, mock_input):
        target_dir = 'data/uploaded_excel_files'
        default_file_path = 'data/1038-0610-0614-day-larger-figures-test.xlsx'
        
        result = upload_file_to_application_directory(target_dir, default_file_path)
        self.assertIsNone(result)


class TestProcessCSV(unittest.TestCase):
    def setUp(self):
        # Path to your actual data file
        self.original_csv_filepath = r'data\uploaded_file.csv'
        self.modified_csv_filepath = 'test_modified.csv'

    def tearDown(self):
        # Remove the modified CSV file after testing
        if os.path.exists(self.modified_csv_filepath):
            os.remove(self.modified_csv_filepath)

    def test_file_creation_and_meta_data(self):
        modified_csv_filepath, duration, start_date, end_date, templates, analysis_window = process_csv(
            self.original_csv_filepath, self.modified_csv_filepath
        )
        
        # Assert that the modified file was created
        self.assertTrue(os.path.exists(modified_csv_filepath))
        df = pd.read_csv(self.modified_csv_filepath)
        for index, row in df.iterrows():
            template = templates[index]
            # Basic type assertions
            assert template['duration'] == int(duration)
            assert template['timestamp'] == start_date   
        self.assertIsInstance(start_date, str)
        self.assertIsInstance(end_date, str)
        self.assertIsInstance(templates, list)
        self.assertIsInstance(analysis_window, str)

    def test_dataframe_structure(self):
        _, _, _, _, _, _ = process_csv(self.original_csv_filepath, self.modified_csv_filepath)
        df = pd.read_csv(self.modified_csv_filepath)
        
        expected_columns = ['Host Name', 'cores', 'CPU_average', 'GPU_average', 'Total_MB_Sent', 
                            'Total_MB_Received', 'CPU_memory_average', 'GPU_memory_average', 
                            'machine-family', 'max-cpu-wattage', 'max-gpu-wattage', 
                            'cpu/thermal-design-power', 'device/emissions-embodied', 
                            'time-reserved', 'grid/carbon-intensity', 'device/expected-lifespan', 
                            'network-intensity', 'memory/thermal-design-power']
        for col in expected_columns:
            self.assertIn(col, df.columns)
    
    def test_machine_family_assignment(self):
        _, _, _, _, _, _ = process_csv(self.original_csv_filepath, self.modified_csv_filepath)
        df = pd.read_csv(self.modified_csv_filepath)
        
        self.assertTrue(all(df[df['cores'] == 24]['machine-family'] == 'z2 mini'))
        self.assertTrue(all(df[df['cores'] == 28]['machine-family'] == 'Z4R G4'))

    def test_template_structure(self):
        _, _, start_date, _, templates, _ = process_csv(self.original_csv_filepath, self.modified_csv_filepath)
        df = pd.read_csv(self.modified_csv_filepath)
        
        self.assertEqual(len(templates), len(df))
        for template in templates:
            self.assertIsInstance(template, dict)
            self.assertEqual(len(template), 22)  # Number of keys in process_row output
        
        # Check specific values for the first template
        first_template = templates[0]
        self.assertEqual(first_template['timestamp'], start_date)
        self.assertIn(first_template['instance-type'], ['z2 mini', 'Z4R G4'])
    
    def test_values(self):
        modified_csv_filepath, duration, start_date, end_date, templates, analysis_window = process_csv(
          self.original_csv_filepath, self.modified_csv_filepath
          ) 
 
        df = pd.read_csv('test_modified.csv')
        
        for index, row in df.iterrows():
            template = templates[index]
            
            assert row['CPU_average'] == pytest.approx(template['cpu/utilization']), \
                f"Mismatch in CPU utilization for row {index}"
            
            assert row['CPU_memory_average'] == pytest.approx(template['cpu-memory/utilization']), \
                f"Mismatch in CPU memory utilization for row {index}"
            
            assert row['GPU_memory_average'] == pytest.approx(template['gpu-memory/utilization']), \
                f"Mismatch in GPU memory utilization for row {index}"
            
            assert row['GPU_average'] == pytest.approx(template['gpu/utilization']), \
                f"Mismatch in GPU utilization for row {index}"

class TestExtractDataFromYAML(unittest.TestCase):
    def setUp(self):
        # Sample YAML data for testing
        self.yaml_data = {
            'tree': {
                'children': {
                    'child': {
                        'children': {
                            'machine1': {
                                'outputs': [
                                    {
                                        'timestamp': '2023-06-15T12:00:00Z',
                                        'instance-type': 'type1',
                                        'sci': 0.123456,
                                        'carbon-embodied': 1.23456,
                                        'carbon-operational': 2.34567,
                                        'duration': 3600,
                                        'carbon': 3.45678
                                    }
                                ]
                            },
                            'machine2': {
                                'outputs': [
                                    {
                                        'timestamp': '2023-06-16T12:00:00Z',
                                        'instance-type': 'type2',
                                        'sci': 0.234567,
                                        'carbon-embodied': 2.34567,
                                        'carbon-operational': 3.45678,
                                        'duration': 7200,
                                        'carbon': 4.56789
                                    }
                                ]
                            }
                        }
                    }
                }
            }
        }

    def test_extract_data_from_yaml(self):
        machine_emissions_list, machine_id_dict = extract_data_from_yaml(self.yaml_data)

        # Test the length of the output list
        self.assertEqual(len(machine_emissions_list), 2)

        # Test the content of the first machine emission
        first_emission = machine_emissions_list[0]
        self.assertEqual(first_emission['timestamp'], '2023-06-15')
        self.assertEqual(first_emission['machine-family'], 'type1')
        self.assertEqual(first_emission['sci'], 0.12)
        self.assertEqual(first_emission['carbon-embodied'], 1.23)
        self.assertEqual(first_emission['carbon-operational'], 2.35)
        self.assertEqual(first_emission['duration'], 3600)
        self.assertEqual(first_emission['carbon'], 3.46)

        # Test the machine ID dictionary
        self.assertEqual(len(machine_id_dict), 2)
        self.assertIn('machine1', machine_id_dict)
        self.assertIn('machine2', machine_id_dict)

        # Test that the machine IDs are strings
        self.assertIsInstance(machine_id_dict['machine1'], str)
        self.assertIsInstance(machine_id_dict['machine2'], str)

    def test_missing_required_fields(self):
        yaml_data = {
            'tree': {
                'children': {
                    'child': {
                        'children': {
                            'machine1': {
                                'outputs': [
                                    {
                                        'timestamp': '2023-06-15T12:00:00Z',
                                        # 'instance-type' is missing
                                        'sci': 0.123456,
                                        'carbon-embodied': 1.23456,
                                        'carbon-operational': 2.34567,
                                        'duration': 3600,
                                        'carbon': 3.45678
                                    }
                                ]
                            }
                        }
                    }
                }
            }
        }
        
        with self.assertRaises(KeyError):
            extract_data_from_yaml(yaml_data)

    def test_unexpected_structure(self):
        yaml_data = {
            'unexpected': {
                'structure': {
                    'here': ' '
                }
            }
        }
        
        with self.assertRaises(KeyError):
            extract_data_from_yaml(yaml_data)

    def test_large_number_of_machines(self):
        large_yaml_data = {
            'tree': {
                'children': {
                    'child': {
                        'children': {}
                    }
                }
            }
        }
        
        # Create 1000 machines
        for i in range(1000):
            large_yaml_data['tree']['children']['child']['children'][f'machine{i}'] = {
                'outputs': [
                    {
                        'timestamp': '2023-06-15T12:00:00Z',
                        'instance-type': 'type1',
                        'sci': 0.123456,
                        'carbon-embodied': 1.23456,
                        'carbon-operational': 2.34567,
                        'duration': 3600,
                        'carbon': 3.45678
                    }
                ]
            }
        
        machine_emissions_list, machine_id_dict = extract_data_from_yaml(large_yaml_data)
        
        self.assertEqual(len(machine_emissions_list), 1000)
        self.assertEqual(len(machine_id_dict), 1000)
        for i in range(1000):
            self.assertIn(f'machine{i}', machine_id_dict)

if __name__ == '__main__':
    unittest.main()




    