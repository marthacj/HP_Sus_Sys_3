import unittest
from unittest.mock import patch
import os
from CSV_upload import upload_file_to_application_directory
from manifest_generation import process_csv
import pandas as pd
from tempfile import NamedTemporaryFile

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
        self.original_csv_filepath = r'data\1038-0610-0614-evening.csv'
        self.modified_csv_filepath = 'test_modified.csv'

    def tearDown(self):
        # Remove the modified CSV file after testing
        if os.path.exists(self.modified_csv_filepath):
            os.remove(self.modified_csv_filepath)

    def test_process_csv(self):
        # Call the function with the actual data file
        modified_csv_filepath, duration, start_date, end_date, templates, analysis_window = process_csv(
            self.original_csv_filepath, self.modified_csv_filepath
        )

        # Assert that the modified file was created
        self.assertTrue(os.path.exists(modified_csv_filepath))

        # Read the modified CSV file
        df = pd.read_csv(modified_csv_filepath)

        # Basic assertions
        self.assertIsInstance(duration, int)
        self.assertIsInstance(start_date, str)
        self.assertIsInstance(end_date, str)
        self.assertIsInstance(templates, list)
        self.assertIsInstance(analysis_window, str)

        # Check if all expected columns are present
        expected_columns = ['Host Name', 'cores', 'CPU_average', 'GPU_average', 'Total_MB_Sent', 
                            'Total_MB_Received', 'CPU_memory_average', 'GPU_memory_average', 
                            'machine-family', 'max-cpu-wattage', 'max-gpu-wattage', 
                            'cpu/thermal-design-power', 'device/emissions-embodied', 
                            'time-reserved', 'grid/carbon-intensity', 'device/expected-lifespan', 
                            'network-intensity', 'memory/thermal-design-power']
        for col in expected_columns:
            self.assertIn(col, df.columns)

        # Check if machine-family is correctly assigned
        self.assertTrue(all(df[df['cores'] == 24]['machine-family'] == 'z2 mini'))
        self.assertTrue(all(df[df['cores'] == 28]['machine-family'] == 'Z4R G4'))

        # # Check if GPU_average is correctly adjusted
        # self.assertTrue(all(df['GPU_average'] >= 0.1))
 

        # Check templates
        self.assertEqual(len(templates), len(df))
        for template in templates:
            self.assertIsInstance(template, dict)
            self.assertEqual(len(template), 22)  # Number of keys in process_row output

        # Check specific values for the first template
        first_template = templates[0]
        self.assertEqual(first_template['timestamp'], start_date)
        self.assertEqual(first_template['duration'], duration)
        self.assertIn(first_template['instance-type'], ['z2 mini', 'Z4R G4'])



if __name__ == '__main__':
    unittest.main()




    