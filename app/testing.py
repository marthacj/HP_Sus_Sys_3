import unittest
from unittest.mock import patch, mock_open
import os
from CSV_upload import upload_file_to_application_directory

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
    @patch('builtins.input', return_value='')
    @patch('os.path.isfile', return_value=True)
    @patch('os.makedirs')
    @patch('shutil.copy')
    @patch('os.access', return_value=True)
    def test_upload_default_file(self, mock_access, mock_copy, mock_makedirs, mock_isfile, mock_input):
        target_dir = 'data/uploaded_excel_files'
        default_file_path = 'data/1038-0610-0614-day-larger-figures-test.xlsx'
        
        result = upload_file_to_application_directory(target_dir, default_file_path)
        self.assertEqual(result, default_file_path)
        mock_copy.assert_not_called()
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

if __name__ == '__main__':
    unittest.main()