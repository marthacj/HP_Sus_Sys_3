# CARBON ESTIMATOR SYSTEM

##  Table of contents

- Description
- Prerequisites
- Installation
- Usage
- Configuration
- Troubleshooting
- FAQ

## Description
This project aimed to build an integrated system which converts telemetry data into carbon emissions estimations which was part of a pipeline using a Large Language Model and a Retrieval Augmented Generation system to support natural language user input to query the data. 

The system is compatible with 2 types of HP remote workstations (Z2 Mini and ZCentral 4R ENERGY STAR ). The Excel file uploaded by the user must only include telemetry data collected from these machines.

The system utilises the Green Software Foundation's Impact Framework to support its generation of carbon emissions therefore the user must have already installed this framework on their machine in order for the system to work (See Installation).

The system uses Ollama server to support the use of the offline Llama3 8B Model to run the queries.

## Prerequisites
- Windows 10 or MacOS
- Python 3.10 
 **IMPORTANT**: Using a later Python version is possible but requires pip installing some packages separately to requirements.txt (see below)
- Node.js (**v20.14.0**) and npm (v**10.8.1**)

### For BEST results, please use with a GPU. The application can run on the CPU but is incredibly slow.
- NVIDIA GPU with CUDA support (see https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#:~:text=2.2.-,Download%20the%20NVIDIA%20CUDA%20Toolkit,downloads%20packages%20required%20for%20installation)
- Please check here to see whether your GPU is compatible with Ollama:  https://github.com/ollama/ollama/blob/main/docs/gpu.md

## Installation
1. Clone or download this repository.


2. Install CUDA Toolkit:
   - Download and install the CUDA Toolkit from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads).


3. Install Python dependencies: 
   pip install -r requirements.txt 
   Depending on your Python version, you may find the following packages need to be pip-installed separately:
   - sentence-transformer
   - pytest
   - ollama
   - openpyxl 

5. Install the Impact Framework globally using npm (**you may need to use 'sudo' in front of each command**).

   **IMPORTANT**: Check what the latest version of IF is using **npm view @grnsft/if versions**. At the time of development, the latest was 0.6.0 - the syntax of this project is not compatible with earlier versions and may not be compatible with later versions.
   
   **npm install -g @grnsft/if@0.6.0**
   
   OR if still the latest version:
   
   **npm install -g @grnsft/if** 
   
   
   Install the plugins to support the running of the pipeline.
   
   **npm install -g @grnsft/if-plugins
   npm install -g @grnsft/if-unofficial-plugins**


5. Next the llama3 model from Ollama is needed. If you are on Windows OS, the executable is already included in this directory. Skip to 6.
   5.a) If you are on MacOS, please go here https://github.com/ollama/ollama/blob/main/docs/faq.md#how-do-i-configure-ollama-server to download the Ollama application.
   IMPORTANT: When you download Ollama, the system may ask you if you want to move the application to Applications. Once there, right-click and select 'Show Package Contents > Contents > MacOS' then move the Ollama executable into    the HP_Sus_Sys/app folder.
 

6. For size limitations, all users will need to pull the latest llama3 model themselves. 
   6.a) On Windows, run :
     **.\app\ollama.exe pull llama3**
    Once you have pulled the llama3 model, you must move the blobs and manifest files from their original location on your machine to the app/models folder.
   
   6.b) On Mac, run:
   **Ollama pull llama3**
   Once you have pulled the model, you do not need to move the files.



## Usage

1. Open a command prompt and cd into the **application directory**.

2. Run the application: **python app/main.py** 



## Troubleshooting

This application is best compatible with Windows 10 or MacOS systems, using NVIDIA RTX A2000 GPU with Cuda. Whilst the application can run without GPU, its time for responses slows from **seconds** to **minutes**. Please ensure you have configured your machine with Cuda: https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html

If you encounter CUDA-related errors, ensure your NVIDIA drivers are up to date. This machine uses CUDA 11.8.0

If you have issues with imports (e.g. with sentence-transformers, numpy, or pandas in particular), you may be required to uninstall sentence-transformers, uninstall numpy, and then reinstall numpy with pip install numpy==1.26.4. Ensure you have all the dependencies required for sentence-transformers with pip show sentence-transformers. Use python3 app/main.py if "python app/main.py" not working. 

Due to GPU memory requirements, this application has not be tested on another OS, but did run successfully on Docker using Linux distribution with Ubuntu22.04. 
