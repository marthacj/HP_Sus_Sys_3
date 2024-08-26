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
This aim of this project was to build an integrated system which converts telemetry data into carbon emissions estimations. 

The system is compatible with 2 types of HP remote workstations (Z2 Mini and ZCentral 4R ENERGY STAR ). The Excel file uploaded by the user must only include elemetry data collected from these machines.

The system utilises the Green Software Foundation's Impact Framework to support its generation of carbon emissions therefore the user must have already installed this framework on their machine in order for the system to work (See Installation).

The system uses Ollama server to support the use of the offline Llama3 8B Model to run the queries.

## Prerequisites
- Windows 10 or later
- Python 3.10
- NVIDIA GPU with CUDA support (see https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#:~:text=2.2.-,Download%20the%20NVIDIA%20CUDA%20Toolkit,downloads%20packages%20required%20for%20installation)
- Please check here to see whether your GPU is compatible with Ollama:  https://github.com/ollama/ollama/blob/main/docs/gpu.md
- Node.js (**v20.14.0**) and npm (v**10.8.1**)


## Installation
1. Clone or download this repository.


2. Install CUDA Toolkit:
   - Download and install the CUDA Toolkit from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads).


3. Install Python dependencies: 
   pip install -r requirements.txt 


4. Install the Impact Framework globally using npm (**you may need to use 'sudo in front of each command'**).

**IMPORTANT**: Check what the latest version of IF is using **npm view @grnsft/if versions**. At the time of development, the latest was 0.6.0 - the syntax of this project is not compatible with earlier versions and may not be with later versions.

**npm install -g @grnsft/if@0.6.0 **

OR if still the latest version:

**npm install -g @grnsft/if** 


Install the plugins to support the running of the pipeline.

**npm install -g @grnsft/if-plugins
npm install -g @grnsft/if-unofficial-plugins**


5. Next the llama3 model from Ollama is needed. If you are on Windows OS, the executable is already included in this directory. Skip to 5.b

5.a If you are on MacOS, please go here https://github.com/ollama/ollama/blob/main/docs/faq.md#how-do-i-configure-ollama-server to download the ollama.bin model and move it into the app folder in this directory.
IMPORTANT: When moving the application's directory, it may lose its executable status. You will need to cd into the app directory, and run chmod +x ollama. The application will check this and do so if it has not been done manually, but it is better if done by the user.
 
Next, run the command:

   **.(/app)/ollama pull llama3**

5.b On Windows, run:

  ** .\app\ollama.exe pull llama3**


6. For size limitations, all users will need to pull the latest llama3 model themselves. The system will set the Ollama Models Environment variable to location: /app/models.
   On Windows: Once you have pulled the llama3 model, you must move the blobs and manifest files from their original location on your machine to the above specified folder.
   On Mac: Once you have pulled the model, you do not need to move the files.


## Usage

1. Open a command prompt and cd into the **application directory**.

2. Run the application: **/app/main.py**


## Troubleshooting

This application is compatible Windows 10 system, using NVIDIA RTX A2000 GPU with Cuda. Please ensure you have configured your machine with Cuda: https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html

If you encounter CUDA-related errors, ensure your NVIDIA drivers are up to date. This machine uses CUDA 11.8.0

If you have issues with imports (e.g. with sentence-transformer, numpy, or pandas in particular), ensure you downgrade python to Python 3.10.

Due to GPU memory requirements, this application has not be tested on another OS, but did run successfully on Docker using Linux distribution with Ubuntu22.04. 
