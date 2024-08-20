# HP_Sus_Sys_3

##  Table of contents

- Description
- Installation
- Configuration
- Troubleshooting
- FAQ

## Description
This aim of this project was to build an integrated system which converts telemetry data into carbon emissions estimations. 

The system is compatible with 2 types of HP remote workstations (Z2 Mini and ZCentral 4R ENERGY STAR ). The Excel file uploaded by the user must only include elemetry data collected from these machines.

The system utilises the Green Software Foundation's Impact Framework to support its generation of carbon emissions therefore the user must have already installed this framework on their machine in order for the system to work (See Installation).

The system uses Ollama server to support the use of the offline Llama3 8B Model to run the queries.



## Installation
### Impact Framework
Install the Impact Framework globally using npm.

npm install -g @grnsft/if

Install the plugins to support the running of the pipeline.

npm install -g @grnsft/if-plugins
npm install -g @grnsft/if-unofficial-plugins

The system will automatically generate and run customised manifest files per user upload. 

### Ollama

The application comes packaged with the Ollama.exe file in the main app directory. It also includes the Llama3-8B, Llam3-Chat-QA and Mistral-7B models in the same directory (used for testing and evaluation). The system will set the Ollama Models Environment variable to this location. Please refer to Ollama for more help regarding specific installation queries: https://github.com/ollama/ollama/blob/main/docs/faq.md#how-do-i-configure-ollama-server

## Configuration

Please run the following command to to install dependencies:

pip install r-requirements.txt 

This application is compatible Windows 11 system, using NVIDIA GPU with Cuda. Please ensure you have configured your machine with Cuda: https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html

Due to GPU memory requirements, this application has not be tested on another OS. 
