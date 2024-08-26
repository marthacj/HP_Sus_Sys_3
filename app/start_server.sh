#!/bin/bash

# Navigate to the directory where this script is located
cd "$(dirname "$0")"

# Echo the current directory and the command being run
echo "Current directory: $(pwd)"
echo "Running command: Ollama serve"

echo "Ollama Model path set to: $OLLAMA_MODELS"

chmod +x Ollama 
# Start the ollama server
Ollama serve
