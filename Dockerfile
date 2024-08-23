# Use an official CUDA base image with Ubuntu
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install curl and other necessary dependencies
RUN apt-get update && apt-get install -y \
    lsb-release \
    curl \
    wget \
    bash \
    build-essential

RUN curl -fsSL https://nodejs.org/dist/v20.14.0/node-v20.14.0-linux-x64.tar.xz -o node-v20.14.0-linux-x64.tar.xz \
&& tar -xJf node-v20.14.0-linux-x64.tar.xz -C /usr/local --strip-components=1 \
&& rm node-v20.14.0-linux-x64.tar.xz

# Verify Node.js and npm versions
RUN node -v \
    && npm -v

# Install the specific npm version
RUN npm install -g npm@10.8.1
# Install npm packages globally
RUN npm install -g @grnsft/if @grnsft/if-plugins @grnsft/if-unofficial-plugins

# Install Python 3.10 and pip
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    dos2unix \
    sed \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# RUN curl -fsSL https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add - \
# && curl -fsSL https://nvidia.github.io/nvidia-docker/ubuntu18.04/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list \
# && apt-get update && apt-get install -y nvidia-container-toolkit

# Set the working directory
WORKDIR /app

# Upgrade pip to the latest version
RUN python3.10 -m pip install --upgrade pip

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN python3.10 -m pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .
# Convert Windows line endings to Unix for shell scripts
RUN sed -i 's/\r$//' /app/app/start_server.sh

# Ensure Ollama is executable
RUN chmod +x /app/app/ollama
RUN chmod +x /app/app/start_server.sh

# RUN /app/app/ollama serve
# RUN /app/app/ollama pull llama3

# Expose the necessary port
EXPOSE 11434

# Default command to run the application
CMD ["python3.10", "app/main.py"]



# # Use an official CUDA base image with Ubuntu
# FROM  nvidia/cuda:11.8.0-runtime-ubuntu22.04

# ENV DEBIAN_FRONTEND=noninteractive
# # Install Python 3.12 and pip
# RUN apt-get update && apt-get install -y \
#     software-properties-common \
#     && add-apt-repository ppa:deadsnakes/ppa \
#     && apt-get update && apt-get install -y \
#     python3.12 \
#     python3.12-distutils \
#     python3.12-venv \
#     && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
#     && rm -rf /var/lib/apt/lists/*

# # Set the working directory in the container
# WORKDIR /app

# # Upgrade pip to the latest version
# RUN python3 -m pip install --upgrade pip
# RUN which python3
# RUN which pip3

# # Copy the requirements.txt into the container
# COPY requirements.txt .

# # Install Python dependencies
# RUN pip3 install --no-cache-dir -r requirements.txt

# # Copy the entire project directory into the container
# COPY . .

# # Ensure Ollama is executable
# RUN chmod +x /app/app/ollama.exe

# EXPOSE 11434

# # Set the entrypoint or command to run your application
# CMD ["python3", "/app/app/main.py"]


# # Use an official CUDA base image with Ubuntu
# FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# ENV DEBIAN_FRONTEND=noninteractive

# # Install Python 3.12 and pip
# RUN apt-get update && apt-get install -y \
#     software-properties-common \
#     && add-apt-repository ppa:deadsnakes/ppa \
#     && apt-get update && apt-get install -y \
#     python3.12 \
#     python3.12-distutils \
#     python3.12-venv \
#     curl \
#     && rm -rf /var/lib/apt/lists/*

# # Install pip for Python 3.12
# RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# # Set the working directory in the container
# WORKDIR /app

# # Ensure pip3 is available and upgrade pip to the latest version
# RUN /usr/bin/python3.12 -m pip install --upgrade pip

# # Copy the requirements.txt into the container
# COPY requirements.txt .

# # Install Python dependencies
# # Upgrade pip and install Python dependencies
# RUN python3.12 -m pip install --upgrade pip \
#     && python3.12 -m pip install --no-cache-dir -r requirements.txt


# # Copy the entire project directory into the container
# COPY . .

# # Ensure Ollama is executable
# RUN chmod +x /app/app/ollama.exe

# EXPOSE 11434
