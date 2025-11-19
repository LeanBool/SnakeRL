FROM continuumio/miniconda3:latest
ENV PATH="/root/miniconda3/bin:$PATH"  
 
# Suppress Conda warnings (e.g., "conda init" prompts)  
ENV PYTHONDONTWRITEBYTECODE=1  
ENV PYTHONUNBUFFERED=1

RUN apt-get update 

RUN apt-get install -y --no-install-recommends \  
    build-essential \  
    git \  
    && rm -rf /var/lib/apt/lists/*

RUN conda create -n myenv python=3.10 -y
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

COPY requirements.txt ./requirements.txt

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .
RUN conda clean -afy

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "myenv", "python", "runner.py"]