FROM continuumio/miniconda3:latest
ENV PATH="/root/miniconda3/bin:$PATH"  
 
# Suppress Conda warnings (e.g., "conda init" prompts)  
ENV PYTHONDONTWRITEBYTECODE=1  
ENV PYTHONUNBUFFERED=1

RUN apt-get update 

RUN apt-get install -y --no-install-recommends sudo \  
    build-essential \  
    git \  
    ffmpeg libsm6 libxext6 \
    -qqy x11-apps \
    && rm -rf /var/lib/apt/lists/*

RUN export uid=1000 gid=1000
RUN mkdir -p /home/docker_user
RUN echo "docker_user:x:${uid}:${gid}:docker_user,,,:/home/docker_user:/bin/bash" >> /etc/passwd
RUN echo "docker_user:x:${uid}:" >> /etc/group
RUN echo "docker_user ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/docker_user
RUN chmod 0440 /etc/sudoers.d/docker_user
RUN chown ${uid}:${gid} -R /home/docker_user 

RUN conda create -n myenv python=3.10 -y
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

COPY requirements.txt ./requirements.txt

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .
RUN conda clean -afy

USER    docker_user 
ENV     HOME=/home/docker_user 
CMD     xeyes

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "myenv", "python", "runner.py"]