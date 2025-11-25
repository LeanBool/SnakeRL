# snake-rl

## A containerized snake reinforcement learning environment + agent for playing snake 

<p align="center">
  <img src="https://github.com/user-attachments/assets/0a734ed3-f441-4446-85d2-6e8f67085026" alt="Bot playing game">  
</p>

When using Docker:
Requires the NVIDIA Container Tookit to use CUDA capabilities
(https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

On Debian:
To render GUI correctly, set the uid and gid (obtained from the id $Username command) inside the dockerfile (change line containing export uid=... gid=...)
and run "sudo xhost +local:docker" to allow connecting to the local x-server.

Untested on Windows.

Tensorboard can be accessed on localhost:8080.