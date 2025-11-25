# snake-rl

## A containerized snake reinforcement learning environment + agent for playing snake 

<p align="center">
<<<<<<< HEAD
  <img src="https://raw.githubusercontent.com/LeanBool/SnakeRL/refs/heads/main/img/playing_game.gif" alt="Bot playing game on 6x4 grid">  
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/LeanBool/SnakeRL/refs/heads/main/img/tensorboard.png" alt="tensorboard chart after 4m steps on 6x4">  
=======
  <img src="[https://github.com/user-attachments/assets/0a734ed3-f441-4446-85d2-6e8f67085026](https://github.com/user-attachments/assets/fbe204dc-78d1-4ea7-986d-35b0439829d0)" alt="Bot playing game">  
>>>>>>> 2c719856986bd41e27918b57b1dcc24a0e94c84a
</p>

When using Docker:
Requires the NVIDIA Container Tookit to use CUDA capabilities
(https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

On Debian:
To render GUI correctly, set the uid and gid (obtained from the id $Username command) inside the dockerfile (change line containing export uid=... gid=...)
and run "sudo xhost +local:docker" to allow connecting to the local x-server.

Untested on Windows.

Tensorboard can be accessed on localhost:8080.
