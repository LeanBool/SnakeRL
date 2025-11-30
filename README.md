# snake-rl

## A containerized snake reinforcement learning environment + agent for playing snake 

<p align="center">
  <img src="https://raw.githubusercontent.com/LeanBool/SnakeRL/refs/heads/main/img/playing_game.gif" alt="Bot playing game on 10x9 grid">  
  <em>Bot playing game on 10x9 grid</em>
</p>
<p align="center">
  <img src="https://raw.githubusercontent.com/LeanBool/SnakeRL/refs/heads/main/img/tensorboard.png" alt="Tensorboard chart of training 2 million steps on a 5x5 grid">  
  <em>Tensorboard chart of training 2 million steps on a 5x5 grid</em>
</p>

### How to use:
- Install Docker
- Install the NVIDIA Container Toolkit (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- Clone this repository
- Edit runner.py to specify parameters like grid size, training steps, ...
- Run ./run.sh
- If you have issues with the testing window not opening after training is done, check if the uid and gid in Dockerfile on line 17 match the xhost's respective ids.

Note: Tensorboard can be accessed on localhost:8080