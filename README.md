# snake-rl

## A containerized snake reinforcement learning environment + agent for playing snake on X11 

Requires the NVIDIA Container Tookit to use CUDA capabilities
(https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

Requires Docker.

To render GUI correctly, set the uid and gid (obtained from the id $Username command) inside the dockerfile (change line containing export uid=... guid=...).

TODO:
- [x] Create Docker environment
- [x] Set up CUDA use inside container
- [x] Set up rendering inside container
- [x] Create gym environment
- [x] Create pygame rendering
- [ ] Agent, different network architectures, ...
- [ ] Add tensorboard writers and service