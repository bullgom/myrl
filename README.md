# What Is This Repo?
I do some rl experiments here. Nothing suspicious.

# Requirements
- Docker
- CUDA 11.2

# Docker Image Build
```cmd
docker build -t myrl:latest .
```

# Running Docker Container
- Start docker container with: 
```cmd
docker run --gpus all -it --rm --network host myrl
```
The `--network host` argument is not needed, but needs double check. Too lazy to check

# Running Scripts
- TD3
```cmd
python -m scripts.td3
```