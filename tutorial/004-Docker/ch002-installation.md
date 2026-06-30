# Installing Docker

Docker is available on all major operating systems. Choose your platform below.

## Linux

### Ubuntu / Debian
```bash
sudo apt update && sudo apt install docker.io
sudo systemctl enable --now docker
```

### Fedora
```bash
sudo dnf install docker
sudo systemctl enable --now docker
```

### Arch Linux
```bash
sudo pacman -S docker
sudo systemctl enable --now docker
```

### Post-Install (Linux)
Add your user to the `docker` group to run Docker without `sudo`:
```bash
sudo usermod -aG docker $USER
```
Log out and back in for the group change to take effect.

## macOS

Download **Docker Desktop for Mac** from [docker.com](https://www.docker.com/products/docker-desktop) and install the `.dmg`. It includes the Docker CLI, Docker Compose, and a GUI dashboard.

## Windows

Download **Docker Desktop for Windows** from [docker.com](https://www.docker.com/products/docker-desktop). WSL 2 backend is recommended for better performance.

## Verify Installation

```bash
docker --version
```

You should see something like:
```
Docker version 27.0.3, build 7d4bcd8
```

## Test That It Works

```bash
docker run hello-world
```

This downloads a tiny test image and runs a container that prints a welcome message and exits. If you see the message, Docker is working correctly.

## Common Issues

| Symptom | Fix |
|---------|-----|
| `permission denied` | User not in `docker` group — run `sudo usermod -aG docker $USER` and relog |
| `Cannot connect to the Docker daemon` | Docker daemon not running — `sudo systemctl start docker` |
| `no space left on device` | Clean up unused images/containers — `docker system prune` |

Next up: running your first real container.
