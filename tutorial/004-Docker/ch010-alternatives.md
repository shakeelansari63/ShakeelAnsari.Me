# Alternative Container Technologies

While Docker is the most popular container platform, several alternatives offer different trade-offs in security, architecture, and workflow.

## Podman

Podman (Pod Manager) is a daemonless container engine developed by Red Hat. It's designed as a drop-in replacement for Docker.

### Key Differences from Docker

| Aspect | Docker | Podman |
|--------|--------|--------|
| Architecture | Client + daemon (dockerd) | Daemonless — forks directly |
| Rootless | Requires configuration | Native rootless by default |
| Pods | Manual networking | Native pod support (like Kubernetes) |
| Systemd integration | Manual | Built-in with `podman generate systemd` |
| Docker Compose | Native support | Supports via `podman-compose` |

### Installation

```bash
# Fedora / RHEL
sudo dnf install podman

# Ubuntu / Debian
sudo apt install podman

# Arch Linux
sudo pacman -S podman
```

### Basic Usage

Podman commands are identical to Docker's — you can alias it:

```bash
alias docker=podman
```

Or use the commands directly:

```bash
podman pull docker.io/library/nginx
podman run -d -p 8080:80 --name web docker.io/library/nginx
podman ps
podman exec -it web sh
podman stop web
podman rm web
```

### Rootless Containers

Podman runs rootless out of the box — no `sudo`, no `docker` group:

```bash
podman run -d -p 8080:80 docker.io/library/nginx
```

The container runs as your user, with ranges of subordinate UIDs/GIDs mapped from `/etc/subuid` and `/etc/subgid`. This means even if a process escapes the container, it has no privileges on the host.

### Pods

Podman introduces **pods** — groups of containers that share network namespace, similar to Kubernetes pods:

```bash
podman pod create --name my-pod -p 8080:80

podman run -d --pod my-pod --name web docker.io/library/nginx
podman run -d --pod my-pod --name api docker.io/library/my-api
```

Both containers share the same IP and ports, and can communicate via `localhost`.

### Generating Systemd Units

Podman can generate systemd service files for containers:

```bash
podman generate systemd --name web --files --new
```

This creates `container-web.service` that you can install and manage with `systemctl`.

### Podman Compose

```bash
pip install podman-compose
podman-compose up -d
```

Supports most Docker Compose features with a compatible YAML format.

## LXC / LXD

LXC (Linux Containers) is an OS-level virtualization system that runs multiple isolated Linux systems (containers) on a single host. Unlike Docker's application containers, LXC provides **system containers** that behave like lightweight VMs.

### Containers vs System Containers

| Feature | Docker (Application Container) | LXC (System Container) |
|---------|-------------------------------|------------------------|
| Scope | Single process / app | Full OS with init system |
| Init system | None (unless specified) | systemd / openrc |
| SSH | Not included | Can run SSH daemon |
| Boot time | Milliseconds | 2-5 seconds |
| Image size | 5-200 MB | 100-2000 MB |
| Use case | Microservices | Full OS environments |

### LXD

LXD is the next-generation system container manager built on top of LXC. It provides a Docker-like experience for system containers.

```bash
# Install LXD
sudo snap install lxd

# Initialize
lxd init

# Launch a container
lxc launch ubuntu:24.04 my-container

# Execute commands
lxc exec my-container -- apt update
lxc exec my-container -- bash

# List containers
lxc list

# Stop and delete
lxc stop my-container
lxc delete my-container
```

### Key LXD Features

- **Live migration** — move running containers between hosts
- **Snapshots** — instant point-in-time backups
- **Resource limits** — CPU, memory, disk quotas per container
- **Nesting** — run Docker inside LXD containers
- **Profiles** — reusable configuration templates

### Running Docker Inside LXD

```bash
lxc launch ubuntu:24.04 docker-host
lxc exec docker-host -- apt install docker.io
lxc exec docker-host -- docker run hello-world
```

Useful for testing Docker configurations in isolated environments.

## containerd

containerd is the industry-standard core container runtime. It's the runtime that Docker itself uses under the hood.

```bash
# Install (Ubuntu)
sudo apt install containerd

# Use with ctr CLI
ctr images pull docker.io/library/nginx:alpine
ctr run docker.io/library/nginx:alpine web
```

containerd is used by Docker, Kubernetes (via CRI), and many other platforms. It implements the OCI (Open Container Initiative) runtime specification.

## Buildah

Buildah is a tool for building OCI-compliant images without requiring a container daemon. It pairs well with Podman.

```bash
# Build from a Dockerfile
buildah bud -t my-app .

# Build from scratch (minimal image)
buildah from scratch
buildah add working-container /app
buildah config --entrypoint '["/app/server"]' working-container
buildah commit working-container my-app:latest
```

### Advantages

- **Daemonless builds** — no Docker daemon needed
- **Fine-grained control** — build images layer by layer
- **Rootless** — works without privileges
- **Multi-architecture** — build for different platforms

## Skopeo

Skopeo is a tool for working with remote container registries. It can inspect, copy, and delete images without pulling them.

```bash
# Inspect an image without pulling
skopeo inspect docker://docker.io/library/nginx:alpine

# Copy between registries
skopeo copy docker://docker.io/library/nginx:alpine docker://myregistry.local/nginx:alpine

# Delete from registry
skopeo delete docker://myregistry.local/nginx:alpine
```

## Comparison Summary

| Tool | Focus | Daemon | Rootless | Pods | OCI Compliant |
|------|-------|--------|----------|------|---------------|
| Docker | General container platform | Yes | Limited | No | Yes |
| Podman | Daemonless alternative | No | Native | Yes | Yes |
| LXC/LXD | System containers | Yes | Limited | No | No |
| containerd | Low-level runtime | Yes | Partial | No | Yes |
| Buildah | Image building | No | Yes | — | Yes |
| Skopeo | Registry operations | No | Yes | — | — |

## When to Use What

- **Docker** — best for development, CI/CD, and when you need the largest ecosystem
- **Podman** — production environments requiring rootless security, or Kubernetes-native workflows with pods
- **LXC/LXD** — when you need full OS containers that behave like VMs (testing, CI runners, legacy apps)
- **containerd** — when you're building a custom platform and need a minimal, OCI-compliant runtime
- **Buildah + Podman** — secure CI/CD pipelines where the Docker daemon is a security concern
- **Skopeo** — registry management and image mirroring in air-gapped environments

The container ecosystem is rich and evolving. Docker remains the most accessible starting point, but understanding these alternatives helps you choose the right tool for each workload.
