# What is Docker?

Docker is an open-source platform that automates the deployment, scaling, and management of applications inside lightweight, isolated environments called **containers**.

Unlike traditional virtual machines that each bundle a full operating system, containers share the host OS kernel while running in isolated user-space processes. This makes containers far more resource-efficient — they start in milliseconds and consume only what the application needs.

## Containers vs Virtual Machines

| Feature | Container | Virtual Machine |
|---------|-----------|----------------|
| Startup time | Milliseconds | Minutes |
| Size | Megabytes | Gigabytes |
| OS | Shares host kernel | Full OS per VM |
| Isolation | Process-level | Hardware-level |
| Resource usage | Lightweight | Heavy |

## Docker Architecture

Docker uses a client-server architecture:

- **Docker Daemon (dockerd)** — runs on the host, manages containers, images, networks, and storage volumes
- **Docker Client (docker)** — the CLI that talks to the daemon via a REST API
- **Docker Registries** — repositories for Docker images (Docker Hub is the default public registry)
- **Docker Objects** — images, containers, networks, volumes

## Why Use Docker?

- **Consistency** — works the same on your laptop, CI server, and production
- **Isolation** — each container has its own filesystem, networking, and process space
- **Portability** — run anywhere Docker is installed
- **Version Control** — images are layered and versioned
- **Ecosystem** — thousands of pre-built images on Docker Hub

## Key Concepts

| Term | Definition |
|------|------------|
| **Image** | A read-only template with instructions for creating a container |
| **Container** | A runnable instance of an image |
| **Dockerfile** | A text file with commands to build an image |
| **Volume** | Persistent storage outside the container's filesystem |
| **Network** | Connects containers to each other and the outside world |
| **Registry** | A repository for storing and distributing images |

In the next chapter, we'll get Docker installed and running on your system.
