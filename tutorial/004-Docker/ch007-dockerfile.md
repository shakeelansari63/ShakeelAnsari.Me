# Building Images with Dockerfile

A Dockerfile is a text file with instructions for building a Docker image. It automates the process of setting up an environment and application.

## Basic Dockerfile

```dockerfile
FROM node:20-alpine
WORKDIR /app
COPY package.json .
RUN npm install
COPY . .
EXPOSE 3000
CMD ["node", "index.js"]
```

Build it:

```bash
docker build -t my-app .
docker build -t my-app:1.0 .
```

## Common Instructions

| Instruction | Purpose |
|-------------|---------|
| `FROM` | Base image to start from |
| `WORKDIR` | Set working directory for subsequent instructions |
| `COPY` | Copy files from host to container |
| `ADD` | Like COPY but supports URLs and tar auto-extraction |
| `RUN` | Execute a command during build (creates a layer) |
| `EXPOSE` | Document which port the container listens on |
| `ENV` | Set environment variables |
| `ARG` | Build-time variables (not persisted in the image) |
| `CMD` | Default command when the container starts |
| `ENTRYPOINT` | Main command that runs — arguments get appended |

## CMD vs ENTRYPOINT

```dockerfile
# CMD can be overridden by providing arguments
CMD ["node", "index.js"]
# docker run my-app          -> node index.js
# docker run my-app node app.js  -> node app.js

# ENTRYPOINT always runs — arguments are appended
ENTRYPOINT ["node"]
CMD ["index.js"]
# docker run my-app          -> node index.js
# docker run my-app app.js   -> node app.js
```

## Multi-Stage Builds

Reduce final image size by separating build and runtime stages:

```dockerfile
# Stage 1: Build
FROM golang:1.22 AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -o server

# Stage 2: Runtime
FROM alpine:3.20
RUN apk add --no-cache ca-certificates
COPY --from=builder /app/server /server
EXPOSE 8080
CMD ["/server"]
```

The final image contains only the binary + Alpine (~15 MB) instead of the full Go toolchain (~1 GB).

## .dockerignore

Prevent unnecessary files from being sent to the Docker daemon:

```
node_modules
.git
.env
*.md
.gitignore
```

## Build Cache

Docker caches layers. If a layer hasn't changed, it reuses the cached version. Order instructions from least to most frequently changing:

```dockerfile
# Good: dependencies first
COPY package.json package-lock.json ./
RUN npm install

# Then application code (changes often)
COPY . .
```

## Practical Example: Python Flask App

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

```bash
docker build -t flask-app .
docker run -d -p 5000:5000 flask-app
```

Now you can build your own images for any application. Next: orchestrating multiple containers with Docker Compose.
