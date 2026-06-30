# Your First Container

Now that Docker is installed, let's run a real application in a container.

## Pulling an Image

Before you can run a container, you need an image. Images are stored in registries. The default registry is **Docker Hub**.

```bash
docker pull nginx
```

This downloads the official Nginx image. You can specify a tag for a specific version:

```bash
docker pull nginx:1.27
docker pull nginx:alpine
```

If no tag is specified, `:latest` is used by default.

## Running a Container

```bash
docker run -d -t --name my-nginx nginx
```

| Flag | Meaning |
|------|---------|
| `-d` | Detached mode — run in the background |
| `-t` | Allocate a pseudo-TTY |
| `--name my-nginx` | Give the container a friendly name |
| `nginx` | The image to use |

Without `--name`, Docker assigns a random name like `boring_morse`.

## Viewing Running Containers

```bash
docker ps
```

Example output:
```
CONTAINER ID   IMAGE   COMMAND                  CREATED         STATUS         PORTS     NAMES
a1b2c3d4e5f6   nginx   "/docker-entrypoint.…"   2 minutes ago   Up 2 minutes   80/tcp    my-nginx
```

## Connecting to a Running Container

```bash
docker exec -it my-nginx bash
```

This opens an interactive shell inside the container. You're now in the container's filesystem — it looks like a minimal Linux machine.

- `-i` — interactive (keep STDIN open)
- `-t` — allocate a pseudo-TTY

Exit the shell with `exit` or `Ctrl+D`.

## Viewing Logs

```bash
docker logs my-nginx
```

Add `-f` to follow the log stream in real time, like `tail -f`:

```bash
docker logs -f my-nginx
```

## Stopping a Container

```bash
docker stop my-nginx
```

This sends a SIGTERM signal to the main process, giving it time to shut down gracefully. After 10 seconds, Docker sends SIGKILL.

## Starting a Stopped Container

```bash
docker start my-nginx
```

## Removing a Container

```bash
docker rm my-nginx
```

To remove a running container, add `-f` (force):

```bash
docker rm -f my-nginx
```

## One-Liner: Run and Remove

For quick tests, use `--rm` to automatically delete the container when it exits:

```bash
docker run --rm -it nginx bash
```

Now that you can run basic containers, let's learn how to manage them effectively.
