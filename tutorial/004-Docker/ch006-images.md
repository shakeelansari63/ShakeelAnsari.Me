# Working with Images

Images are the blueprints for containers. Understanding how to find, download, inspect, and clean images is essential.

## Searching for Images

Search Docker Hub from the CLI:

```bash
docker search nginx
docker search --limit 10 --filter "stars=1000" postgres
```

## Pulling Images

```bash
docker pull nginx              # latest tag
docker pull nginx:1.27         # specific version
docker pull nginx:alpine       # Alpine Linux variant (smaller)
docker pull alpine             # minimal base image (~5 MB)
```

## Listing Images

```bash
docker images
docker image ls
```

Example output:
```
REPOSITORY   TAG       IMAGE ID       CREATED       SIZE
nginx        latest    6f715d38cfe0   2 weeks ago   187 MB
alpine       latest    b2aa39c304c2   4 weeks ago   7.05 MB
```

## Tagging Images

Tags let you version and organize images:

```bash
docker tag nginx:latest myrepo/nginx:1.27
docker tag nginx:latest myrepo/nginx:latest
```

## Removing Images

```bash
docker rmi nginx                    # remove by name:tag
docker rmi 6f715d38cfe0            # remove by image ID
docker rmi $(docker images -q)     # remove all images
```

Add `-f` to force removal if a container is using the image.

## Image Layers

Images are built in layers. Each instruction in a Dockerfile creates a new layer. Layers are cached and shared across images, saving disk space and speeding up builds.

```bash
docker history nginx
```

This shows every layer in the image, its size, and how it was created.

## Inspecting an Image

```bash
docker inspect nginx
```

Returns JSON with configuration, environment variables, exposed ports, volumes, and more.

## Saving and Loading Images

Transfer an image without a registry:

```bash
docker save nginx -o nginx.tar
docker load -i nginx.tar
```

## Image Size Considerations

| Image | Size | Notes |
|-------|------|-------|
| `alpine` | ~7 MB | Minimal, based on musl libc |
| `debian:slim` | ~80 MB | Smaller Debian variant |
| `ubuntu` | ~80 MB | Popular base |
| `node:alpine` | ~130 MB | Node + Alpine |
| `python:3.12-slim` | ~150 MB | Python + Debian slim |

Prefer smaller base images for faster pulls and smaller attack surface.

Next: building your own images with Dockerfiles.
