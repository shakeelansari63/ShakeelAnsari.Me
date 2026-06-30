# Port Mapping & Volume Mounts

By default, containers are isolated from the host network and have an ephemeral filesystem. Two features bridge that gap: **port mapping** and **volume mounts**.

## Port Mapping

Expose a container's internal port to the host:

```bash
docker run -d -p 8080:80 --name web nginx
```

- `8080` — host port (where you connect from your browser)
- `80` — container port (where Nginx listens inside)

Now visit `http://localhost:8080` — Nginx is running inside the container but accessible from the host.

### Multiple Ports

```bash
docker run -d \
  -p 8080:80 \
  -p 8443:443 \
  --name web nginx
```

### Random Host Port

Let Docker choose an available host port:

```bash
docker run -d -p 80 --name web nginx
```

Check which port was assigned with:

```bash
docker port web
```

### Binding to a Specific Interface

```bash
docker run -d -p 127.0.0.1:8080:80 nginx   # localhost only
docker run -d -p 0.0.0.0:8080:80 nginx     # all interfaces
```

## Volume Mounts

Containers are ephemeral — when you remove a container, its data is gone. **Volumes** persist data outside the container's lifecycle.

### Bind Mounts

Mount a host directory into the container:

```bash
docker run -d \
  -v /home/user/data:/usr/share/nginx/html \
  -p 8080:80 \
  nginx
```

- `/home/user/data` — absolute path on the host
- `/usr/share/nginx/html` — path inside the container

Changes on either side are immediately visible on the other.

### Named Volumes

Managed by Docker. More portable than bind mounts:

```bash
docker volume create my-data

docker run -d \
  -v my-data:/usr/share/nginx/html \
  -p 8080:80 \
  nginx
```

List and inspect volumes:

```bash
docker volume ls
docker volume inspect my-data
```

### Anonymous Volumes

Docker creates a random-named volume:

```bash
docker run -d -v /data nginx
```

Useful when you don't need to manage the volume directly.

## Combining Ports and Volumes

A real-world example serving custom content:

```bash
docker run -d \
  --name my-site \
  -p 8080:80 \
  -v /home/user/my-site:/usr/share/nginx/html:ro \
  nginx
```

The `:ro` flag makes the mount read-only inside the container for extra safety.

## Practical Example: PostgreSQL with Persistent Data

```bash
docker volume create pgdata

docker run -d \
  --name postgres-db \
  -e POSTGRES_PASSWORD=secret \
  -e POSTGRES_DB=mydb \
  -v pgdata:/var/lib/postgresql/data \
  -p 5432:5432 \
  postgres:16
```

Now your database survives container restarts and removals.

Next: working with Docker images — pulling, tagging, and removing.
