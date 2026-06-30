# Managing Containers

Once you have several containers running, you need tools to inspect, organize, and clean them up.

## List All Containers

```bash
docker ps          # running only
docker ps -a       # all (including stopped)
docker ps -a -q    # all, IDs only (useful for scripting)
```

## Filtering Containers

```bash
docker ps --filter "status=exited"
docker ps --filter "name=my-nginx"
docker ps --filter "ancestor=nginx"   # created from the nginx image
```

## Inspecting a Container

Get detailed low-level information:

```bash
docker inspect my-nginx
```

This returns a JSON blob with everything: mount points, network settings, environment variables, etc.

For a specific field:

```bash
docker inspect my-nginx --format '{{.State.Status}}'
docker inspect my-nginx --format '{{.NetworkSettings.IPAddress}}'
```

## Resource Usage

```bash
docker stats
```

Shows live CPU, memory, network I/O, and disk I/O for all running containers. Press `Ctrl+C` to exit.

## Restart Policies

Control what happens when a container exits:

```bash
docker run -d --restart always nginx
docker run -d --restart on-failure:5 nginx
```

| Policy | Behavior |
|--------|----------|
| `no` | Never restart (default) |
| `always` | Always restart unless explicitly stopped |
| `unless-stopped` | Always restart, except when manually stopped |
| `on-failure[:max]` | Restart only on non-zero exit, optionally capped |

## Cleanup Commands

```bash
docker container prune          # remove all stopped containers
docker container prune -f       # force, no prompt
```

Remove everything unused (containers, networks, images, build cache):

```bash
docker system prune -a
```

## Resource Limits

Prevent a single container from consuming all host resources:

```bash
docker run -d \
  --memory="512m" \
  --cpus="1.5" \
  --memory-swap="1g" \
  nginx
```

| Flag | Limits |
|------|--------|
| `--memory` | Hard limit on RAM |
| `--memory-swap` | Total memory + swap (omit to disable swap) |
| `--cpus` | CPU cores (1.5 = one and a half cores) |

Next, we'll expose container ports to the outside world.
