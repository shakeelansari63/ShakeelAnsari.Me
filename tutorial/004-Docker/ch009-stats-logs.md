# Monitoring & Debugging

Once your containers are running in production (or even locally), you need tools to observe their behavior and diagnose problems.

## Real-Time Resource Monitoring

```bash
docker stats
```

Shows live CPU, memory, network I/O, and disk I/O for all running containers.

```bash
docker stats --no-stream    # single snapshot
docker stats my-nginx       # specific container
docker stats --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"
```

## Viewing Logs

```bash
docker logs my-nginx                    # all logs
docker logs --tail 100 my-nginx         # last 100 lines
docker logs -f my-nginx                 # follow (like tail -f)
docker logs --since 2024-01-01 my-nginx  # since a date
docker logs --until 2024-01-02 my-nginx  # until a date
```

## Inspecting Processes Inside a Container

```bash
docker top my-nginx
```

Shows running processes within the container, similar to `ps aux` on the host.

## Docker Events

Stream real-time events from the Docker daemon:

```bash
docker events
docker events --filter 'container=my-nginx'
docker events --filter 'event=stop'
```

## Debugging with Interactive Shell

```bash
docker exec -it my-nginx sh      # Alpine-based images
docker exec -it my-nginx bash    # Debian/Ubuntu-based
```

Once inside, use standard Linux tools:

```bash
# Check network connectivity
curl http://localhost
ping db

# Check disk usage
df -h

# Inspect environment variables
env

# Check running processes
ps aux
```

## Copying Files

```bash
docker cp my-nginx:/etc/nginx/nginx.conf ./nginx.conf
docker cp ./custom.conf my-nginx:/etc/nginx/conf.d/
```

## Container Logs to JSON File

```bash
docker logs my-nginx > /tmp/nginx-logs.json
```

## Debugging Startup Failures

If a container exits immediately:

```bash
# Run with --rm to auto-clean
docker run --rm my-app

# Check exit code
docker ps -a --filter "name=my-app"

# View logs from the failed run
docker logs my-app

# Override the entrypoint to debug
docker run -it --entrypoint sh my-app
```

## Common Debugging Scenarios

### Container exits immediately
```bash
docker logs <container>          # check error message
docker run -it <image> sh        # enter and run manually
```

### Network connectivity issues
```bash
docker exec -it <container> ping google.com
docker exec -it <container> curl http://other-service:8080
docker network inspect bridge
```

### Disk space issues
```bash
docker system df                # show disk usage
docker system prune -a          # clean everything unused
du -sh /var/lib/docker/         # Docker's root directory
```

### Permission issues
```bash
# Check what user the container runs as
docker exec <container> whoami

# Run as a different user
docker run -u 1000:1000 my-app
```

## Cleanup Commands Reference

```bash
docker container prune       # remove stopped containers
docker image prune           # remove dangling images
docker volume prune          # remove unused volumes
docker network prune         # remove unused networks
docker system prune -a       # remove everything unused
```

This concludes the Docker tutorial series. You now have a solid foundation to build, run, and manage Docker containers for development and production.
