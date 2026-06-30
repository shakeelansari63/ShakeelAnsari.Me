# Docker Compose

Docker Compose lets you define and run multi-container applications using a YAML file. Instead of typing long `docker run` commands for each service, you declare everything in `compose.yaml` (or `docker-compose.yaml`).

## Installation

Docker Desktop includes Compose. On Linux:

```bash
sudo apt install docker-compose-plugin
```

Verify:

```bash
docker compose version
```

## Basic Compose File

```yaml
services:
  web:
    image: nginx:alpine
    ports:
      - "8080:80"
    volumes:
      - ./html:/usr/share/nginx/html

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: myapp
      POSTGRES_USER: app
      POSTGRES_PASSWORD: secret
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

Start everything:

```bash
docker compose up -d
```

This creates a default network and starts both containers. They can reach each other by service name (`web`, `db`).

## Common Commands

| Command | Purpose |
|---------|---------|
| `docker compose up -d` | Start all services in background |
| `docker compose down` | Stop and remove containers + network |
| `docker compose down -v` | Also remove named volumes |
| `docker compose ps` | List service status |
| `docker compose logs -f` | Follow logs from all services |
| `docker compose logs -f web` | Follow logs from one service |
| `docker compose exec web sh` | Open shell in a running service |
| `docker compose build` | Rebuild images (for services with `build`) |
| `docker compose restart` | Restart all services |

## Building from Source

Instead of `image:`, use `build:` to build from a Dockerfile:

```yaml
services:
  app:
    build: .
    ports:
      - "3000:3000"
    environment:
      - DATABASE_URL=postgres://app:secret@db:5432/myapp
    depends_on:
      - db

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: myapp
      POSTGRES_USER: app
      POSTGRES_PASSWORD: secret
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

`depends_on` ensures the database starts before the app (but doesn't wait for it to be ready — use a health check or wait script for that).

## Environment Variables

Define variables in the compose file or use an `.env` file:

```yaml
services:
  app:
    image: myapp
    environment:
      - NODE_ENV=production
      - DEBUG=false
    env_file:
      - .env
```

## Networks

By default, Compose creates one network for all services. You can define custom networks:

```yaml
services:
  frontend:
    image: nginx
    networks:
      - web

  api:
    image: myapi
    networks:
      - web
      - internal

  db:
    image: postgres
    networks:
      - internal

networks:
  web:
  internal:
```

Services on `internal` are isolated from the frontend.

## Health Checks

Compose supports health checks to wait for services to be ready:

```yaml
services:
  db:
    image: postgres:16-alpine
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U app -d myapp"]
      interval: 5s
      timeout: 5s
      retries: 5

  app:
    image: myapp
    depends_on:
      db:
        condition: service_healthy
```

Now `app` only starts after the database passes its health check.

## Real-World Example: Web App + Redis + PostgreSQL

```yaml
services:
  redis:
    image: redis:7-alpine
    restart: always
    ports:
      - "6379:6379"

  db:
    image: postgres:16-alpine
    restart: always
    environment:
      POSTGRES_DB: myapp
      POSTGRES_USER: app
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U app -d myapp"]
      interval: 5s
      timeout: 5s
      retries: 5

  app:
    build: .
    restart: always
    ports:
      - "3000:3000"
    environment:
      DATABASE_URL: postgres://app:${DB_PASSWORD}@db:5432/myapp
      REDIS_URL: redis://redis:6379
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_started

volumes:
  pgdata:
```

Run with:

```bash
echo "DB_PASSWORD=changeme" > .env
docker compose up -d
```

## Compose vs docker run

| Feature | `docker run` | Docker Compose |
|---------|-------------|----------------|
| Single container | Good | Overkill |
| Multiple related containers | Tedious | Perfect |
| Development workflow | Manual | Built-in |
| Production deployment | Manual | Good starting point |
| Portability | Per-command | Single YAML file |

Docker Compose is the standard way to define and share multi-service applications. Next: monitoring and debugging containers.
