# shakeelansari.me

Personal website for [shakeelansari.me](https://shakeelansari.me).

## Structure

```
ui/   – React frontend (Vite)
api/  – PHP API backend (Slim Framework)
```

## Setup

```bash
make setup     # install deps for both
make start     # start both together
make ui        # start UI only (port 3000)
make api       # start API only (port 8080)
```

## Deployment

On push to `main`, the GitHub Action at `.github/workflows/deploy.yml` builds both projects and deploys to your server.

### GitHub Secrets

Create a **production** environment in your repo settings with these secrets:

| Secret           | Description                          |
|------------------|--------------------------------------|
| `DEPLOY_HOST`    | FTP server hostname                   |
| `DEPLOY_USER`    | FTP username                          |
| `DEPLOY_PASSWORD`| FTP password                          |
| `DEPLOY_PATH`    | Remote path (e.g. `/public_html`)    |
