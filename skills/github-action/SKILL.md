# GitHub Actions CI/CD Skill

## Pipeline Overview
**File:** `.github/workflows/deploy.yml`  
**Trigger:** Push to `main` branch  
**Deployment:** FTP to Hostinger with parallel transfers (lftp)

## Pipeline Steps

### 1. Checkout
Standard `actions/checkout@v4` — clean checkout of the repo.

### 2. Replace SEO Placeholders
Use `sed` to replace `[{#SEO-*#}]` tokens in source files with environment variables **before** build:
```yaml
- name: Replace SEO Tokens
  run: |
    sed -i "s/\[{#SEO-TITLE#}\]/${{ vars.SEO_TITLE }}/g" ui/index.html
    sed -i "s/\[{#SEO-DESCRIPTION#}\]/${{ vars.SEO_DESCRIPTION }}/g" ui/index.html
    # ... repeat for all SEO tokens
```
- Tokens are defined in the workflow environment for production and can be overridden via GitHub Secrets.
- **Never replace these tokens in source files during development** — only in CI/CD pipeline.
- Run this step **before** building UI and API.

### 3. Build UI
```yaml
- name: Build UI
  working-directory: ./ui
  run: |
    npm ci
    npm run build
```
- Node 20.x.
- Uses `npm ci` for clean install (not `npm install`).
- Output goes to `ui/dist/`.

### 4. Build API
```yaml
- name: Build API
  working-directory: ./api
  run: |
    composer install --no-dev --optimize-autoloader
```
- PHP 8.2.
- Composer with `--no-dev` (no dev dependencies on production) and `--optimize-autoloader` for performance.
- Output: `api/vendor/`, `api/src/`, `api/public/`.

### 5. Generate .env
Generate `api/.env` from `api/.env.template` by replacing `{#VAR#}` placeholders:
```yaml
- name: Generate .env
  run: |
    cp api/.env.template api/.env
    sed -i "s/{#DB_HOST#}/${{ secrets.DB_HOST }}/g" api/.env
    # ... repeat for all template variables
```
- Never commit `.env` to the repository.
- Source of truth for production env vars is GitHub Secrets.
- The template includes `SEO_*` vars (name, title, domain, desc, desc_short, github, linkedin, twitter_user, avatar_url). They must be **double-quoted in the template** (`SEO_NAME="{#SEO_NAME#}"`) because their values contain spaces — an unquoted value makes phpdotenv throw `InvalidFileException`, which `safeLoad()` does NOT catch, crashing the whole API.
- The `grep -oP '{#\K[^#]+(?=#})'` + `sed` loop automatically picks up any new `{#VAR#}` token added to the template, so no workflow change is needed when adding vars.

### 6. Assemble Deployable
Copy all artifacts into a `deploy/` directory:
```
deploy/
├── index.html              # from ui/dist/
├── assets/                 # from ui/dist/assets/
├── api/
│   ├── public/             # from api/public/ (index.php, page.php, sitemap.php, .htaccess)
│   ├── src/                # from api/src/
│   ├── vendor/             # from api/vendor/
│   ├── .htaccess           # from api/.htaccess
│   └── .env                # generated in step 5
├── blogs/                  # root-level blogs/
├── tutorial/               # root-level tutorial/
├── products/               # root-level products/
└── .htaccess               # root-level .htaccess
```
- Do NOT include `api/db/`, `api/composer.*`, `api/.env.template`, `node_modules/`, `ui/src/`, `ui/public/`, `.git/`, `.github/`, `skills/`, `prompts/`.
- Do NOT include source files that have already been compiled or processed.

### 7. Deploy via FTP
```yaml
- name: Deploy to Hostinger
  run: |
    lftp -c "set ssl:verify-certificate no; open -u ${{ secrets.DEPLOY_USER }},'${{ secrets.DEPLOY_PASSWORD }}' ${{ secrets.DEPLOY_HOST }}; mirror -R --parallel=10 --delete-first deploy/ ${{ secrets.DEPLOY_PATH }}"
```
- Uses lftp with parallel transfers (`--parallel=10`).
- SSL enabled but cert verification disabled for Hostinger.
- `--delete-first` removes files on remote that don't exist in local deploy/.
- Requires these GitHub Secrets (production environment):
  - `DEPLOY_HOST`, `DEPLOY_USER`, `DEPLOY_PASSWORD`, `DEPLOY_PATH`
  - `ADMIN_USERNAME`, `ADMIN_PASSWORD`, `JWT_SECRET`
  - `DB_HOST`, `DB_NAME`, `DB_USER`, `DB_PASS`
  - SEO token values as `vars` (organization variables) or `secrets`.

## Local Pre-Deploy Test (`make test-prod`)

Before pushing to `main`, test the production build locally with **Podman** (no Node/PHP/Apache install needed on the host):
- `test-prod.dockerfile` is a **multi-stage** build: `node:20` runs `npm ci` + `npm run build` (mirrors deploy.yml's Build UI), `composer:2` runs `composer install --no-dev --optimize-autoloader` (Build API), and the final `php:8.2-apache` stage (`pdo_mysql`, `mbstring`, `mod_rewrite`) assembles the docroot exactly like the deploy step does — UI dist, `api/public`, `api/src`, `api/vendor`, `api/.htaccess`, content dirs, root `.htaccess`.
- `make test-prod-image` builds `shakeel-test-prod`; `make test-prod` builds (cached) and runs:
  ```
  podman run --rm -p 127.0.0.1:8081:80 \
      -v ./api/.env:/var/www/html/api/.env:ro \
      shakeel-test-prod
  ```
- The container's Apache processes the **real `.htaccess` chain** (root → `api/` → `api/public/`), so `page.php` serves per-URL SEO heads exactly like production on Hostinger. **No router file, no local build.**
- `api/.env` is **not baked into the image** — it's mounted read-only at runtime (so `.env` edits don't require a rebuild, and no secrets land in image storage). Requires `api/.env` to exist locally; run `make api-deps` first if you haven't `composer install`ed. The API's remote DB (`DB_HOST`) is reachable over bridge networking.
- `.dockerignore` keeps `.git`, `node_modules`, `vendor`, `.env`, etc. out of the build context.
- Stop with Ctrl-C; `--rm` removes the container. Override port with `PORT=xxxx make test-prod`.

## Workflow Best Practices

### Environment Variables vs Secrets
- **Secrets** (`${{ secrets.* }}`): Sensitive data (passwords, keys, tokens).
- **Variables** (`${{ vars.* }}`): Non-sensitive config (SEO titles, descriptions).
- Use GitHub Environments to group by deploy target (production/staging).

### Required Production Secrets
```
DEPLOY_HOST      → FTP hostname
DEPLOY_USER      → FTP username
DEPLOY_PASSWORD  → FTP password
DEPLOY_PATH      → Remote path (e.g., public_html/)
ADMIN_USERNAME   → Auth admin username
ADMIN_PASSWORD   → Auth admin password
JWT_SECRET       → JWT signing secret
DB_HOST          → MySQL host
DB_NAME          → MySQL database name
DB_USER          → MySQL user
DB_PASS          → MySQL password
```

## Critical Guardrails
- ❌ **Never commit `.env` files or secrets** to the repository.
- ❌ **Never replace `[{#SEO-*#}]` in source files manually** — only in CI/CD pipeline via `sed`.
- ❌ **Never deploy without running the build steps** — deploy `ui/dist/` output, not `ui/src/`.
- ❌ **Never change the deployment tool** (`lftp`) without verifying Hostinger compatibility.
- ❌ **Never skip the SEO token replacement step** — tokens will appear raw in production.
- ✅ Always use `npm ci` (not `npm install`) for reproducible builds.
- ✅ Always use `composer install --no-dev --optimize-autoloader` for production.
- ✅ Always deploy to a clean directory (use `--delete-first` on mirror).
