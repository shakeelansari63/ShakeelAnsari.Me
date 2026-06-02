# shakeelansari.me

Personal portfolio website for [shakeelansari.me](https://shakeelansari.me) — a full-stack app with a React frontend, PHP Slim Framework API backend, and MySQL database.

## Tech Stack

| Layer    | Technology                                      |
|----------|-------------------------------------------------|
| Frontend | React 18, TypeScript, Vite, PrimeReact, PrimeFlex |
| Backend  | PHP 8.1+, Slim Framework 4, PDO, JWT Auth       |
| Database | MySQL                                           |
| Deploy   | GitHub Actions → FTP                            |

## Project Structure

```
├── ui/          – React frontend (Vite)
│   ├── src/
│   │   ├── components/   – React components
│   │   ├── pages/        – Page-level components
│   │   ├── services/     – API client, data, stats
│   │   └── models/       – TypeScript interfaces
│   └── public/fonts/     – SpaceMono font files
├── api/         – PHP API backend (Slim Framework)
│   ├── public/           – Entry point (index.php)
│   ├── src/              – Routes (blogs, admin), helpers
│   └── db/               – Database schema (schema.sql)
├── blogs/       – Markdown blog posts and images
└── .github/     – CI/CD workflow
```

---

## Running Locally

### Prerequisites

- Node.js 20+
- PHP 8.1+
- Composer
- MySQL server (optional for full features; API works without DB)

### 1. Setup

```bash
# Install all dependencies
make setup
```

Or manually:

```bash
cd ui && npm install
cd api && composer install
```

### 2. Configure Environment

```bash
cp api/.env.template api/.env
```

Edit `api/.env` and fill in your values:

| Variable         | Description                     |
|------------------|---------------------------------|
| `APP_ENV`        | `development` or `production`   |
| `APP_URL`        | Frontend URL (e.g. `http://localhost:3000`) |
| `ADMIN_USERNAME` | Admin panel login username      |
| `ADMIN_PASSWORD` | Admin panel login password      |
| `JWT_SECRET`     | Secret key for JWT tokens       |
| `DB_HOST`        | MySQL host (leave empty to skip DB) |
| `DB_PORT`        | MySQL port (default `3306`)     |
| `DB_NAME`        | MySQL database name             |
| `DB_USER`        | MySQL user                      |
| `DB_PASS`        | MySQL password                  |

### 3. Database Setup (Optional)

If using MySQL, create the database and run the schema:

```bash
mysql -u root -p your_database_name < api/db/schema.sql
```

This creates three tables:
- **`blog`** — Blog posts synced from markdown files
- **`blog_views`** — View tracking (dedup by IP + hour)
- **`blog_likes`** — Like tracking (dedup by IP)

Without a database, the API runs in degraded mode (blogs list and content still work via markdown files, but views/likes are not recorded).

### 4. Start Development Servers

Start both frontend and backend together:

```bash
make start
```

Or separately:

```bash
make ui     # React dev server on http://localhost:3000
make api    # PHP server on http://localhost:8080
```

The UI proxies `/api/*` requests to the PHP server automatically (configured in `ui/vite.config.js`).

Open [http://localhost:3000](http://localhost:3000) in your browser.

### 5. Sync Blog Posts

After setting up the database, sync markdown files from `blogs/` into the DB:

1. Visit `http://localhost:3000/admin`
2. Log in with the credentials from `api/.env`
3. Click **Sync Blogs**

---

## Deployment

### Manual Deploy to PHP Server with MySQL

1. **Build the frontend:**

```bash
cd ui && npm run build
```

2. **Upload to your server** with this structure:

```
public_html/
├── index.html             # from ui/dist/
├── assets/                # from ui/dist/assets/
├── api/
│   ├── public/            # from api/public/
│   │   └── index.php
│   ├── src/               # from api/src/
│   ├── vendor/            # from api/vendor/
│   ├── .htaccess          # from api/.htaccess
│   └── .env               # production environment file
├── blogs/                 # from blogs/
│   ├── images/
│   └── *.md
└── .htaccess              # from root .htaccess
```

3. **Set up the database** on your MySQL server:

```bash
mysql -u your_user -p your_database < api/db/schema.sql
```

4. **Configure `api/.env`** on the server with your production values (DB credentials, admin login, JWT secret, `APP_ENV=production`).

5. **Set up Apache rewrite** — the root `.htaccess` routes API calls to `api/` and SPA routes to `index.html`. Ensure `mod_rewrite` and `AllowOverride` are enabled.

6. **Sync blog posts** by visiting `https://yoursite.com/admin`.

### Automated Deploy via GitHub Actions

On push to `main`, the CI/CD pipeline at `.github/workflows/deploy.yml` builds both projects and deploys via FTP.

### Required GitHub Secrets

Create a **production** environment in your repo settings with:

| Secret             | Description                          |
|--------------------|--------------------------------------|
| `DEPLOY_HOST`      | FTP server hostname                  |
| `DEPLOY_USER`      | FTP username                         |
| `DEPLOY_PASSWORD`  | FTP password                         |
| `DEPLOY_PATH`      | Remote path (e.g. `/public_html`)    |
| `ADMIN_USERNAME`   | Admin panel login username           |
| `ADMIN_PASSWORD`   | Admin panel login password           |
| `JWT_SECRET`       | Secret key for JWT tokens            |
| `DB_HOST`          | MySQL host                           |
| `DB_NAME`          | MySQL database name                  |
| `DB_USER`          | MySQL user                           |
| `DB_PASS`          | MySQL password                       |
| `APP_ENV`          | `production`                         |
| `APP_URL`          | `https://yourdomain.com`             |

---

## Features

- **Homepage** — GitHub profile, contribution heatmap, stats cards, project listing
- **Blog** — Markdown-based blog with pagination, views, likes, and sharing
- **Expo** — Portfolio showcase with App/Code links
- **Admin** — Password-protected panel for syncing blog posts
- **Dark/Light mode** — Toggle on blog reader
- **Responsive** — Mobile-first layout with PrimeFlex

## API Endpoints

| Method | Path                  | Description                     |
|--------|-----------------------|---------------------------------|
| GET    | `/api/blogs`          | Paginated blog list             |
| GET    | `/api/blogs/{id}`     | Single blog metadata            |
| GET    | `/api/blogs/{id}/content` | Blog markdown content       |
| GET    | `/api/blogs/{id}/stats`   | Views, likes, liked status  |
| POST   | `/api/blogs/{id}/view`    | Record a view               |
| POST   | `/api/blogs/{id}/like`    | Toggle like                 |
| POST   | `/api/admin/login`        | Admin authentication        |
| POST   | `/api/admin/sync-blogs`   | Sync markdown files to DB   |
