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
│   └── index.html        – Vite entry with meta tags, OG, schema.org JSON-LD
├── api/         – PHP API backend (Slim Framework)
│   ├── public/           – Entry point (index.php, sitemap.php)
│   ├── src/              – Routes (blogs, admin), helpers, DB
│   └── db/               – Database schema (schema.sql)
├── blogs/       – Markdown blog posts and images
├── products/    – Markdown product pages (detailed write-ups)
├── prompts/     – Reusable AI system prompts (e.g. blog writer)
├── tutorial/    – Multi-subject tutorial content (Python, Golang, Scala, etc.)
└── .github/     – CI/CD workflow
```

---

## Running Locally

### Prerequisites

- Node.js 20+
- PHP 8.1+
- Composer
- Make
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

This creates five tables:
- **`blog`** — Blog posts synced from markdown files
- **`blog_views`** — View tracking (dedup by IP + 8-hour window)
- **`blog_likes`** — Like tracking (dedup by IP)
- **`learn_subjects`** — Tutorial subjects (e.g. Python, Golang, Scala)
- **`learn_chapters`** — Individual tutorial chapters per subject

Without a database, the API runs in degraded mode (blogs list/content still work via markdown files, but views/likes and learn/tutorial features are unavailable).

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

### 5. Sync Blog Posts & Tutorials

After setting up the database, sync markdown files:

1. Visit `http://localhost:3000/admin`
2. Log in with the credentials from `api/.env`
3. Click **Sync Blogs** to ingest/update blog posts from `blogs/`
4. Click **Sync Learn** to ingest/update tutorial subjects and chapters from `tutorial/`

### Defensive Timeouts

The app includes two layers of timeout protection:

- **PHP `max_execution_time = 60`** — set in `api/public/index.php:3` and `api/public/sitemap.php:3`. Self-terminates any script that hangs for 60 seconds of CPU time.
- **MySQL query timeout** — `SET SESSION max_execution_time = 30000` (30 seconds) executed after every DB connection in `api/src/DB.php:26-29`. Kills SELECT queries that exceed 30 seconds (MySQL 8.0+).

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
├── tutorial/              # from tutorial/ (optional — for Learn platform)
├── products/              # from products/ (optional — for product detail pages)
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

---

## Features

- **Homepage** — GitHub profile card (avatar, bio), contribution heatmap calendar, stats cards (repos, stars, followers, total contributions), language breakdown, streak stats, project listing, work experience timeline
- **Blog** — Markdown-based blog with pagination, syntax-highlighted code blocks (`react-syntax-highlighter`), Mermaid diagram rendering, view tracking (IP-deduped per 8-hour window), interactive like/unlike toggle, share support via Web Share API
- **Learn/Tutorials** — Multi-subject tutorial platform with chapters, subject thumbnails, markdown content rendering with Mermaid diagram support. Subjects auto-synced from `tutorial/` directory
- **Expo** — Portfolio/project showcase with App and Code buttons (optional URLs, hidden when not provided). Detailed product pages via markdown in `products/` directory
- **Admin** — JWT-authenticated password-protected panel with file-based rate limiting (5 attempts per 15 minutes per IP), one-click blog sync from markdown files (upserts + marks deleted files), one-click learn/tutorial sync from `tutorial/` directory (subjects + chapters)
- **UI/UX** — Dark/light theme toggle, scroll-shrink toolbar animation, lazy image loading with skeleton placeholders, skeleton loading cards, responsive mobile sidebar navigation, custom 404 page
- **Performance** — Code-splitting via `React.lazy()` + `Suspense` (each page loads as its own chunk), vendor-split PrimeReact into a separate cacheable chunk, lazy-loaded mobile sidebar, below-fold chunking, gzip compression
- **SEO** — Dynamic XML sitemap (auto-includes blog entries), Open Graph tags, Twitter cards, schema.org JSON-LD structured data, `<meta name="robots">`, canonical URL

## API Endpoints

| Method | Path                              | Description                              |
|--------|-----------------------------------|------------------------------------------|
| GET    | `/api/blogs`                      | Paginated blog list                      |
| GET    | `/api/blogs/{id}`                 | Single blog metadata                     |
| GET    | `/api/blogs/{id}/content`         | Blog markdown content                    |
| GET    | `/api/blogs/{id}/stats`           | Views, likes, liked status               |
| POST   | `/api/blogs/{id}/view`            | Record a view                            |
| POST   | `/api/blogs/{id}/like`            | Toggle like                              |
| GET    | `/api/blogs/images/{name}`        | Blog image asset                         |
| GET    | `/api/learn/subjects`             | List tutorial subjects                   |
| GET    | `/api/learn/subjects/{id}/chapters` | List chapters for a subject            |
| GET    | `/api/learn/chapters/{id}/content`  | Chapter markdown content               |
| GET    | `/api/learn/images/{folder}/{name}` | Tutorial image asset                  |
| GET    | `/api/products/{id}/content`      | Product detail page markdown content     |
| GET    | `/api/products/{id}/images/{name}`  | Product image asset                    |
| POST   | `/api/admin/login`                | Admin authentication (JWT)               |
| POST   | `/api/admin/sync-blogs`           | Sync blog markdown files to DB           |
| POST   | `/api/admin/sync-learn`           | Sync tutorial markdown files to DB       |

---

## SEO & Customization

The project uses `[{#SEO-KEY#}]` placeholders for SEO/meta values that vary per deployment. The CI/CD pipeline replaces them before build. Static personal data (profile, projects, work history) lives in source files and is edited directly.

### SEO Placeholders

These appear in `ui/index.html`, page `document.title` calls, `ui/public/robots.txt`, and `api/public/sitemap.php`:

| Placeholder | Purpose | Example Value |
|---|---|---|
| `[{#SEO-NAME#}]` | Full name (page titles, meta, JSON-LD) | `Shakeel Ansari` |
| `[{#SEO-TITLE#}]` | Job title / tagline | `Data, AI & Full-Stack Engineer` |
| `[{#SEO-DOMAIN#}]` | Domain name (without protocol) | `shakeelansari.me` |
| `[{#SEO-DESC#}]` | Meta description (long) | `Personal portfolio showcasing...` |
| `[{#SEO-DESC-SHORT#}]` | Twitter description (short) | `Data, AI & Full-Stack Engineer. Portfolio, blogs, and projects.` |
| `[{#SEO-GITHUB-URL#}]` | Full GitHub profile URL | `https://github.com/shakeelansari63` |
| `[{#SEO-LINKEDIN-URL#}]` | Full LinkedIn profile URL | `https://www.linkedin.com/in/shakeelansari63/` |
| `[{#SEO-TWITTER-USER#}]` | Twitter/X username | `shakeelansari63` |
| `[{#SEO-AVATAR-URL#}]` | Avatar image URL (OG image) | `https://avatars.githubusercontent.com/shakeelansari63` |

### Replacing Placeholders

**For CI/CD (GitHub Actions):** Edit the `SEO_*` environment variables in `.github/workflows/deploy.yml` with your values:

```yaml
SEO_NAME: "Your Name"
SEO_TITLE: "Your Title"
SEO_DOMAIN: "yourdomain.com"
SEO_DESC: "Your meta description"
SEO_DESC_SHORT: "Your short description"
SEO_GITHUB_URL: "https://github.com/your-username"
SEO_LINKEDIN_URL: "https://linkedin.com/in/your-profile"
SEO_TWITTER_USER: "your-handle"
SEO_AVATAR_URL: "https://avatars.githubusercontent.com/your-username"
```

**For local development:** Run the same `sed` commands from the workflow to replace placeholders, or simply ignore them — they only affect meta tags and page titles, and won't break functionality.

### Static Data (edit directly)

These files contain your personal data and must be updated manually:

| File | What to edit |
|---|---|
| `ui/src/data/profile.ts` | GitHub username, social links, email, timezone (used by footer too) |
| `ui/src/data/skills.ts` | Skills list |
| `ui/src/data/work.ts` | Work experience / job history |
| `ui/src/data/expo.ts` | Portfolio projects (name, description, URLs) |
| `ui/public/favicon.svg` | Replace with your own SVG favicon |
| `ui/index.html:13-18` | Google Fonts (optional) |

### Environment Template (`api/.env.template`)

| Variable | Default | Description |
|---|---|---|
| `APP_ENV` | `development` | Set `production` on deploy |
| `APP_URL` | `http://localhost:3000` | Set to `https://[your-domain].com` in production |
| `ADMIN_USERNAME` | _(empty)_ | Your admin username |
| `ADMIN_PASSWORD` | *(empty)* | Your admin password |
| `JWT_SECRET` | *(empty)* | A random secure string |
| `DB_HOST` | `localhost` | MySQL host (leave empty to skip DB) |
| `DB_PORT` | `3306` | MySQL port |
| `DB_NAME` | *(empty)* | MySQL database name |
| `DB_USER` | *(empty)* | MySQL user |
| `DB_PASS` | *(empty)* | MySQL password |

---

## ❤️ Credits

This work wouldn't be possible without the hard work of following repos' amazing owners.

- [`@primefaces/primereact`](https://github.com/primefaces/primereact)
- [`@said7388/github-portfolio`](https://github.com/said7388/github-portfolio)
- [`@vn7n24fzkq/github-profile-summary-cards`](https://github.com/vn7n24fzkq/github-profile-summary-cards)
- [`@anuraghazra/github-readme-stats`](https://github.com/anuraghazra/github-readme-stats)
- [`@denvercoder1/github-readme-streak-stats`](https://github.com/denvercoder1/github-readme-streak-stats)
- [`@rschristian/github-contribution-calendar-api`](https://github.com/rschristian/github-contribution-calendar-api)
- [`@cjosue15/ngx-heatmap-calendar`](https://github.com/cjosue15/ngx-heatmap-calendar)

Go ahead give them some stars as well.
