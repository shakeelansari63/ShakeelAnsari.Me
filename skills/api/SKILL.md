# API Development Skill — `shakeelansari.me`

## Context
This project is a **PHP 8.1+ REST API** using **Slim Framework 4** with naked **PDO** for database access and **JWT** for admin auth. There is **no ORM, no Node.js backend, no Express/Fastify**.

## Architecture Rules

### Route Organization
- Routes live in `/api/src/` as standalone PHP files, **not** classes.
- Each route file follows the pattern:
  ```php
  return function (App $app, ?PDO $pdo) {
      $app->get('/api/blogs', function (Request $request, Response $response) use ($pdo) {
          // ...
          return jsonResponse($response, $data);
      });
  };
  ```
- Route files are `require`-d from `api/public/index.php` and receive `($app, $pdo)`.
- Split routes by domain: `blogs-routes.php`, `learn-routes.php`, `products-routes.php`, `admin-routes.php`.

### Response Format
- All responses use the `jsonResponse()` helper from `helpers.php`:
  ```php
  jsonResponse($response, $data, $statusCode);
  ```
- Always use `JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE` flags.

### Database Access
- **No ORM.** Use raw PDO with **named/prepared parameters** (`?` placeholders).
- Connection is via `App\DB::connect()` which returns `null` in degraded mode (DB_HOST empty).
- Every DB query must be in try/catch. If PDO is null, return 503 or graceful fallback.
- All PDO connections run with: `SET SESSION max_execution_time = 30000`.
- PHP runtime caps at `max_execution_time = 60` seconds — keep loops chunked.

### Authentication (JWT)
- Admin endpoints **must** gate-check with `verifyAdmin(Request $request): bool`.
- The function extracts Bearer token from `Authorization` header, decodes with `JWT_SECRET` env var using HS256.
- Login uses `hash_equals()` for credential comparison.
- Rate limiting for login: 5 attempts per 15 min per IP (file-based).

### Security Rules
- Use `basename()` when dealing with file paths to prevent directory traversal.
- Use `hash_equals()` for string comparison of secrets.
- Parameterize ALL SQL queries — never interpolate user input. Use prepared statements with `?` placeholders via `$pdo->prepare()` — never `$pdo->quote()` string interpolation.
- CORS middleware handles OPTIONS preflight and sets `Access-Control-Allow-Origin`, `Access-Control-Allow-Headers`, `Access-Control-Allow-Methods`, `Content-Security-Policy`.
- `Access-Control-Allow-Origin` must fall back to `http://localhost:5173` in dev, never `*`.
- `APP_ENV` defaults to `production`, not `development`.
- **Conditional WHERE clauses:** When building dynamic SQL that conditionally appends `WHERE` clauses (e.g., analytics blog filter), insert the clause BEFORE `GROUP BY`/`ORDER BY`. Use `preg_replace('/\b(GROUP\s+BY)\b/i', 'WHERE ... $1', $base, 1)` to insert at the correct position.
- **JWT:** Guard with `jwtSecret()` — reject empty or <16-char `JWT_SECRET`. Login must check for empty credentials early. Use `verifyAdmin(Request $request): bool` with HS256 decode.
- **Input validation:** Use `validateId(string $id, string $pattern): bool` with regex pattern checks on ALL route parameters (learn subjects, chapters, images; products, images; blogs).
- **Trusted proxies:** In `clientIp()`, restrict `X-Forwarded-For` trust to `TRUSTED_PROXIES` env var only. Reject private/reserved IPs from the header (`FILTER_FLAG_NO_PRIV_RANGE | FILTER_FLAG_NO_RES_RANGE`).
- **IP geolocation:** Always use `https://` for ip-api.com with a 3-second timeout via `stream_context_create`.
- **Rate limiting:** Login rate limiter file operations must use `flock(LOCK_EX)` for atomic read-modify-write and `flock(LOCK_SH)` for reads.

### Image Serving
- Blog images: serve from `blogs/images/` at `/api/blogs/images/{name}`.
- Learn images: serve from `tutorial/{folder}/images/` at `/api/learn/images/{folder}/{name}`.
- Product images: serve from `products/{id}/images/` at `/api/products/{id}/images/{name}`.
- Read the file, detect MIME type, set `Content-Type` header, output with `fpassthru()`.

### Markdown Content (Sync Logic)
**Blog sync algorithm:**
1. Scan `blogs/*.md` for markdown files.
2. Parse YAML frontmatter using `parseFrontmatter()` helper — extracts title, excerpt, date, readTime, tags, bannerImage, content.
3. Upsert into `blog` table (`INSERT ... ON DUPLICATE KEY UPDATE`).
4. Soft-delete (set `deleted = 1`) any blog IDs not in file listing.

**Learn sync algorithm:**
1. Scan subdirectories of `tutorial/` (pattern: `{sort_order}-{title}`).
2. Upsert each into `learn_subjects`.
3. Scan `*.md` files in each folder (pattern: `ch{sort_order}-{title}.md`).
4. Upsert each into `learn_chapters`.
5. Remove subjects/chapters not in current scan.
6. Detect `images/thumbnail.*` for subject card thumbnail.

### Helpers (from `api/src/helpers.php`)
- `parseFrontmatter(string $content): array` — parses `---` delimited YAML frontmatter.
- `rewriteImageUrls(string $content): string` — rewrites `](images/` to `](/api/blogs/` (for blogs) or `/api/products/{id}/` (for products).
- `isRateLimited(string $ip): bool` — file-based rate limiting check.
- `incrementAttempts(string $ip): void` — increment login counter.
- `resetAttempts(string $ip): void` — reset login counter.
- `clientIp(Request $request): string` — extract client IP (respects X-Forwarded-For).
- `resolveLocation(string $ip, PDO $pdo): object` — IP geolocation via ip-api.com with DB caching.
- `jsonResponse($response, $data, $status): ResponseInterface` — standard JSON response.

### Coding Style
- No classes for route handlers — pure functions.
- Variables in `camelCase`.
- Error responses return JSON with `error` key.
- When DB is unavailable, return 503 with descriptive message.
- Use type hints where possible (`?PDO`, `Request`, `Response`, `string`, `int`).

### What NOT to Do
- ❌ Never write Node.js backend code (Express, Fastify, etc.).
- ❌ Never use an ORM (Eloquent, Doctrine, etc.).
- ❌ Never hardcode backend URLs in frontend code — use Vite proxy.
- ❌ Never modify `[{#SEO-*#}]` placeholders — they are CI/CD pipeline tokens.
- ❌ Never commit `.env` files or secrets.
- ❌ Never introduce long-running loops without chunking (PHP max_execution_time = 60s).
