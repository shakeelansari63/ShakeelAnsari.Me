# Local production-parity test image (used by `make test-prod`).
# Everything is built inside the image — no local compile/assembly needed.

# ---- Stage 1: Build the UI (mirrors deploy.yml "Build UI") ----
FROM docker.io/library/node:20 AS ui-build
WORKDIR /build
COPY ui/package.json ui/package-lock.json ./
RUN npm ci
COPY ui/ ./
RUN npm run build

# ---- Stage 2: PHP dependencies (mirrors deploy.yml "Build API") ----
FROM docker.io/library/composer:2 AS api-vendor
WORKDIR /app
COPY api/composer.json api/composer.lock ./
RUN composer install --no-dev --optimize-autoloader

# ---- Stage 3: Runtime - Apache + PHP, assembled like deploy/ ----
FROM docker.io/library/php:8.2-apache

# Extensions required by the API (pdo_mysql for DB, mbstring for text helpers).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libonig-dev \
    && docker-php-ext-install pdo_mysql mbstring \
    && a2enmod rewrite \
    && echo "ServerName localhost" >> /etc/apache2/apache2.conf \
    && rm -rf /var/lib/apt/lists/*

# UI build output (index.html, assets/, favicon.svg, robots.txt)
COPY --from=ui-build /build/dist/ /var/www/html/

# API (same layout as deploy/api)
COPY api/public /var/www/html/api/public
COPY api/src /var/www/html/api/src
COPY api/.htaccess /var/www/html/api/
COPY --from=api-vendor /app/vendor /var/www/html/api/vendor

# Content directories (same layout as deploy/)
COPY blogs /var/www/html/blogs
COPY tutorial /var/www/html/tutorial
COPY products /var/www/html/products

# Root .htaccess (SPA fallback -> api/public/page.php)
COPY .htaccess /var/www/html/

# NOTE: api/.env is intentionally NOT baked into the image.
# Mount it at runtime: podman run -v ./api/.env:/var/www/html/api/.env:ro
