.PHONY: all setup ui-deps api-deps ui api start test-prod test-prod-image

all: start

PORT ?= 8081
TEST_PROD_IMAGE ?= test-portfolio-prod
# Strip a polluted LD_LIBRARY_PATH (e.g. leaked from the Flatpak Zed runtime)
# so conmon loads the system glib instead of crashing on missing symbols.
PODMAN := env -u LD_LIBRARY_PATH podman

setup: ui-deps api-deps

ui-deps:
	cd ui && npm install

api-deps:
	cd api && composer install

ui:
	cd ui && npm run dev

api:
	cd api && PHP_CLI_SERVER_WORKERS=4 php -S localhost:8080 -t public

start:
	@echo "Starting UI (port 3000) and API (port 8080)..."
	@trap 'kill 0' EXIT; \
		$(MAKE) api & \
		$(MAKE) ui & \
		wait

# Build a production-parity image and serve it with Apache in Podman.
# The multi-stage test-prod.dockerfile builds the UI, PHP deps, and final
# docroot inside the image — nothing is compiled or assembled locally.
# api/.env is mounted read-only at runtime (not baked into the image).
# Requires api/.env to exist locally (run `make api-deps` first if vendor is
# missing, and copy api/.env.template to api/.env). Stop with Ctrl-C.
test-prod: test-prod-image
	@test -f api/.env || { echo "error: api/.env missing — copy api/.env.template to api/.env first"; exit 1; }
	@echo "Serving production-like site at http://localhost:$(PORT) (Apache in Podman)"
	$(PODMAN) run --rm -p $(PORT):80 \
		-v "$(CURDIR)/api/.env:/var/www/html/api/.env:ro" \
		$(TEST_PROD_IMAGE)

test-prod-image:
	$(PODMAN) build -t $(TEST_PROD_IMAGE) -f test-prod.dockerfile .
