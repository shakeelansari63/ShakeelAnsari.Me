.PHONY: all setup ui-deps api-deps ui api start

all: start

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
