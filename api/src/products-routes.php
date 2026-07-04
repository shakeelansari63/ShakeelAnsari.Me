<?php

use Psr\Http\Message\ResponseInterface as Response;
use Psr\Http\Message\ServerRequestInterface as Request;
use Slim\App;

return function (App $app) {
    $app->get("/products/{id}/content", function (
        Request $request,
        Response $response,
        array $args,
    ) {
        $id = basename($args["id"]);
        if (!validateId($id, '/^[a-zA-Z0-9_-]+$/')) {
            return jsonResponse($response, ["error" => "Invalid product ID"], 400);
        }
        $file = PRODUCTS_DIR . "/" . $id . "/index.md";

        if (!file_exists($file)) {
            return jsonResponse($response, ["error" => "Product not found"], 404);
        }

        $raw = file_get_contents($file);
        $meta = parseFrontmatter($raw);

        $content = preg_replace(
            "/\]\((images\/)/",
            "](/api/products/{$id}/$1",
            $meta["content"],
        );

        return jsonResponse($response, [
            "title" => $meta["title"],
            "excerpt" => $meta["excerpt"],
            "content" => $content,
        ]);
    });

    $app->get("/products/{id}/images/{name}", function (
        Request $request,
        Response $response,
        array $args,
    ) {
        $id = basename($args["id"]);
        $name = basename($args["name"]);
        if (!validateId($id, '/^[a-zA-Z0-9_-]+$/') || !validateId($name, '/^[a-zA-Z0-9_\.-]+$/')) {
            return jsonResponse($response, ["error" => "Invalid path"], 400);
        }
        $file = PRODUCTS_DIR . "/" . $id . "/images/" . $name;

        if (!file_exists($file)) {
            return jsonResponse($response, ["error" => "Image not found"], 404);
        }

        $ext = strtolower(pathinfo($file, PATHINFO_EXTENSION));
        $mimeTypes = [
            "jpg" => "image/jpeg",
            "jpeg" => "image/jpeg",
            "png" => "image/png",
            "gif" => "image/gif",
            "svg" => "image/svg+xml",
            "webp" => "image/webp",
        ];
        $mime = $mimeTypes[$ext] ?? "application/octet-stream";

        $response->getBody()->write(file_get_contents($file));
        return $response->withHeader("Content-Type", $mime);
    });
};
