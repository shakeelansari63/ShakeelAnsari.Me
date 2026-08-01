<?php

ini_set("max_execution_time", 60);

use Slim\Factory\AppFactory;
use Slim\Middleware\BodyParsingMiddleware;
use Dotenv\Dotenv;
use Psr\Http\Message\ServerRequestInterface as Request;

require __DIR__ . "/../vendor/autoload.php";

$dotenv = Dotenv::createMutable(__DIR__ . "/..");
$dotenv->safeLoad();

$app = AppFactory::create();

$env = $_ENV["APP_ENV"] ?? "production";
$displayErrors = $env === "development";
$app->addErrorMiddleware($displayErrors, true, true);

$app->add(new BodyParsingMiddleware());

$app->setBasePath("/api");

$app->add(function (Request $request, $handler) {
    if ($request->getMethod() === "OPTIONS") {
        $response = new \Slim\Psr7\Response();
    } else {
        $response = $handler->handle($request);
    }

    $origin = $_ENV["APP_URL"] ?? "";
    if ($origin === "") {
        $origin = "http://localhost:5173";
    }
    $response = $response
        ->withHeader("Access-Control-Allow-Origin", $origin)
        ->withHeader("Vary", "Origin")
        ->withHeader(
            "Access-Control-Allow-Headers",
            "Content-Type, Accept, Authorization",
        )
        ->withHeader(
            "Access-Control-Allow-Methods",
            "GET, POST, OPTIONS",
        );
    foreach (securityHeaders() as $name => $value) {
        $response = $response->withHeader($name, $value);
    }
    return $response;
});

require __DIR__ . "/../src/helpers.php";

$pdo = \App\DB::connect();
(require __DIR__ . "/../src/blogs-routes.php")($app, $pdo);
(require __DIR__ . "/../src/learn-routes.php")($app, $pdo);
(require __DIR__ . "/../src/admin-routes.php")($app, $pdo);
(require __DIR__ . "/../src/products-routes.php")($app);

$app->run();
