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

$displayErrors = ($_ENV["APP_ENV"] ?? "development") === "development";
$app->addErrorMiddleware($displayErrors, true, true);

$app->add(new BodyParsingMiddleware());

$app->setBasePath("/api");

$app->add(function (Request $request, $handler) {
    if ($request->getMethod() === "OPTIONS") {
        $response = new \Slim\Psr7\Response();
    } else {
        $response = $handler->handle($request);
    }

    return $response
        ->withHeader("Access-Control-Allow-Origin", $_ENV["APP_URL"] ?? "*")
        ->withHeader(
            "Access-Control-Allow-Headers",
            "Content-Type, Accept, Authorization",
        )
        ->withHeader(
            "Access-Control-Allow-Methods",
            "GET, POST, PUT, DELETE, OPTIONS",
        )
        ->withHeader(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' https: data:; style-src 'self' 'unsafe-inline'; font-src 'self' data:; script-src 'self'",
        );
});

require __DIR__ . "/../src/helpers.php";

$pdo = \App\DB::connect();
(require __DIR__ . "/../src/blogs-routes.php")($app, $pdo);
(require __DIR__ . "/../src/admin-routes.php")($app, $pdo);

$app->run();
