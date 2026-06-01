<?php

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
        );
});

$pdo = null;
if (!empty($_ENV["DB_HOST"])) {
    try {
        $dsn = sprintf(
            "mysql:host=%s;port=%s;dbname=%s;charset=utf8mb4",
            $_ENV["DB_HOST"],
            $_ENV["DB_PORT"] ?? "3306",
            $_ENV["DB_NAME"],
        );
        $pdo = new PDO($dsn, $_ENV["DB_USER"], $_ENV["DB_PASS"], [
            PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION,
            PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
            PDO::ATTR_EMULATE_PREPARES => false,
        ]);
    } catch (PDOException $e) {
        error_log("DB connection failed: " . $e->getMessage());
    }
}

require __DIR__ . "/../src/helpers.php";
require __DIR__ . "/../src/blogs-routes.php"($app, $pdo);
require __DIR__ . "/../src/admin-routes.php"($app, $pdo);

$app->run();
