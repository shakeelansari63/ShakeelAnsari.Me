<?php

use Slim\Factory\AppFactory;
use Slim\Middleware\BodyParsingMiddleware;
use Dotenv\Dotenv;
use Psr\Http\Message\ResponseInterface as Response;
use Psr\Http\Message\ServerRequestInterface as Request;

require __DIR__ . '/../vendor/autoload.php';

$dotenv = Dotenv::createImmutable(__DIR__ . '/..');
$dotenv->safeLoad();

$app = AppFactory::create();

$app->addErrorMiddleware(true, true, true);

$app->add(new BodyParsingMiddleware());

$app->setBasePath('/api');

$app->add(function (Request $request, $handler) {
    if ($request->getMethod() === 'OPTIONS') {
        $response = new \Slim\Psr7\Response();
    } else {
        $response = $handler->handle($request);
    }

    return $response
        ->withHeader('Access-Control-Allow-Origin', '*')
        ->withHeader('Access-Control-Allow-Headers', 'Content-Type, Accept, Authorization')
        ->withHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
});

(require __DIR__ . '/../src/routes.php')($app);

$app->run();
