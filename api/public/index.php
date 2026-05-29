<?php

use Slim\Factory\AppFactory;
use Dotenv\Dotenv;

require __DIR__ . '/../vendor/autoload.php';

$dotenv = Dotenv::createImmutable(__DIR__ . '/..');
$dotenv->safeLoad();

$app = AppFactory::create();

$app->addErrorMiddleware(true, true, true);

$app->setBasePath('/api');

(require __DIR__ . '/../src/routes.php')($app);

$app->run();
