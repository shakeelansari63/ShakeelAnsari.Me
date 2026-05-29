<?php

use Psr\Http\Message\ResponseInterface as Response;
use Psr\Http\Message\ServerRequestInterface as Request;
use Slim\App;

return function (App $app) {
  $app->get('/hello', function (Request $request, Response $response) {
    $payload = json_encode(['message' => 'Hello from the API!']);
    $response->getBody()->write($payload);
    return $response->withHeader('Content-Type', 'application/json');
  });
};
