<?php

use Psr\Http\Message\ResponseInterface as Response;
use Psr\Http\Message\ServerRequestInterface as Request;
use Slim\App;
use Firebase\JWT\JWT;

return function (App $app, ?PDO $pdo) {
    $app->post('/admin/login', function (Request $request, Response $response) {
        $ip = clientIp($request);
        if (!checkRateLimit($ip)) {
            return jsonResponse($response, ['error' => 'Too many attempts'], 429);
        }

        $body = $request->getParsedBody();
        $username = trim($body['username'] ?? '');
        $password = $body['password'] ?? '';

        $adminUser = $_ENV['ADMIN_USERNAME'] ?? '';
        $adminPass = $_ENV['ADMIN_PASSWORD'] ?? '';
        $jwtSecret = $_ENV['JWT_SECRET'] ?? '';

        if (!hash_equals($adminUser ?? '', $username) || !hash_equals($adminPass ?? '', $password)) {
            return jsonResponse($response, ['error' => 'Invalid credentials'], 401);
        }

        $token = JWT::encode([
            'sub' => $username,
            'iat' => time(),
            'exp' => time() + 86400,
        ], $jwtSecret, 'HS256');

        return jsonResponse($response, ['token' => $token]);
    });
};
