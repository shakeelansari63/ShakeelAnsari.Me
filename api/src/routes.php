<?php

use Psr\Http\Message\ResponseInterface as Response;
use Psr\Http\Message\ServerRequestInterface as Request;
use Psr\Http\Message\ResponseInterface;
use Slim\App;
use Slim\Routing\RouteCollectorProxy;
use Firebase\JWT\JWT;

const BLOGS_DIR = __DIR__ . '/../../blogs';

function parseFrontmatter(string $content): array {
    $meta = [
        'title' => '',
        'excerpt' => '',
        'date' => '',
        'readTime' => '',
        'tags' => [],
        'content' => $content,
    ];

    if (preg_match('/^---\s*\n(.*?)\n---\s*\n(.*)/s', $content, $matches)) {
        $frontmatter = $matches[1];
        $meta['content'] = trim($matches[2]);

        foreach (explode("\n", $frontmatter) as $line) {
            if (preg_match('/^(\w+):\s*(.+)$/', trim($line), $fm)) {
                $key = $fm[1];
                $val = trim($fm[2]);
                if ($key === 'tags') {
                    $meta['tags'] = array_map('trim', explode(',', $val));
                } else {
                    $meta[$key] = $val;
                }
            }
        }
    }

    return $meta;
}

function rewriteImageUrls(string $content): string {
    return preg_replace('/\]\((images\/)/', '](/api/blogs/$1', $content);
}

function checkRateLimit(string $ip): bool {
    $file = sys_get_temp_dir() . '/login_attempts_' . md5($ip) . '.lock';

    if (file_exists($file) && time() - filemtime($file) > 900) {
        unlink($file);
    }

    $attempts = @file_get_contents($file) ?: 0;
    $attempts = (int) $attempts;

    if ($attempts >= 5) {
        return false;
    }

    file_put_contents($file, $attempts + 1);
    return true;
}

function clientIp(Request $request): string {
    $forwarded = $request->getHeaderLine('X-Forwarded-For');
    if ($forwarded !== '') {
        $ips = array_map('trim', explode(',', $forwarded));
        return $ips[0];
    }
    return $request->getServerParams()['REMOTE_ADDR'] ?? '127.0.0.1';
}

function jsonResponse(ResponseInterface $response, array $data, int $status = 200): ResponseInterface {
    $payload = json_encode($data, JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE);
    $response->getBody()->write($payload);
    return $response
        ->withHeader('Content-Type', 'application/json')
        ->withStatus($status);
}

return function (App $app, ?PDO $pdo) {
    $app->get('/blogs', function (Request $request, Response $response) {
        $dir = BLOGS_DIR;
        if (!is_dir($dir)) {
            return jsonResponse($response, ['error' => 'Blogs directory not found'], 500);
        }

        $params = $request->getQueryParams();
        $page = max(1, (int)($params['page'] ?? 1));
        $limit = max(1, min(50, (int)($params['limit'] ?? 5)));

        $files = glob($dir . '/*.md');
        $blogs = [];

        foreach ($files as $file) {
            $id = pathinfo($file, PATHINFO_FILENAME);
            $raw = file_get_contents($file);
            $meta = parseFrontmatter($raw);

            $blogs[] = [
                'id' => $id,
                'title' => $meta['title'],
                'excerpt' => $meta['excerpt'],
                'date' => $meta['date'],
                'readTime' => $meta['readTime'],
                'tags' => $meta['tags'],
            ];
        }

        usort($blogs, fn($a, $b) => $b['date'] <=> $a['date'] ?: $a['id'] <=> $b['id']);

        $total = count($blogs);
        $totalPages = (int)ceil($total / $limit);
        $offset = ($page - 1) * $limit;
        $data = array_slice($blogs, $offset, $limit);

        return jsonResponse($response, [
            'data' => $data,
            'page' => $page,
            'limit' => $limit,
            'total' => $total,
            'totalPages' => $totalPages,
        ]);
    });

    $app->get('/blogs/{id}', function (Request $request, Response $response, array $args) {
        $id = basename($args['id']);
        $file = BLOGS_DIR . '/' . $id . '.md';

        if (!file_exists($file)) {
            return jsonResponse($response, ['error' => 'Blog not found'], 404);
        }

        $raw = file_get_contents($file);
        $meta = parseFrontmatter($raw);

        $blog = [
            'id' => $id,
            'title' => $meta['title'],
            'excerpt' => $meta['excerpt'],
            'content' => rewriteImageUrls($meta['content']),
            'date' => $meta['date'],
            'readTime' => $meta['readTime'],
            'tags' => $meta['tags'],
        ];

        return jsonResponse($response, $blog);
    });

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

    $app->get('/blogs/images/{name}', function (Request $request, Response $response, array $args) {
        $name = basename($args['name']);
        $file = BLOGS_DIR . '/images/' . $name;

        if (!file_exists($file)) {
            return jsonResponse($response, ['error' => 'Image not found'], 404);
        }

        $ext = strtolower(pathinfo($file, PATHINFO_EXTENSION));
        $mimeTypes = [
            'jpg' => 'image/jpeg',
            'jpeg' => 'image/jpeg',
            'png' => 'image/png',
            'gif' => 'image/gif',
            'svg' => 'image/svg+xml',
            'webp' => 'image/webp',
        ];
        $mime = $mimeTypes[$ext] ?? 'application/octet-stream';

        $response->getBody()->write(file_get_contents($file));
        return $response->withHeader('Content-Type', $mime);
    });

    $app->get('/blogs/{id}/stats', function (Request $request, Response $response, array $args) use ($pdo) {
        if (!$pdo) {
            return jsonResponse($response, ['error' => 'Database not available'], 503);
        }
        $id = basename($args['id']);
        $ip = clientIp($request);

        $stmt = $pdo->prepare('SELECT views, likes FROM blog_stats WHERE id = ?');
        $stmt->execute([$id]);
        $row = $stmt->fetch();

        $liked = false;
        if ($row) {
            $check = $pdo->prepare('SELECT 1 FROM blog_likes WHERE blog_id = ? AND ip = ?');
            $check->execute([$id, $ip]);
            $liked = (bool) $check->fetchColumn();
        }

        return jsonResponse($response, ($row ?: ['views' => 0, 'likes' => 0]) + ['liked' => $liked]);
    });

    $app->post('/blogs/{id}/view', function (Request $request, Response $response, array $args) use ($pdo) {
        if (!$pdo) {
            return jsonResponse($response, ['error' => 'Database not available'], 503);
        }
        $id = basename($args['id']);
        $pdo->prepare('INSERT INTO blog_stats (id, views) VALUES (?, 1) ON DUPLICATE KEY UPDATE views = views + 1')->execute([$id]);

        $stmt = $pdo->prepare('SELECT views FROM blog_stats WHERE id = ?');
        $stmt->execute([$id]);
        return jsonResponse($response, $stmt->fetch());
    });

    $app->post('/blogs/{id}/like', function (Request $request, Response $response, array $args) use ($pdo) {
        if (!$pdo) {
            return jsonResponse($response, ['error' => 'Database not available'], 503);
        }
        $id = basename($args['id']);
        $ip = clientIp($request);

        $pdo->beginTransaction();

        $pdo->prepare('INSERT IGNORE INTO blog_stats (id) VALUES (?)')->execute([$id]);

        $check = $pdo->prepare('SELECT 1 FROM blog_likes WHERE blog_id = ? AND ip = ?');
        $check->execute([$id, $ip]);
        $liked = (bool) $check->fetchColumn();

        if ($liked) {
            $pdo->prepare('DELETE FROM blog_likes WHERE blog_id = ? AND ip = ?')->execute([$id, $ip]);
            $pdo->prepare('UPDATE blog_stats SET likes = likes - 1 WHERE id = ?')->execute([$id]);
            $liked = false;
        } else {
            $pdo->prepare('INSERT INTO blog_likes (blog_id, ip) VALUES (?, ?)')->execute([$id, $ip]);
            $pdo->prepare('UPDATE blog_stats SET likes = likes + 1 WHERE id = ?')->execute([$id]);
            $liked = true;
        }

        $pdo->commit();

        $stmt = $pdo->prepare('SELECT likes FROM blog_stats WHERE id = ?');
        $stmt->execute([$id]);
        $likes = (int) $stmt->fetchColumn();

        return jsonResponse($response, ['likes' => $likes, 'liked' => $liked]);
    });
};
