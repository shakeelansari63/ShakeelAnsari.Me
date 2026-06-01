<?php

use Psr\Http\Message\ResponseInterface as Response;
use Psr\Http\Message\ServerRequestInterface as Request;
use Slim\App;
use Firebase\JWT\JWT;
use Firebase\JWT\Key;

function verifyAdmin(Request $request): bool
{
    $header = $request->getHeaderLine("Authorization");
    if (!str_starts_with($header, "Bearer ")) {
        return false;
    }
    try {
        JWT::decode(
            substr($header, 7),
            new Key($_ENV["JWT_SECRET"] ?? "", "HS256"),
        );
        return true;
    } catch (\Exception $e) {
        return false;
    }
}

return function (App $app, ?PDO $pdo) {
    $app->post("/admin/login", function (Request $request, Response $response) {
        $ip = clientIp($request);
        if (!checkRateLimit($ip)) {
            return jsonResponse(
                $response,
                ["error" => "Too many attempts"],
                429,
            );
        }

        $body = $request->getParsedBody();
        $username = trim($body["username"] ?? "");
        $password = $body["password"] ?? "";

        $adminUser = $_ENV["ADMIN_USERNAME"] ?? "";
        $adminPass = $_ENV["ADMIN_PASSWORD"] ?? "";
        $jwtSecret = $_ENV["JWT_SECRET"] ?? "";

        if (
            !hash_equals($adminUser ?? "", $username) ||
            !hash_equals($adminPass ?? "", $password)
        ) {
            return jsonResponse(
                $response,
                ["error" => "Invalid credentials"],
                401,
            );
        }

        $token = JWT::encode(
            [
                "sub" => $username,
                "iat" => time(),
                "exp" => time() + 86400,
            ],
            $jwtSecret,
            "HS256",
        );

        return jsonResponse($response, ["token" => $token]);
    });

    $app->post("/admin/sync-blogs", function (
        Request $request,
        Response $response,
    ) use ($pdo) {
        if (!$pdo) {
            return jsonResponse(
                $response,
                ["error" => "Database not available"],
                503,
            );
        }

        if (!verifyAdmin($request)) {
            return jsonResponse($response, ["error" => "Unauthorized"], 401);
        }

        $dir = BLOGS_DIR;
        $files = glob($dir . "/*.md");

        if (empty($files)) {
            return jsonResponse(
                $response,
                ["error" => "No markdown files found"],
                500,
            );
        }

        $upsert = $pdo->prepare(
            'INSERT INTO blog (id, title, excerpt, date, read_time, tags, md_file, deleted)
             VALUES (?, ?, ?, ?, ?, ?, ?, 0)
             ON DUPLICATE KEY UPDATE
                 title = VALUES(title),
                 excerpt = VALUES(excerpt),
                 date = VALUES(date),
                 read_time = VALUES(read_time),
                 tags = VALUES(tags),
                 md_file = VALUES(md_file),
                 deleted = 0',
        );

        $ids = [];
        $parsed = 0;
        foreach ($files as $file) {
            $id = pathinfo($file, PATHINFO_FILENAME);
            $ids[] = $id;
            $raw = file_get_contents($file);
            $meta = parseFrontmatter($raw);

            $upsert->execute([
                $id,
                $meta["title"],
                $meta["excerpt"],
                $meta["date"],
                $meta["readTime"],
                json_encode($meta["tags"]),
                $file,
            ]);
            $parsed++;
        }

        if ($ids) {
            $placeholders = implode(",", array_fill(0, count($ids), "?"));
            $pdo->prepare(
                "UPDATE blog SET deleted = 1 WHERE id NOT IN ({$placeholders})",
            )->execute($ids);
        } else {
            $pdo->exec("UPDATE blog SET deleted = 1");
        }

        return jsonResponse($response, [
            "message" => "Synced {$parsed} blog posts",
        ]);
    });
};
