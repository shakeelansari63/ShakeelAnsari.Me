<?php

use Psr\Http\Message\ResponseInterface as Response;
use Psr\Http\Message\ServerRequestInterface as Request;
use Slim\App;

return function (App $app, ?PDO $pdo) {
    $app->get("/blogs", function (Request $request, Response $response) use (
        $pdo,
    ) {
        if (!$pdo) {
            return jsonResponse(
                $response,
                ["error" => "Database not available"],
                503,
            );
        }

        $params = $request->getQueryParams();
        $page = max(1, (int) ($params["page"] ?? 1));
        $limit = max(1, min(50, (int) ($params["limit"] ?? 5)));

        try {
            $countStmt = $pdo->query(
                "SELECT COUNT(*) FROM blog WHERE deleted = 0",
            );
            $total = (int) $countStmt->fetchColumn();
        } catch (PDOException $e) {
            return jsonResponse($response, ["error" => "Database error"], 500);
        }

        $blogs = [];
        if ($total > 0) {
            $offset = ($page - 1) * $limit;
            $stmt = $pdo->prepare(
                'SELECT id, title, excerpt, date, read_time, tags, views, likes
                 FROM blog
                 WHERE deleted = 0
                 ORDER BY date DESC, id ASC
                 LIMIT ? OFFSET ?',
            );
            $stmt->execute([$limit, $offset]);

            foreach ($stmt as $row) {
                $blogs[] = [
                    "id" => $row["id"],
                    "title" => $row["title"],
                    "excerpt" => $row["excerpt"],
                    "date" => $row["date"],
                    "readTime" => $row["read_time"],
                    "tags" => json_decode($row["tags"] ?? "[]", true),
                    "views" => (int) $row["views"],
                    "likes" => (int) $row["likes"],
                ];
            }
        }

        $totalPages = $total > 0 ? (int) ceil($total / $limit) : 1;

        return jsonResponse($response, [
            "data" => $blogs,
            "page" => $page,
            "limit" => $limit,
            "total" => $total,
            "totalPages" => $totalPages,
        ]);
    });

    $app->get("/blogs/{id}/content", function (
        Request $request,
        Response $response,
        array $args,
    ) use ($pdo) {
        $id = basename($args["id"]);
        $file = BLOGS_DIR . "/" . $id . ".md";

        if ($pdo) {
            $stmt = $pdo->prepare("SELECT md_file FROM blog WHERE id = ?");
            $stmt->execute([$id]);
            $row = $stmt->fetch();
            if ($row && !empty($row["md_file"])) {
                $file = BLOGS_DIR . "/" . $row["md_file"];
            }
        }

        if (!file_exists($file)) {
            return jsonResponse($response, ["error" => "Blog not found"], 404);
        }

        $raw = file_get_contents($file);
        $meta = parseFrontmatter($raw);

        return jsonResponse($response, [
            "content" => rewriteImageUrls($meta["content"]),
        ]);
    });

    $app->get("/blogs/{id}", function (
        Request $request,
        Response $response,
        array $args,
    ) use ($pdo) {
        $id = basename($args["id"]);

        if (!$pdo) {
            return jsonResponse(
                $response,
                ["error" => "Database not available"],
                503,
            );
        }

        $stmt = $pdo->prepare(
            "SELECT id, title, excerpt, date, read_time, tags, views, likes FROM blog WHERE id = ? AND deleted = 0",
        );
        $stmt->execute([$id]);
        $row = $stmt->fetch();

        if (!$row) {
            return jsonResponse($response, ["error" => "Blog not found"], 404);
        }

        return jsonResponse($response, [
            "id" => $row["id"],
            "title" => $row["title"],
            "excerpt" => $row["excerpt"],
            "date" => $row["date"],
            "readTime" => $row["read_time"],
            "tags" => json_decode($row["tags"] ?? "[]", true),
            "views" => (int) $row["views"],
            "likes" => (int) $row["likes"],
        ]);
    });

    $app->get("/blogs/images/{name}", function (
        Request $request,
        Response $response,
        array $args,
    ) {
        $name = basename($args["name"]);
        $file = BLOGS_DIR . "/images/" . $name;

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

    $app->get("/blogs/{id}/stats", function (
        Request $request,
        Response $response,
        array $args,
    ) use ($pdo) {
        if (!$pdo) {
            return jsonResponse(
                $response,
                ["error" => "Database not available"],
                503,
            );
        }
        $id = basename($args["id"]);
        $ip = clientIp($request);

        $stmt = $pdo->prepare("SELECT views, likes FROM blog WHERE id = ?");
        $stmt->execute([$id]);
        $row = $stmt->fetch();

        $liked = false;
        if ($row) {
            $check = $pdo->prepare(
                "SELECT 1 FROM blog_likes WHERE blog_id = ? AND ip = ?",
            );
            $check->execute([$id, $ip]);
            $liked = (bool) $check->fetchColumn();
        }

        return jsonResponse(
            $response,
            ($row ?: ["views" => 0, "likes" => 0]) + ["liked" => $liked],
        );
    });

    $app->post("/blogs/{id}/view", function (
        Request $request,
        Response $response,
        array $args,
    ) use ($pdo) {
        if (!$pdo) {
            return jsonResponse(
                $response,
                ["error" => "Database not available"],
                503,
            );
        }
        $id = basename($args["id"]);
        $ip = clientIp($request);
        $windowStart = floor((int) date("G") / 8) * 8;
        $hour = date("Y-m-d {$windowStart}:00:00");

        $seen = $pdo->prepare(
            "SELECT 1 FROM blog_views WHERE blog_id = ? AND ip = ? AND view_hour = ?",
        );
        $seen->execute([$id, $ip, $hour]);
        $alreadyViewed = (bool) $seen->fetchColumn();

        if (!$alreadyViewed) {
            $pdo->prepare("INSERT IGNORE INTO blog (id) VALUES (?)")->execute([
                $id,
            ]);
            $pdo->prepare(
                "INSERT IGNORE INTO blog_views (blog_id, ip, view_hour) VALUES (?, ?, ?)",
            )->execute([$id, $ip, $hour]);

            $cnt = $pdo->prepare(
                "SELECT COUNT(*) FROM blog_views WHERE blog_id = ?",
            );
            $cnt->execute([$id]);
            $pdo->prepare("UPDATE blog SET views = ? WHERE id = ?")->execute([
                (int) $cnt->fetchColumn(),
                $id,
            ]);
        }

        $stmt = $pdo->prepare("SELECT views FROM blog WHERE id = ?");
        $stmt->execute([$id]);
        return jsonResponse($response, $stmt->fetch());
    });

    $app->post("/blogs/{id}/like", function (
        Request $request,
        Response $response,
        array $args,
    ) use ($pdo) {
        if (!$pdo) {
            return jsonResponse(
                $response,
                ["error" => "Database not available"],
                503,
            );
        }
        $id = basename($args["id"]);
        $ip = clientIp($request);

        $pdo->beginTransaction();

        $pdo->prepare("INSERT IGNORE INTO blog (id) VALUES (?)")->execute([
            $id,
        ]);

        $check = $pdo->prepare(
            "SELECT 1 FROM blog_likes WHERE blog_id = ? AND ip = ?",
        );
        $check->execute([$id, $ip]);
        $liked = (bool) $check->fetchColumn();

        if ($liked) {
            $pdo->prepare(
                "DELETE FROM blog_likes WHERE blog_id = ? AND ip = ?",
            )->execute([$id, $ip]);
            $liked = false;
        } else {
            $pdo->prepare(
                "INSERT INTO blog_likes (blog_id, ip) VALUES (?, ?)",
            )->execute([$id, $ip]);
            $liked = true;
        }

        $cnt = $pdo->prepare(
            "SELECT COUNT(*) FROM blog_likes WHERE blog_id = ?",
        );
        $cnt->execute([$id]);
        $pdo->prepare("UPDATE blog SET likes = ? WHERE id = ?")->execute([
            (int) $cnt->fetchColumn(),
            $id,
        ]);

        $pdo->commit();

        $stmt = $pdo->prepare("SELECT likes FROM blog WHERE id = ?");
        $stmt->execute([$id]);
        $likes = (int) $stmt->fetchColumn();

        return jsonResponse($response, ["likes" => $likes, "liked" => $liked]);
    });
};
