<?php

use Psr\Http\Message\ResponseInterface as Response;
use Psr\Http\Message\ServerRequestInterface as Request;
use Slim\App;

return function (App $app, ?PDO $pdo) {
    $app->get("/learn/subjects", function (
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

        try {
            $stmt = $pdo->query(
                "SELECT id, title, folder, sort_order, thumbnail
                 FROM learn_subjects
                 ORDER BY sort_order ASC, id ASC",
            );
            $subjects = $stmt->fetchAll();
            return jsonResponse($response, ["data" => $subjects]);
        } catch (PDOException $e) {
            return jsonResponse($response, ["error" => "Database error"], 500);
        }
    });

    $app->get("/learn/subjects/{id}/chapters", function (
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

        $subjectId = basename($args["id"]);

        try {
            $stmt = $pdo->prepare(
                "SELECT id, chapter_id, title, sort_order
                 FROM learn_chapters
                 WHERE subject_id = ?
                 ORDER BY sort_order ASC, id ASC",
            );
            $stmt->execute([$subjectId]);
            $chapters = $stmt->fetchAll();
            return jsonResponse($response, ["data" => $chapters]);
        } catch (PDOException $e) {
            return jsonResponse($response, ["error" => "Database error"], 500);
        }
    });

    $app->get("/learn/chapters/{id}/content", function (
        Request $request,
        Response $response,
        array $args,
    ) use ($pdo) {
        $chapterId = (int) $args["id"];

        if (!$pdo) {
            return jsonResponse(
                $response,
                ["error" => "Database not available"],
                503,
            );
        }

        try {
            $stmt = $pdo->prepare(
                "SELECT c.md_file, c.title, s.folder
                 FROM learn_chapters c
                 JOIN learn_subjects s ON s.id = c.subject_id
                 WHERE c.id = ?",
            );
            $stmt->execute([$chapterId]);
            $row = $stmt->fetch();

            if (!$row) {
                return jsonResponse(
                    $response,
                    ["error" => "Chapter not found"],
                    404,
                );
            }

            $file = LEARN_DIR . "/" . $row["folder"] . "/" . $row["md_file"];

            if (!file_exists($file)) {
                return jsonResponse(
                    $response,
                    ["error" => "Chapter file not found"],
                    404,
                );
            }

            $content = file_get_contents($file);

            return jsonResponse($response, [
                "title" => $row["title"],
                "content" => $content,
            ]);
        } catch (PDOException $e) {
            return jsonResponse($response, ["error" => "Database error"], 500);
        }
    });

    $app->get("/learn/images/{folder}/{name}", function (
        Request $request,
        Response $response,
        array $args,
    ) {
        $folder = basename($args["folder"]);
        $name = basename($args["name"]);
        $file = LEARN_DIR . "/" . $folder . "/" . $name;

        if (!file_exists($file)) {
            $file = LEARN_DIR . "/_default_thumbnail.svg";
        }

        $ext = strtolower(pathinfo($file, PATHINFO_EXTENSION));
        $mimeTypes = [
            "jpg" => "image/jpeg",
            "jpeg" => "image/jpeg",
            "png" => "image/png",
            "gif" => "image/gif",
            "svg" => "image/svg+xml",
            "webp" => "image/webp",
            "bmp" => "image/bmp",
        ];
        $mime = $mimeTypes[$ext] ?? "application/octet-stream";

        $response->getBody()->write(file_get_contents($file));
        return $response->withHeader("Content-Type", $mime);
    });
};
