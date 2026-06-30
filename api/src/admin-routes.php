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
        if (isRateLimited($ip)) {
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
            !hash_equals($adminUser, $username) ||
            !hash_equals($adminPass, $password)
        ) {
            incrementAttempts($ip);
            return jsonResponse(
                $response,
                ["error" => "Invalid credentials"],
                401,
            );
        }

        resetAttempts($ip);

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
            'INSERT INTO blog (id, title, excerpt, date, read_time, tags, banner_image, md_file, deleted)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
             ON DUPLICATE KEY UPDATE
                 title = VALUES(title),
                 excerpt = VALUES(excerpt),
                 date = VALUES(date),
                 read_time = VALUES(read_time),
                 tags = VALUES(tags),
                 banner_image = VALUES(banner_image),
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
                $meta["bannerImage"] ?? "",
                basename($file),
            ]);
            $parsed++;
        }

        $placeholders = implode(",", array_fill(0, count($ids), "?"));
        $pdo->prepare(
            "UPDATE blog SET deleted = 1 WHERE id NOT IN ({$placeholders})",
        )->execute($ids);

        return jsonResponse($response, [
            "message" => "Synced {$parsed} blog posts",
        ]);
    });

    $app->post("/admin/sync-learn", function (
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

        $dir = LEARN_DIR;

        if (!is_dir($dir)) {
            return jsonResponse(
                $response,
                ["error" => "Learn directory not found"],
                500,
            );
        }

        $subjectFolders = glob($dir . "/*", GLOB_ONLYDIR);
        $syncedSubjects = 0;
        $syncedChapters = 0;

        $catalogExts = ['png', 'jpg', 'jpeg', 'webp', 'bmp'];

        $upsertSubject = $pdo->prepare(
            'INSERT INTO learn_subjects (id, title, folder, sort_order, thumbnail)
             VALUES (?, ?, ?, ?, ?)
             ON DUPLICATE KEY UPDATE
                 title = VALUES(title),
                 folder = VALUES(folder),
                 sort_order = VALUES(sort_order),
                 thumbnail = VALUES(thumbnail)',
        );

        $upsertChapter = $pdo->prepare(
            'INSERT INTO learn_chapters (subject_id, chapter_id, title, md_file, sort_order)
             VALUES (?, ?, ?, ?, ?)
             ON DUPLICATE KEY UPDATE
                 title = VALUES(title),
                 md_file = VALUES(md_file),
                 sort_order = VALUES(sort_order)',
        );

        $subjectIds = [];

        foreach ($subjectFolders as $folderPath) {
            $folder = basename($folderPath);

            if (preg_match('/^(\d+)-(.*)$/', $folder, $m)) {
                $sortOrder = (int) $m[1];
                $title = trim(str_replace(['-', '_'], ' ', $m[2]));
                $id = strtolower($folder);
            } else {
                $sortOrder = 0;
                $title = trim(str_replace(['-', '_'], ' ', $folder));
                $id = strtolower($folder);
            }

            $thumbnail = '_default_thumbnail.svg';
            $imagesDir = $folderPath . '/images';
            if (is_dir($imagesDir)) {
                foreach ($catalogExts as $ext) {
                    $candidate = $imagesDir . '/thumbnail.' . $ext;
                    if (file_exists($candidate)) {
                        $thumbnail = 'thumbnail.' . $ext;
                        break;
                    }
                }
            }

            $subjectIds[] = $id;
            $upsertSubject->execute([$id, $title, $folder, $sortOrder, $thumbnail]);
            $syncedSubjects++;

            $chapterFiles = glob($folderPath . "/*.md");
            foreach ($chapterFiles as $filePath) {
                $chapterId = pathinfo($filePath, PATHINFO_FILENAME);

                if (preg_match('/^ch(\d+)-(.*)$/', $chapterId, $cm)) {
                    $chSort = (int) $cm[1];
                    $chTitle = trim(str_replace(['-', '_'], ' ', $cm[2]));
                } else {
                    $chSort = 0;
                    $chTitle = trim(str_replace(['-', '_'], ' ', $chapterId));
                }

                $upsertChapter->execute([
                    $id,
                    $chapterId,
                    ucwords($chTitle),
                    basename($filePath),
                    $chSort,
                ]);
                $syncedChapters++;
            }
        }

        if (!empty($subjectIds)) {
            $placeholders = implode(",", array_fill(0, count($subjectIds), "?"));
            $pdo->prepare(
                "DELETE FROM learn_chapters WHERE subject_id NOT IN ({$placeholders})",
            )->execute($subjectIds);
            $pdo->prepare(
                "DELETE FROM learn_subjects WHERE id NOT IN ({$placeholders})",
            )->execute($subjectIds);
        }

        return jsonResponse($response, [
            "message" => "Synced {$syncedSubjects} subjects and {$syncedChapters} chapters",
        ]);
    });

    $app->post("/admin/analytics", function (
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

        try {
            $viewsByDate = $pdo->query(
                "SELECT DATE(view_hour) AS date, COUNT(*) AS count FROM blog_views GROUP BY DATE(view_hour) ORDER BY date ASC",
            )->fetchAll();

            $likesByDate = $pdo->query(
                "SELECT DATE(created_at) AS date, COUNT(*) AS count FROM blog_likes GROUP BY DATE(created_at) ORDER BY date ASC",
            )->fetchAll();

            $topBlogs = $pdo->query(
                "SELECT b.id, b.title, b.views, b.likes FROM blog b WHERE b.deleted = 0 ORDER BY b.views DESC LIMIT 10",
            )->fetchAll();

            $viewIps = $pdo->query(
                "SELECT DISTINCT ip FROM blog_views",
            )->fetchAll(PDO::FETCH_COLUMN);

            $likeIps = $pdo->query(
                "SELECT DISTINCT ip FROM blog_likes",
            )->fetchAll(PDO::FETCH_COLUMN);

            $allIps = array_unique(array_merge($viewIps, $likeIps));
            $countryViewCounts = [];
            $countryLikeCounts = [];

            foreach ($allIps as $ip) {
                $loc = resolveLocation($ip, $pdo);
                $key = $loc->country_code . "|" . $loc->country;
                if (!isset($countryViewCounts[$key])) {
                    $countryViewCounts[$key] = ["code" => $loc->country_code, "country" => $loc->country, "count" => 0];
                    $countryLikeCounts[$key] = ["code" => $loc->country_code, "country" => $loc->country, "count" => 0];
                }
            }

            foreach ($viewIps as $ip) {
                $loc = resolveLocation($ip, $pdo);
                $key = $loc->country_code . "|" . $loc->country;
                $countryViewCounts[$key]["count"]++;
            }

            foreach ($likeIps as $ip) {
                $loc = resolveLocation($ip, $pdo);
                $key = $loc->country_code . "|" . $loc->country;
                $countryLikeCounts[$key]["count"]++;
            }

            $totalViews = $pdo->query("SELECT COUNT(*) FROM blog_views")->fetchColumn();
            $totalLikes = $pdo->query("SELECT COUNT(*) FROM blog_likes")->fetchColumn();
            $uniqueVisitors = $pdo->query("SELECT COUNT(DISTINCT ip) FROM blog_views")->fetchColumn();

            return jsonResponse($response, [
                "viewsByDate" => $viewsByDate,
                "likesByDate" => $likesByDate,
                "viewsByCountry" => array_values($countryViewCounts),
                "likesByCountry" => array_values($countryLikeCounts),
                "topBlogs" => $topBlogs,
                "totalViews" => (int) $totalViews,
                "totalLikes" => (int) $totalLikes,
                "uniqueVisitors" => (int) $uniqueVisitors,
            ]);
        } catch (PDOException $e) {
            return jsonResponse($response, ["error" => "Database error"], 500);
        }
    });
};
