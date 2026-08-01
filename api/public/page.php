<?php

ini_set("max_execution_time", 60);

require_once __DIR__ . "/../vendor/autoload.php";
require_once __DIR__ . "/../src/helpers.php";
require_once __DIR__ . "/../src/seo-helpers.php";

$dotenv = Dotenv\Dotenv::createMutable(__DIR__ . "/..");
$dotenv->safeLoad();

$seo = seoDefaults();
$pdo = \App\DB::connect();

$path = parse_url($_SERVER["REQUEST_URI"] ?? "/", PHP_URL_PATH) ?: "/";
$segments = array_values(
    array_filter(explode("/", trim($path, "/")), fn (string $s) => $s !== ""),
);

$route = null;

if (count($segments) === 0) {
    $route = ["type" => "home"];
} elseif (count($segments) === 1) {
    switch ($segments[0]) {
        case "blog":
            $route = ["type" => "blog-list"];
            break;
        case "expo":
            $route = ["type" => "expo"];
            break;
        case "learn":
            $route = ["type" => "learn"];
            break;
        case "stats":
            $route = ["type" => "stats"];
            break;
        case "admin":
            $route = ["type" => "admin"];
            break;
        default:
            $route = ["type" => "404"];
    }
} elseif (count($segments) === 2 && $segments[0] === "blog") {
    $id = basename(rawurldecode($segments[1]));
    if (validateId($id)) {
        $route = ["type" => "blog", "id" => $id];
    }
} elseif (count($segments) === 2 && $segments[0] === "product") {
    $id = basename(rawurldecode($segments[1]));
    if (validateId($id, '/^[a-zA-Z0-9_-]+$/')) {
        $route = ["type" => "product", "id" => $id];
    }
} elseif (count($segments) === 2 && $segments[0] === "learn") {
    $id = basename(rawurldecode($segments[1]));
    if (validateId($id)) {
        $route = ["type" => "subject", "id" => $id];
    }
} elseif (count($segments) === 3 && $segments[0] === "learn") {
    $subjectId = basename(rawurldecode($segments[1]));
    $chapterId = basename(rawurldecode($segments[2]));
    if (validateId($subjectId) && validateId($chapterId)) {
        $route = [
            "type" => "chapter",
            "subject" => $subjectId,
            "chapter" => $chapterId,
        ];
    }
}

if ($route === null) {
    $route = ["type" => "404"];
}

function blogSeoData(?PDO $pdo, string $id): ?array
{
    if ($pdo) {
        try {
            $stmt = $pdo->prepare(
                "SELECT title, excerpt, date, banner_image FROM blog WHERE id = ? AND deleted = 0",
            );
            $stmt->execute([$id]);
            $row = $stmt->fetch();
            if ($row) {
                $image = $row["banner_image"] ?? "";
                if ($image !== "" && strpos($image, "://") === false) {
                    $image = seoDomain() . "/api/blogs/images/" . rawurlencode($image);
                }
                return [
                    "title" => $row["title"] ?? "",
                    "excerpt" => $row["excerpt"] ?? "",
                    "date" => $row["date"] ?? "",
                    "image" => $image,
                ];
            }
        } catch (PDOException) {
        }
    }

    $file = BLOGS_DIR . "/" . $id . ".md";
    if (!file_exists($file)) {
        return null;
    }
    $meta = parseFrontmatter(file_get_contents($file));
    $image = $meta["bannerImage"] ?? "";
    if ($image !== "") {
        $image = seoDomain() . "/api/blogs/images/" . rawurlencode($image);
    }
    return [
        "title" => $meta["title"] ?? "",
        "excerpt" => $meta["excerpt"] ?? "",
        "date" => $meta["date"] ?? "",
        "image" => $image,
    ];
}

function subjectSeoData(?PDO $pdo, string $id): ?array
{
    $title = "";
    $folder = null;

    if ($pdo) {
        try {
            $stmt = $pdo->prepare(
                "SELECT title, folder FROM learn_subjects WHERE id = ?",
            );
            $stmt->execute([$id]);
            $row = $stmt->fetch();
            if (!$row) {
                return null;
            }
            $title = $row["title"];
            $folder = $row["folder"];
        } catch (PDOException) {
            return null;
        }
    }

    if ($folder === null) {
        foreach (glob(LEARN_DIR . "/*", GLOB_ONLYDIR) as $dir) {
            if (strtolower(basename($dir)) === strtolower($id)) {
                $folder = basename($dir);
                break;
            }
        }
        if ($folder === null) {
            return null;
        }
    }

    $files = glob(LEARN_DIR . "/" . $folder . "/*.md");
    sort($files);
    $excerpt = "";
    if ($files) {
        $excerpt = markdownExcerpt(file_get_contents($files[0]));
    }
    if ($title === "") {
        $title = ucwords(str_replace(["-", "_"], " ", $id));
    }

    return ["title" => $title, "excerpt" => $excerpt];
}

function chapterSeoData(?PDO $pdo, string $subjectId, string $chapterId): ?array
{
    $title = "";
    $file = null;

    if ($pdo) {
        try {
            $stmt = $pdo->prepare(
                "SELECT c.title, c.md_file, s.folder
                 FROM learn_chapters c
                 JOIN learn_subjects s ON s.id = c.subject_id
                 WHERE c.subject_id = ? AND c.chapter_id = ?",
            );
            $stmt->execute([$subjectId, $chapterId]);
            $row = $stmt->fetch();
            if ($row) {
                $title = $row["title"] ?? "";
                $file = LEARN_DIR . "/" . $row["folder"] . "/" . $row["md_file"];
            }
        } catch (PDOException) {
        }
    }

    if ($file === null || !file_exists($file)) {
        $folder = null;
        foreach (glob(LEARN_DIR . "/*", GLOB_ONLYDIR) as $dir) {
            if (strtolower(basename($dir)) === strtolower($subjectId)) {
                $folder = basename($dir);
                break;
            }
        }
        if ($folder === null) {
            return null;
        }
        $file = null;
        foreach (glob(LEARN_DIR . "/" . $folder . "/*.md") as $candidate) {
            if (pathinfo($candidate, PATHINFO_FILENAME) === $chapterId) {
                $file = $candidate;
                break;
            }
        }
        if ($file === null) {
            return null;
        }
    }

    $raw = file_get_contents($file);
    $meta = parseFrontmatter($raw);
    if ($title === "") {
        $title = $meta["title"];
        if ($title === "") {
            $title = ucwords(
                str_replace(["-", "_"], " ", preg_replace('/^ch\d+-/', "", $chapterId)),
            );
        }
    }

    return [
        "title" => $title,
        "excerpt" => markdownExcerpt($meta["content"]),
    ];
}

function productSeoData(string $id): ?array
{
    $file = PRODUCTS_DIR . "/" . $id . "/index.md";
    if (!file_exists($file)) {
        return null;
    }

    $meta = parseFrontmatter(file_get_contents($file));
    $content = $meta["content"];

    $title = $meta["title"];
    if ($title === "") {
        if (preg_match('/!\[([^\]]+)\]\([^)]*\)/', $content, $m)) {
            $title = $m[1];
        } elseif (preg_match('/^#\s+(.+)$/m', $content, $m)) {
            $title = $m[1];
        } else {
            $title = ucwords(str_replace(["-", "_"], " ", $id));
        }
    }

    $image = "";
    if (preg_match('/!\[[^\]]*\]\(images\/([^)]+)\)/', $content, $m)) {
        $image =
            seoDomain() .
            "/api/products/" .
            rawurlencode($id) .
            "/images/" .
            rawurlencode($m[1]);
    }

    return [
        "title" => $title,
        "excerpt" => markdownExcerpt($content),
        "image" => $image,
    ];
}

switch ($route["type"]) {
    case "home":
        $seo["jsonLd"] = jsonLdPerson($seo);
        break;

    case "blog-list":
        $seo["title"] = "Blog — " . $seo["name"];
        $seo["canonical"] = $seo["domain"] . "/blog";
        break;

    case "blog":
        $post = blogSeoData($pdo, $route["id"]);
        if ($post === null) {
            $route["type"] = "404";
            $seo["canonical"] = $seo["domain"] . "/blog/" . rawurlencode($route["id"]);
            $seo["robots"] = "noindex, follow";
            http_response_code(404);
            break;
        }
        $seo["title"] = $post["title"] . " — " . $seo["name"];
        if ($post["excerpt"] !== "") {
            $seo["description"] = $post["excerpt"];
        }
        if ($post["image"] !== "") {
            $seo["avatar"] = $post["image"];
        }
        $seo["canonical"] = $seo["domain"] . "/blog/" . rawurlencode($route["id"]);
        $seo["type"] = "article";
        $seo["jsonLd"] = jsonLdBlogPosting($seo, [
            "title" => $post["title"],
            "excerpt" => $seo["description"],
            "date" => $post["date"],
            "image" => $post["image"] !== "" ? $post["image"] : $seo["avatar"],
            "canonical" => $seo["canonical"],
        ]);
        break;

    case "expo":
        $seo["title"] = "Expo — " . $seo["name"];
        $seo["canonical"] = $seo["domain"] . "/expo";
        break;

    case "product":
        $product = productSeoData($route["id"]);
        if ($product === null) {
            $route["type"] = "404";
            $seo["canonical"] = $seo["domain"] . "/product/" . rawurlencode($route["id"]);
            $seo["robots"] = "noindex, follow";
            http_response_code(404);
            break;
        }
        $seo["title"] = $product["title"] . " — " . $seo["name"];
        if ($product["excerpt"] !== "") {
            $seo["description"] = $product["excerpt"];
        }
        if ($product["image"] !== "") {
            $seo["avatar"] = $product["image"];
        }
        $seo["canonical"] = $seo["domain"] . "/product/" . rawurlencode($route["id"]);
        $seo["type"] = "product";
        $seo["jsonLd"] = jsonLdProduct($seo, [
            "title" => $product["title"],
            "excerpt" => $seo["description"],
            "image" => $product["image"] !== "" ? $product["image"] : $seo["avatar"],
            "canonical" => $seo["canonical"],
        ]);
        break;

    case "learn":
        $seo["title"] = "Learn — " . $seo["name"];
        $seo["canonical"] = $seo["domain"] . "/learn";
        break;

    case "subject":
        $subject = subjectSeoData($pdo, $route["id"]);
        if ($subject === null) {
            $route["type"] = "404";
            $seo["canonical"] = $seo["domain"] . "/learn/" . rawurlencode($route["id"]);
            $seo["robots"] = "noindex, follow";
            http_response_code(404);
            break;
        }
        $seo["title"] = $subject["title"] . " — " . $seo["name"];
        if ($subject["excerpt"] !== "") {
            $seo["description"] = $subject["excerpt"];
        }
        $seo["canonical"] = $seo["domain"] . "/learn/" . rawurlencode($route["id"]);
        $seo["type"] = "article";
        break;

    case "chapter":
        $chapter = chapterSeoData($pdo, $route["subject"], $route["chapter"]);
        if ($chapter === null) {
            $route["type"] = "404";
            $seo["canonical"] =
                $seo["domain"] .
                "/learn/" .
                rawurlencode($route["subject"]) .
                "/" .
                rawurlencode($route["chapter"]);
            $seo["robots"] = "noindex, follow";
            http_response_code(404);
            break;
        }
        $seo["title"] = $chapter["title"] . " — " . $seo["name"];
        if ($chapter["excerpt"] !== "") {
            $seo["description"] = $chapter["excerpt"];
        }
        $seo["canonical"] =
            $seo["domain"] .
            "/learn/" .
            rawurlencode($route["subject"]) .
            "/" .
            rawurlencode($route["chapter"]);
        $seo["type"] = "article";
        $seo["jsonLd"] = jsonLdLearningResource($seo, [
            "title" => $chapter["title"],
            "excerpt" => $seo["description"],
            "canonical" => $seo["canonical"],
        ]);
        break;

    case "stats":
        $seo["title"] = "Stats — " . $seo["name"];
        $seo["canonical"] = $seo["domain"] . "/stats";
        break;

    case "admin":
        $seo["title"] = "Admin — " . $seo["name"];
        $seo["canonical"] = $seo["domain"] . "/admin";
        $seo["robots"] = "noindex, nofollow";
        break;

    default:
        $seo["title"] = "Page Not Found — " . $seo["name"];
        $seo["description"] = "The page you requested could not be found.";
        $seo["canonical"] = $seo["domain"] . "/";
        $seo["robots"] = "noindex, follow";
        http_response_code(404);
        break;
}

$template = @file_get_contents(__DIR__ . "/../../index.html");
if ($template === false) {
    http_response_code(500);
    echo "Server error";
    exit;
}

if (preg_match('/<head\b[^>]*>/i', $template, $m, PREG_OFFSET_CAPTURE)) {
    $headStart = $m[0][1];
    $headOpenLen = strlen($m[0][0]);
    $headEnd = stripos($template, "</head>", $headStart);
    $headInner = substr(
        $template,
        $headStart + $headOpenLen,
        $headEnd - ($headStart + $headOpenLen),
    );
    $keptHead = trim(stripSeoTags($headInner));
    $newHeadInner =
        ($keptHead !== "" ? $keptHead . "\n        " : "") .
        seoHead($seo) .
        "\n    ";
    $template =
        substr($template, 0, $headStart + $headOpenLen) .
        "\n        " .
        $newHeadInner .
        substr($template, $headEnd);
}

header("Content-Type: text/html; charset=UTF-8");
foreach (securityHeaders() as $name => $value) {
    header("{$name}: {$value}");
}
echo $template;
