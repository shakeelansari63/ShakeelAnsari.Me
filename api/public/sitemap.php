<?php

ini_set("max_execution_time", 60);

require_once __DIR__ . "/../vendor/autoload.php";

$dotenv = Dotenv\Dotenv::createMutable(__DIR__ . "/..");
$dotenv->safeLoad();

$pdo = \App\DB::connect();

$appUrl = $_ENV["APP_URL"] ?? "https://[{#SEO-DOMAIN#}]";

$urls = [
    ["loc" => "$appUrl/", "changefreq" => "weekly", "priority" => "1.0"],
    ["loc" => "$appUrl/blog", "changefreq" => "weekly", "priority" => "0.8"],
    ["loc" => "$appUrl/expo", "changefreq" => "monthly", "priority" => "0.6"],
];

if ($pdo) {
    try {
        $stmt = $pdo->query(
            "SELECT id, date FROM blog WHERE deleted = 0 ORDER BY date DESC",
        );
        foreach ($stmt as $row) {
            $urls[] = [
                "loc" => "$appUrl/blog/{$row["id"]}",
                "lastmod" => $row["date"],
                "changefreq" => "monthly",
                "priority" => "0.7",
            ];
        }
    } catch (PDOException) {
        // proceed without blog entries
    }

    try {
        $stmt = $pdo->query(
            "SELECT id FROM learn_subjects ORDER BY id ASC",
        );
        foreach ($stmt as $row) {
            $urls[] = [
                "loc" => "$appUrl/learn/{$row["id"]}",
                "changefreq" => "monthly",
                "priority" => "0.6",
            ];
        }
    } catch (PDOException) {
        // proceed without subject entries
    }

    try {
        $stmt = $pdo->query(
            "SELECT lc.id, ls.id AS subject_id
             FROM learn_chapters lc
             JOIN learn_subjects ls ON ls.id = lc.subject_id
             ORDER BY ls.id ASC, lc.sort_order ASC",
        );
        foreach ($stmt as $row) {
            $urls[] = [
                "loc" => "$appUrl/learn/{$row["subject_id"]}/{$row["id"]}",
                "changefreq" => "monthly",
                "priority" => "0.5",
            ];
        }
    } catch (PDOException) {
        // proceed without chapter entries
    }
}

$productsDir = __DIR__ . "/../../products";
if (is_dir($productsDir)) {
    $dirs = array_filter(glob($productsDir . "/*", GLOB_ONLYDIR), "is_dir");
    sort($dirs);
    foreach ($dirs as $dir) {
        $productId = basename($dir);
        $urls[] = [
            "loc" => "$appUrl/product/$productId",
            "changefreq" => "monthly",
            "priority" => "0.6",
        ];
    }
}

$xml = '<?xml version="1.0" encoding="UTF-8"?>';
$xml .= '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">';
foreach ($urls as $u) {
    $xml .= "<url>";
    $xml .= "<loc>" . htmlspecialchars($u["loc"]) . "</loc>";
    if (isset($u["lastmod"])) {
        $xml .= "<lastmod>" . htmlspecialchars($u["lastmod"]) . "</lastmod>";
    }
    $xml .=
        "<changefreq>" . htmlspecialchars($u["changefreq"]) . "</changefreq>";
    $xml .= "<priority>" . htmlspecialchars($u["priority"]) . "</priority>";
    $xml .= "</url>";
}
$xml .= "</urlset>";

header("Content-Type: application/xml");
echo $xml;
