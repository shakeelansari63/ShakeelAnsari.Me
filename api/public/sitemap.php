<?php

ini_set("max_execution_time", 60);

require_once __DIR__ . "/../vendor/autoload.php";

$dotenv = Dotenv\Dotenv::createMutable(__DIR__ . "/..");
$dotenv->safeLoad();

$pdo = \App\DB::connect();

$appUrl = $_ENV["APP_URL"] ?? "https://shakeelansari.me";

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
    } catch (PDOException $e) {
        // proceed without blog entries
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
    $xml .= "<changefreq>" . htmlspecialchars($u["changefreq"]) . "</changefreq>";
    $xml .= "<priority>" . htmlspecialchars($u["priority"]) . "</priority>";
    $xml .= "</url>";
}
$xml .= "</urlset>";

header("Content-Type: application/xml");
echo $xml;
