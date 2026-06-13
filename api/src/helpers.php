<?php

use Psr\Http\Message\ResponseInterface;
use Psr\Http\Message\ServerRequestInterface as Request;

const BLOGS_DIR = __DIR__ . "/../../blogs";
const LEARN_DIR = __DIR__ . "/../../tutorial";
const PRODUCTS_DIR = __DIR__ . "/../../products";

function parseFrontmatter(string $content): array
{
    $meta = [
        "title" => "",
        "excerpt" => "",
        "date" => "",
        "readTime" => "",
        "tags" => [],
        "content" => $content,
    ];

    if (preg_match('/^---\s*\n(.*?)\n---\s*\n(.*)/s', $content, $matches)) {
        $frontmatter = $matches[1];
        $meta["content"] = trim($matches[2]);

        foreach (explode("\n", $frontmatter) as $line) {
            if (preg_match('/^(\w+):\s*(.+)$/', trim($line), $fm)) {
                $key = $fm[1];
                $val = trim($fm[2]);
                if ($key === "tags") {
                    $meta["tags"] = array_map("trim", explode(",", $val));
                } else {
                    $meta[$key] = $val;
                }
            }
        }
    }

    return $meta;
}

function rewriteImageUrls(string $content): string
{
    return preg_replace("/\]\((images\/)/", '](/api/blogs/$1', $content);
}

function isRateLimited(string $ip): bool
{
    $file = sys_get_temp_dir() . "/login_attempts_" . md5($ip) . ".lock";

    if (!file_exists($file)) {
        return false;
    }

    if (time() - filemtime($file) > 900) {
        unlink($file);
        return false;
    }

    $attempts = (int) (@file_get_contents($file) ?: 0);
    return $attempts >= 5;
}

function incrementAttempts(string $ip): void
{
    $file = sys_get_temp_dir() . "/login_attempts_" . md5($ip) . ".lock";
    $attempts = (int) (@file_get_contents($file) ?: 0);
    file_put_contents($file, $attempts + 1);
}

function resetAttempts(string $ip): void
{
    $file = sys_get_temp_dir() . "/login_attempts_" . md5($ip) . ".lock";
    if (file_exists($file)) {
        unlink($file);
    }
}

function clientIp(Request $request): string
{
    $remote = $request->getServerParams()["REMOTE_ADDR"] ?? "127.0.0.1";
    $forwarded = $request->getHeaderLine("X-Forwarded-For");
    if ($forwarded !== "") {
        $ips = array_map("trim", explode(",", $forwarded));
        foreach ($ips as $ip) {
            if (filter_var($ip, FILTER_VALIDATE_IP)) {
                return $ip;
            }
        }
    }
    return $remote;
}

function resolveLocation(string $ip, PDO $pdo): object
{
    $private = ["127.0.0.1", "::1", "localhost"];
    if (in_array($ip, $private) || filter_var($ip, FILTER_VALIDATE_IP, FILTER_FLAG_NO_PRIV_RANGE | FILTER_FLAG_NO_RES_RANGE) === false) {
        return (object) ["country" => "Local", "country_code" => "XX", "region" => "", "city" => ""];
    }

    $stmt = $pdo->prepare("SELECT country, country_code, region, city FROM ip_location WHERE ip = ?");
    $stmt->execute([$ip]);
    $cached = $stmt->fetch();

    if ($cached) {
        return (object) $cached;
    }

    $data = @file_get_contents("http://ip-api.com/json/" . $ip . "?fields=status,country,countryCode,region,regionName,city");
    if ($data) {
        $json = json_decode($data);
        if ($json && ($json->status ?? "") === "success") {
            $country = $json->country ?? "";
            $countryCode = $json->countryCode ?? "";
            $region = $json->regionName ?? $json->region ?? "";
            $city = $json->city ?? "";
            $upsert = $pdo->prepare("INSERT INTO ip_location (ip, country, country_code, region, city) VALUES (?, ?, ?, ?, ?) ON DUPLICATE KEY UPDATE country = VALUES(country), country_code = VALUES(country_code), region = VALUES(region), city = VALUES(city)");
            $upsert->execute([$ip, $country, $countryCode, $region, $city]);
            return (object) ["country" => $country, "country_code" => $countryCode, "region" => $region, "city" => $city];
        }
    }

    return (object) ["country" => "Unknown", "country_code" => "??", "region" => "", "city" => ""];
}

function jsonResponse(
    ResponseInterface $response,
    array $data,
    int $status = 200,
): ResponseInterface {
    $payload = json_encode(
        $data,
        JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE,
    );
    $response->getBody()->write($payload);
    return $response
        ->withHeader("Content-Type", "application/json")
        ->withStatus($status);
}
