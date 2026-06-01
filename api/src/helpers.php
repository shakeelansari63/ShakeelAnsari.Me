<?php

use Psr\Http\Message\ResponseInterface;
use Psr\Http\Message\ServerRequestInterface as Request;

const BLOGS_DIR = __DIR__ . "/../../blogs";

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

function checkRateLimit(string $ip): bool
{
    $file = sys_get_temp_dir() . "/login_attempts_" . md5($ip) . ".lock";

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

function clientIp(Request $request): string
{
    $forwarded = $request->getHeaderLine("X-Forwarded-For");
    if ($forwarded !== "") {
        $ips = array_map("trim", explode(",", $forwarded));
        return $ips[0];
    }
    return $request->getServerParams()["REMOTE_ADDR"] ?? "127.0.0.1";
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
