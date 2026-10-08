<?php

use Psr\Http\Message\ResponseInterface as Response;
use Psr\Http\Message\ServerRequestInterface as Request;
use Slim\App;

/**
 * GitHub contribution calendar API.
 *
 * Fetches and parses https://github.com/users/{user}/contributions server-side
 * because the page sends no CORS headers and cannot be read from the browser.
 * Parsing logic adapted from github-contribution-calendar-api
 * (c) Ryan Christian (MIT) - https://github.com/rschristian/github-contribution-calendar-api
 */

const GITHUB_CAL_TTL = 86400;
const GITHUB_CAL_TIMEOUT = 2.5;
const GITHUB_CAL_MAX_SPAN = 20;
const GITHUB_CAL_MAX_SECONDS = 45;
const GITHUB_CAL_ROLLING = 0;

function githubCalCacheFile(string $user, int $year): string
{
    return sys_get_temp_dir()
        . "/github_contrib_"
        . md5(strtolower($user) . "|" . $year)
        . ".json";
}

function githubCalReadCache(string $file): ?array
{
    if (!file_exists($file)) {
        return null;
    }
    $raw = @file_get_contents($file);
    if ($raw === false) {
        return null;
    }
    $data = json_decode($raw, true);
    if (!is_array($data) || !isset($data["total"], $data["days"]) || !is_array($data["days"])) {
        return null;
    }
    return $data;
}

function githubCalWriteCache(string $file, array $data): void
{
    @file_put_contents(
        $file,
        json_encode($data, JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE),
        LOCK_EX,
    );
}

function githubCalFetchHtml(string $user, ?int $year): ?string
{
    $url = "https://github.com/users/" . $user . "/contributions";
    if ($year !== null) {
        $url .= "?to=" . $year . "-01-01";
    }
    $context = stream_context_create([
        "http" => [
            "timeout" => GITHUB_CAL_TIMEOUT,
            "header" => "User-Agent: shakeelansari.me\r\nAccept: text/html\r\n",
            "ignore_errors" => true,
        ],
    ]);
    $html = @file_get_contents($url, false, $context);
    if ($html === false) {
        return null;
    }
    $status = 0;
    foreach ($http_response_header ?? [] as $header) {
        if (preg_match('/^HTTP\/\S+\s+(\d{3})/', $header, $matches)) {
            $status = (int) $matches[1];
        }
    }
    if ($status < 200 || $status >= 300) {
        return null;
    }
    return $html;
}

function githubCalParse(string $html): ?array
{
    libxml_use_internal_errors(true);
    $dom = new DOMDocument();
    $dom->loadHTML($html, LIBXML_NONET | LIBXML_NOERROR | LIBXML_NOWARNING);
    libxml_clear_errors();
    $xpath = new DOMXPath($dom);

    $total = null;
    $headings = $xpath->query(
        "//div[contains(concat(' ', normalize-space(@class), ' '), ' js-yearly-contributions ')]//h2",
    );
    if ($headings !== false && $headings->length > 0) {
        $text = $headings->item(0)->textContent;
        if (preg_match('/[0-9][0-9,]*/', $text, $matches)) {
            $total = (int) str_replace(",", "", $matches[0]);
        }
    }

    $counts = [];
    $tooltips = $xpath->query("//tool-tip[@for]");
    if ($tooltips !== false) {
        foreach ($tooltips as $tooltip) {
            $id = $tooltip->getAttribute("for");
            $text = trim($tooltip->textContent);
            $counts[$id] = preg_match('/^(\d+)/', $text, $matches)
                ? (int) $matches[1]
                : 0;
        }
    }

    $days = [];
    $cells = $xpath->query(
        "//td[contains(concat(' ', normalize-space(@class), ' '), ' ContributionCalendar-day ')]",
    );
    if ($cells !== false) {
        foreach ($cells as $cell) {
            $date = $cell->getAttribute("data-date");
            if ($date === "") {
                continue;
            }
            $id = $cell->getAttribute("id");
            $days[] = [
                "date" => $date,
                "intensity" => $cell->getAttribute("data-level"),
                "count" => $counts[$id] ?? 0,
            ];
        }
    }

    if ($total === null || count($days) === 0) {
        return null;
    }
    usort($days, fn (array $a, array $b) => strcmp($a["date"], $b["date"]));
    return ["total" => $total, "days" => $days];
}

function githubCalGetYear(string $user, int $year): ?array
{
    $file = githubCalCacheFile($user, $year);
    $cached = githubCalReadCache($file);
    if ($cached !== null && time() - filemtime($file) < GITHUB_CAL_TTL) {
        return $cached;
    }
    $html = githubCalFetchHtml($user, $year === GITHUB_CAL_ROLLING ? null : $year);
    if ($html === null) {
        return $cached;
    }
    $parsed = githubCalParse($html);
    if ($parsed === null) {
        return $cached;
    }
    githubCalWriteCache($file, $parsed);
    return $parsed;
}

return function (App $app) {
    $app->get("/github/contributions", function (
        Request $request,
        Response $response,
    ) {
        $params = $request->getQueryParams();
        $user = (string) ($params["user"] ?? "");
        if (!validateId($user, '/^[a-zA-Z0-9](?:[a-zA-Z0-9]|-(?=[a-zA-Z0-9])){0,38}$/')) {
            return jsonResponse($response, ["error" => "Invalid user"], 400);
        }

        $minYear = 2008;
        $maxYear = (int) date("Y") + 1;
        $years = [GITHUB_CAL_ROLLING];

        if (isset($params["year"]) && $params["year"] !== "") {
            $year = filter_var($params["year"], FILTER_VALIDATE_INT);
            if ($year === false || $year < $minYear || $year > $maxYear) {
                return jsonResponse($response, ["error" => "Invalid year"], 400);
            }
            $years = [$year];
        } elseif (
            isset($params["from"], $params["to"])
            && $params["from"] !== ""
            && $params["to"] !== ""
        ) {
            $from = filter_var($params["from"], FILTER_VALIDATE_INT);
            $to = filter_var($params["to"], FILTER_VALIDATE_INT);
            if (
                $from === false || $to === false
                || $from < $minYear || $to > $maxYear
                || $from > $to || $to - $from >= GITHUB_CAL_MAX_SPAN
            ) {
                return jsonResponse($response, ["error" => "Invalid year range"], 400);
            }
            $years = range($from, $to);
        }

        $started = microtime(true);
        $total = 0;
        $days = [];
        foreach ($years as $year) {
            if (microtime(true) - $started > GITHUB_CAL_MAX_SECONDS) {
                return jsonResponse($response, ["error" => "Contribution lookup timed out"], 503);
            }
            $data = githubCalGetYear($user, (int) $year);
            if ($data === null) {
                return jsonResponse($response, ["error" => "Failed to fetch contribution data"], 502);
            }
            $total += (int) $data["total"];
            foreach ($data["days"] as $day) {
                $days[] = $day;
            }
        }
        usort($days, fn (array $a, array $b) => strcmp($a["date"], $b["date"]));

        $weeks = [];
        for ($i = 0; $i < count($days); $i += 7) {
            $weeks[] = array_slice($days, $i, 7);
        }

        return jsonResponse($response, [
            "total" => $total,
            "contributions" => $weeks,
        ])->withHeader("Cache-Control", "public, max-age=3600");
    });
};
