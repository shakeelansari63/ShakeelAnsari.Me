<?php

function seoEnv(string $key, string $fallback = ""): string
{
    return trim((string) ($_ENV[$key] ?? $fallback));
}

function seoDomain(): string
{
    $domain = seoEnv("SEO_DOMAIN", "[{#SEO-DOMAIN#}]");
    return "https://" . preg_replace('/^https?:\/\//', "", $domain);
}

function seoDefaults(): array
{
    $domain = seoDomain();
    return [
        "name" => seoEnv("SEO_NAME", "[{#SEO-NAME#}]"),
        "title" => seoEnv("SEO_NAME", "[{#SEO-NAME#}]") . " — " . seoEnv("SEO_TITLE", "[{#SEO-TITLE#}]"),
        "description" => seoEnv("SEO_DESC", "[{#SEO-DESC#}]"),
        "shortDescription" => seoEnv("SEO_DESC_SHORT", "[{#SEO-DESC-SHORT#}]"),
        "avatar" => seoEnv("SEO_AVATAR_URL", "[{#SEO-AVATAR-URL#}]"),
        "github" => seoEnv("SEO_GITHUB_URL", "[{#SEO-GITHUB-URL#}]"),
        "linkedin" => seoEnv("SEO_LINKEDIN_URL", "[{#SEO-LINKEDIN-URL#}]"),
        "twitterUser" => seoEnv("SEO_TWITTER_USER", "[{#SEO-TWITTER-USER#}]"),
        "domain" => $domain,
        "canonical" => $domain . "/",
        "type" => "website",
        "robots" => "index, follow",
        "jsonLd" => null,
    ];
}

function jsonLdPerson(array $seo): array
{
    return [
        "@context" => "https://schema.org",
        "@type" => "Person",
        "name" => $seo["name"],
        "url" => $seo["domain"] . "/",
        "jobTitle" => seoEnv("SEO_TITLE", "[{#SEO-TITLE#}]"),
        "sameAs" => array_values(array_filter([
            $seo["github"],
            $seo["linkedin"],
            "https://twitter.com/" . $seo["twitterUser"],
        ])),
    ];
}

function jsonLdBlogPosting(array $seo, array $post): array
{
    return [
        "@context" => "https://schema.org",
        "@type" => "BlogPosting",
        "headline" => $post["title"],
        "description" => $post["excerpt"],
        "datePublished" => $post["date"],
        "image" => $post["image"],
        "url" => $post["canonical"],
        "author" => [
            "@type" => "Person",
            "name" => $seo["name"],
            "url" => $seo["domain"] . "/",
        ],
        "publisher" => [
            "@type" => "Person",
            "name" => $seo["name"],
            "url" => $seo["domain"] . "/",
        ],
    ];
}

function jsonLdLearningResource(array $seo, array $chapter): array
{
    return [
        "@context" => "https://schema.org",
        "@type" => "LearningResource",
        "name" => $chapter["title"],
        "description" => $chapter["excerpt"],
        "url" => $chapter["canonical"],
        "inLanguage" => "en",
        "learningResourceType" => "Tutorial",
        "author" => [
            "@type" => "Person",
            "name" => $seo["name"],
        ],
    ];
}

function jsonLdProduct(array $seo, array $product): array
{
    return [
        "@context" => "https://schema.org",
        "@type" => "Product",
        "name" => $product["title"],
        "description" => $product["excerpt"],
        "image" => $product["image"],
        "url" => $product["canonical"],
        "brand" => [
            "@type" => "Brand",
            "name" => $seo["name"],
        ],
    ];
}

function seoHead(array $seo): string
{
    $escape = fn (string $v): string => htmlspecialchars($v, ENT_QUOTES, "UTF-8");

    $title = $escape($seo["title"]);
    $description = $escape($seo["description"]);
    $short = $escape($seo["shortDescription"]);
    $canonical = $escape($seo["canonical"]);
    $name = $escape($seo["name"]);
    $avatar = $escape($seo["avatar"]);
    $robots = $escape($seo["robots"]);

    $tags = [
        '<meta name="description" content="' . $description . '" />',
        '<meta name="robots" content="' . $robots . '" />',
        '<link rel="canonical" href="' . $canonical . '" />',
        "<title>{$title}</title>",
        '<meta property="og:title" content="' . $title . '" />',
        '<meta property="og:description" content="' . $description . '" />',
        '<meta property="og:url" content="' . $canonical . '" />',
        '<meta property="og:type" content="' . $escape($seo["type"]) . '" />',
        '<meta property="og:image" content="' . $avatar . '" />',
        '<meta name="twitter:card" content="summary_large_image" />',
        '<meta name="twitter:title" content="' . $name . '" />',
        '<meta name="twitter:description" content="' . $short . '" />',
    ];

    if (!empty($seo["jsonLd"])) {
        $json = json_encode(
            $seo["jsonLd"],
            JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE,
        );
        $tags[] = '<script type="application/ld+json">' . $json . "</script>";
    }

    return implode("\n        ", $tags);
}

function stripSeoTags(string $head): string
{
    $head = preg_replace('/<title\b[^>]*>.*?<\/title>/is', "", $head);
    $head = preg_replace(
        '/<meta\b[^>]*\b(?:name|property)=["\'](?:description|robots|og:[^"\']*|twitter:[^"\']*)["\'][^>]*>/i',
        "",
        $head,
    );
    $head = preg_replace('/<link\b[^>]*rel=["\']canonical["\'][^>]*>/i', "", $head);
    $head = preg_replace(
        '/<script\b[^>]*type=["\']application\/ld\+json["\'][^>]*>.*?<\/script>/is',
        "",
        $head,
    );
    return $head;
}

function markdownExcerpt(string $content, int $length = 160): string
{
    $content = preg_replace('/!\[[^\]]*\]\([^)]*\)/s', " ", $content);
    $content = preg_replace('/\[([^\]]*)\]\([^)]*\)/s', "$1", $content);
    $content = preg_replace('/```.*?```/s', " ", $content);
    $content = preg_replace('/`[^`]*`/', " ", $content);
    $content = preg_replace('/[#>*_~\-|\[\]]+/', " ", $content);
    $content = trim(preg_replace('/\s+/u', " ", $content));

    if (mb_strlen($content, "UTF-8") > $length) {
        $content = mb_substr($content, 0, $length, "UTF-8");
        $content = rtrim($content, ".,;: ") . "…";
    }

    return $content;
}
