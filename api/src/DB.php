<?php

namespace App;

class DB
{
    public static function connect(): ?\PDO
    {
        if (empty($_ENV["DB_HOST"])) {
            return null;
        }

        try {
            $dsn = sprintf(
                "mysql:host=%s;port=%s;dbname=%s;charset=utf8mb4",
                $_ENV["DB_HOST"],
                $_ENV["DB_PORT"] ?? "3306",
                $_ENV["DB_NAME"],
            );
            $pdo = new \PDO($dsn, $_ENV["DB_USER"], $_ENV["DB_PASS"], [
                \PDO::ATTR_ERRMODE => \PDO::ERRMODE_EXCEPTION,
                \PDO::ATTR_DEFAULT_FETCH_MODE => \PDO::FETCH_ASSOC,
                \PDO::ATTR_EMULATE_PREPARES => false,
                \PDO::ATTR_TIMEOUT => 2,
            ]);
            return $pdo;
        } catch (\PDOException $e) {
            error_log("DB connection failed: " . $e->getMessage());
            return null;
        }
    }
}
