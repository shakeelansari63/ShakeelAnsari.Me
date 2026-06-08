CREATE TABLE IF NOT EXISTS blog (
    id VARCHAR(255) PRIMARY KEY,
    title VARCHAR(255) NOT NULL DEFAULT '',
    excerpt TEXT,
    date DATE,
    read_time VARCHAR(50),
    tags JSON,
    md_file VARCHAR(512) NOT NULL DEFAULT '',
    deleted TINYINT(1) NOT NULL DEFAULT 0,
    views INT NOT NULL DEFAULT 0,
    likes INT NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS blog_views (
    blog_id VARCHAR(255) NOT NULL,
    ip VARCHAR(45) NOT NULL,
    view_hour DATETIME NOT NULL,
    PRIMARY KEY (blog_id, ip, view_hour),
    FOREIGN KEY (blog_id) REFERENCES blog(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS blog_likes (
    blog_id VARCHAR(255) NOT NULL,
    ip VARCHAR(45) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (blog_id, ip),
    FOREIGN KEY (blog_id) REFERENCES blog(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS learn_subjects (
    id VARCHAR(255) PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    folder VARCHAR(255) NOT NULL,
    sort_order INT DEFAULT 0,
    thumbnail VARCHAR(255) NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS learn_chapters (
    id INT AUTO_INCREMENT PRIMARY KEY,
    subject_id VARCHAR(255) NOT NULL,
    chapter_id VARCHAR(255) NOT NULL,
    title VARCHAR(255) NOT NULL,
    md_file VARCHAR(512) NOT NULL,
    sort_order INT DEFAULT 0,
    FOREIGN KEY (subject_id) REFERENCES learn_subjects(id) ON DELETE CASCADE,
    UNIQUE KEY (subject_id, chapter_id)
);
