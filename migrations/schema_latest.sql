-- Complete schema for fresh installations
-- Combines all migrations (001-003) into a single script

-- Subreddits table
CREATE TABLE IF NOT EXISTS subreddits (
    subreddit_id VARCHAR(20) PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    subscriber_count INT,
    created_utc TIMESTAMP,
    date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Authors table
CREATE TABLE IF NOT EXISTS authors (
    author_id VARCHAR(20) PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    comment_karma INT,
    link_karma INT,
    created_utc TIMESTAMP,
    date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Posts table (includes brand sentiment columns)
CREATE TABLE IF NOT EXISTS posts (
    post_id VARCHAR(20) PRIMARY KEY,
    subreddit_id VARCHAR(20) NOT NULL REFERENCES subreddits(subreddit_id),
    author_id VARCHAR(20) REFERENCES authors(author_id),
    title TEXT NOT NULL,
    body TEXT,
    url VARCHAR(2048),
    score INT,
    num_comments INT,
    created_utc TIMESTAMP,
    title_sentiment_score FLOAT,
    title_sentiment_label VARCHAR(10),
    body_sentiment_score FLOAT,
    body_sentiment_label VARCHAR(10),
    brand_sentiment_score FLOAT,
    brand_sentiment_label VARCHAR(30),
    brand_mentions JSONB,
    sentiment_metadata JSONB,
    sentiment_analyzed_at TIMESTAMP,
    last_scraped_utc TIMESTAMP
);

-- Comments table (includes brand sentiment columns)
CREATE TABLE IF NOT EXISTS comments (
    comment_id VARCHAR(20) PRIMARY KEY,
    post_id VARCHAR(20) NOT NULL REFERENCES posts(post_id),
    author_id VARCHAR(20) REFERENCES authors(author_id),
    parent_id VARCHAR(20),
    body TEXT NOT NULL,
    score INT,
    created_utc TIMESTAMP,
    sentiment_score FLOAT,
    sentiment_label VARCHAR(10),
    brand_sentiment_score FLOAT,
    brand_sentiment_label VARCHAR(30),
    brand_mentions JSONB,
    sentiment_metadata JSONB,
    sentiment_analyzed_at TIMESTAMP,
    last_scraped_utc TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_posts_subreddit_id ON posts(subreddit_id);
CREATE INDEX IF NOT EXISTS idx_posts_created_utc ON posts(created_utc);
CREATE INDEX IF NOT EXISTS idx_posts_last_scraped ON posts(last_scraped_utc);
CREATE INDEX IF NOT EXISTS idx_posts_brand_sentiment ON posts(brand_sentiment_label);
CREATE INDEX IF NOT EXISTS idx_posts_brand_score ON posts(brand_sentiment_score);
CREATE INDEX IF NOT EXISTS idx_comments_post_id ON comments(post_id);
CREATE INDEX IF NOT EXISTS idx_comments_author_id ON comments(author_id);
CREATE INDEX IF NOT EXISTS idx_comments_created_utc ON comments(created_utc);
CREATE INDEX IF NOT EXISTS idx_comments_brand_sentiment ON comments(brand_sentiment_label);
CREATE INDEX IF NOT EXISTS idx_comments_brand_score ON comments(brand_sentiment_score);
CREATE INDEX IF NOT EXISTS idx_posts_brand_mentions ON posts USING GIN (brand_mentions);
CREATE INDEX IF NOT EXISTS idx_comments_brand_mentions ON comments USING GIN (brand_mentions);
