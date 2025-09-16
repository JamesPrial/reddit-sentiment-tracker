-- Reddit Sentiment Tracker Database Schema

-- Create subreddits table
CREATE TABLE IF NOT EXISTS subreddits (
    subreddit_id VARCHAR(20) PRIMARY KEY,  -- Increased size for Reddit IDs
    name VARCHAR(255) NOT NULL UNIQUE,
    subscriber_count INT,
    created_utc TIMESTAMP,
    date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create authors table
CREATE TABLE IF NOT EXISTS authors (
    author_id VARCHAR(20) PRIMARY KEY,  -- Increased size for Reddit IDs
    name VARCHAR(255) NOT NULL UNIQUE,
    comment_karma INT,
    link_karma INT,
    created_utc TIMESTAMP,
    date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create posts table
CREATE TABLE IF NOT EXISTS posts (
    post_id VARCHAR(20) PRIMARY KEY,  -- Increased size for Reddit IDs
    subreddit_id VARCHAR(20) NOT NULL,
    author_id VARCHAR(20),
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
    last_scraped_utc TIMESTAMP,
    FOREIGN KEY (subreddit_id) REFERENCES subreddits(subreddit_id),
    FOREIGN KEY (author_id) REFERENCES authors(author_id)
);

-- Create comments table
CREATE TABLE IF NOT EXISTS comments (
    comment_id VARCHAR(20) PRIMARY KEY,  -- Increased size for Reddit IDs
    post_id VARCHAR(20) NOT NULL,
    author_id VARCHAR(20),
    parent_id VARCHAR(20),  -- Increased size for Reddit IDs
    body TEXT NOT NULL,
    score INT,
    created_utc TIMESTAMP,
    sentiment_score FLOAT,
    sentiment_label VARCHAR(10),
    last_scraped_utc TIMESTAMP,
    FOREIGN KEY (post_id) REFERENCES posts(post_id),
    FOREIGN KEY (author_id) REFERENCES authors(author_id)
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_posts_subreddit_id ON posts(subreddit_id);
CREATE INDEX IF NOT EXISTS idx_posts_created_utc ON posts(created_utc);
CREATE INDEX IF NOT EXISTS idx_posts_last_scraped ON posts(last_scraped_utc);
CREATE INDEX IF NOT EXISTS idx_comments_post_id ON comments(post_id);
CREATE INDEX IF NOT EXISTS idx_comments_author_id ON comments(author_id);
CREATE INDEX IF NOT EXISTS idx_comments_created_utc ON comments(created_utc);