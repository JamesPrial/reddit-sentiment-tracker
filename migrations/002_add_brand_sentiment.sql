-- Migration to add brand-aware sentiment columns
-- Run this after the initial schema has been created

-- Add brand sentiment columns to posts table
ALTER TABLE posts
    ADD COLUMN IF NOT EXISTS brand_sentiment_score FLOAT,
    ADD COLUMN IF NOT EXISTS brand_sentiment_label VARCHAR(30),
    ADD COLUMN IF NOT EXISTS brand_mentions JSONB,
    ADD COLUMN IF NOT EXISTS sentiment_metadata JSONB;

-- Add brand sentiment columns to comments table
ALTER TABLE comments
    ADD COLUMN IF NOT EXISTS brand_sentiment_score FLOAT,
    ADD COLUMN IF NOT EXISTS brand_sentiment_label VARCHAR(30),
    ADD COLUMN IF NOT EXISTS brand_mentions JSONB,
    ADD COLUMN IF NOT EXISTS sentiment_metadata JSONB;

-- Create indexes for brand sentiment queries
CREATE INDEX IF NOT EXISTS idx_posts_brand_sentiment ON posts(brand_sentiment_label);
CREATE INDEX IF NOT EXISTS idx_posts_brand_score ON posts(brand_sentiment_score);
CREATE INDEX IF NOT EXISTS idx_comments_brand_sentiment ON comments(brand_sentiment_label);
CREATE INDEX IF NOT EXISTS idx_comments_brand_score ON comments(brand_sentiment_score);

-- Create index for JSON brand mentions (PostgreSQL GIN index)
CREATE INDEX IF NOT EXISTS idx_posts_brand_mentions ON posts USING GIN (brand_mentions);
CREATE INDEX IF NOT EXISTS idx_comments_brand_mentions ON comments USING GIN (brand_mentions);

-- Add sentiment analysis timestamp columns
ALTER TABLE posts
    ADD COLUMN IF NOT EXISTS sentiment_analyzed_at TIMESTAMP;

ALTER TABLE comments
    ADD COLUMN IF NOT EXISTS sentiment_analyzed_at TIMESTAMP;