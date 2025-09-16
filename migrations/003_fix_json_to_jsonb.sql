-- Fix migration: Convert JSON columns to JSONB if they exist as JSON

-- Check and convert posts table columns
DO $$
BEGIN
    -- Check if brand_mentions exists as JSON and convert to JSONB
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'posts'
        AND column_name = 'brand_mentions'
        AND data_type = 'json'
    ) THEN
        ALTER TABLE posts ALTER COLUMN brand_mentions TYPE JSONB USING brand_mentions::JSONB;
    END IF;

    -- Check if sentiment_metadata exists as JSON and convert to JSONB
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'posts'
        AND column_name = 'sentiment_metadata'
        AND data_type = 'json'
    ) THEN
        ALTER TABLE posts ALTER COLUMN sentiment_metadata TYPE JSONB USING sentiment_metadata::JSONB;
    END IF;
END $$;

-- Check and convert comments table columns
DO $$
BEGIN
    -- Check if brand_mentions exists as JSON and convert to JSONB
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'comments'
        AND column_name = 'brand_mentions'
        AND data_type = 'json'
    ) THEN
        ALTER TABLE comments ALTER COLUMN brand_mentions TYPE JSONB USING brand_mentions::JSONB;
    END IF;

    -- Check if sentiment_metadata exists as JSON and convert to JSONB
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'comments'
        AND column_name = 'sentiment_metadata'
        AND data_type = 'json'
    ) THEN
        ALTER TABLE comments ALTER COLUMN sentiment_metadata TYPE JSONB USING sentiment_metadata::JSONB;
    END IF;
END $$;
