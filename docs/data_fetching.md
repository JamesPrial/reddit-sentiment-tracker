# Data Fetching & Pipeline

Deep dive into collecting Reddit data and persisting it to the database.

## Features

- **Unlimited fetching by default**: Fetches all posts and comments within a timeframe (no artificial limits)
- **Timeframe-based fetching**: Specify exact date ranges for data collection
- **Complete comment trees**: Fetches entire comment hierarchies including nested replies
- **Database integration**: Stores all data in PostgreSQL with proper relationships
- **Rate limiting**: Respects Reddit API limits automatically
- **Batch processing**: Process multiple subreddits efficiently
- **Real-time streaming**: Monitor subreddits for new content as it's posted
- **Update tracking**: Tracks when posts were last scraped for incremental updates

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up PostgreSQL and load `migrations/schema_latest.sql`

3. Configure Reddit API credentials:
```bash
cp .env.example .env
# Edit .env with your Reddit API credentials
```

## Default Behavior

By default, the fetcher will:
- Fetch **ALL posts** within the specified timeframe
- Fetch **ALL comments** for each post (entire comment tree)
- Process comments up to 100 levels deep
- Replace all "More Comments" indicators to get complete threads

## Usage

### Basic Fetching (All Data)

```python
from reddit.reddit_fetcher import RedditFetcher
from datetime import datetime, timedelta

# Initialize fetcher
fetcher = RedditFetcher(client_id, client_secret, user_agent)

# Define timeframe
end_time = datetime.utcnow()
start_time = end_time - timedelta(days=7)

# Fetch ALL posts and comments from the timeframe
posts, comments = fetcher.fetch_posts_and_comments(
    subreddit_name='wallstreetbets',
    start_time=start_time,
    end_time=end_time,
    fetch_comments=True  # Fetches all comments by default
)
```

### With Database Storage

```python
from reddit.pipeline import RedditDataPipeline

# Initialize pipeline
pipeline = RedditDataPipeline(
    reddit_client_id=client_id,
    reddit_client_secret=client_secret,
    reddit_user_agent='YourApp/1.0',
    db_connection_string='postgresql://user:pass@localhost/reddit'
)

# Fetch and store ALL data from subreddit
stats = pipeline.fetch_and_store_subreddit(
    subreddit_name='stocks',
    start_time=start_time,
    end_time=end_time,
    fetch_comments=True  # Fetches all comments
)
```

### Setting Limits (Optional)

If you want to limit the amount of data fetched:

```python
# Configure limits (only if needed)
fetcher.config.max_posts = 100  # Limit to 100 posts
fetcher.config.max_comments_per_post = 500  # Limit to 500 comments per post
fetcher.config.max_comment_depth = 10  # Limit comment tree depth
```

## Configuration

The `FetchConfig` class controls fetching behavior:

- `max_posts`: Maximum posts to fetch (None = unlimited, default)
- `max_comments_per_post`: Maximum comments per post (None = unlimited, default)
- `max_comment_depth`: Maximum comment nesting depth (default: 100)
- `include_deleted`: Include deleted content (default: False)
- `include_removed`: Include removed content (default: False)
- `rate_limit_delay`: Seconds between API calls (default: 0.5)
- `replace_more_limit`: Limit for expanding "More Comments" (None = all, default)

## Database Schema

The system uses four main tables:
- `subreddits`: Subreddit information
- `authors`: Reddit user information
- `posts`: Post content and metadata
- `comments`: Comment content and metadata

## Examples

See `fetch_example.py` for complete examples including:
- Fetching all data from a subreddit
- Processing multiple subreddits
- Updating stale posts
- Real-time streaming
- Custom configurations
