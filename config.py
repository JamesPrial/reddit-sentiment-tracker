"""
Centralized configuration management for Reddit Sentiment Tracker
"""
import os
from dataclasses import dataclass
from typing import List, Optional
from dotenv import load_dotenv


@dataclass
class RedditConfig:
    """Reddit API configuration"""
    client_id: str
    client_secret: str
    user_agent: str = 'RedditSentimentTracker/1.0'
    rate_limit_delay: float = 1.0
    max_posts_per_subreddit: Optional[int] = None
    max_comments_per_post: Optional[int] = None


@dataclass
class DatabaseConfig:
    """Database connection configuration"""
    connection_string: str


@dataclass
class SentimentConfig:
    """Sentiment analysis configuration"""
    enabled: bool
    use_api: bool
    api_key: Optional[str] = None
    batch_size: int = 10


@dataclass
class TrackerConfig:
    """Tracker application configuration"""
    subreddits: List[str]
    default_timeframe_hours: int = 24
    fetch_comments: bool = True
    update_interval_hours: int = 6
    default_sort_by: str = 'new'


@dataclass
class AppConfig:
    """Complete application configuration"""
    reddit: RedditConfig
    database: DatabaseConfig
    sentiment: SentimentConfig
    tracker: TrackerConfig

    @classmethod
    def from_env(cls) -> 'AppConfig':
        """Load configuration from environment variables"""
        load_dotenv()

        # Reddit configuration
        reddit_config = RedditConfig(
            client_id=os.getenv('REDDIT_CLIENT_ID', ''),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET', ''),
            user_agent=os.getenv('REDDIT_USER_AGENT', 'RedditSentimentTracker/1.0'),
            rate_limit_delay=float(os.getenv('RATE_LIMIT_DELAY', '1.0')),
            max_posts_per_subreddit=int(os.getenv('MAX_POSTS_PER_SUBREDDIT'))
            if os.getenv('MAX_POSTS_PER_SUBREDDIT') else None,
            max_comments_per_post=int(os.getenv('MAX_COMMENTS_PER_POST'))
            if os.getenv('MAX_COMMENTS_PER_POST') else None
        )

        # Database configuration
        database_config = DatabaseConfig(
            connection_string=os.getenv('DATABASE_URL', '')
        )

        # Sentiment configuration
        anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        use_local_model = os.getenv('USE_LOCAL_MODEL', 'false').lower() == 'true'

        sentiment_config = SentimentConfig(
            enabled=bool(anthropic_api_key or use_local_model),
            use_api=bool(anthropic_api_key),
            api_key=anthropic_api_key,
            batch_size=int(os.getenv('SENTIMENT_BATCH_SIZE', '10'))
        )

        # Tracker configuration
        subreddits_str = os.getenv('SUBREDDITS', 'Anthropic,ClaudeCode,ClaudeAI')
        tracker_config = TrackerConfig(
            subreddits=[s.strip() for s in subreddits_str.split(',')],
            default_timeframe_hours=int(os.getenv('DEFAULT_TIMEFRAME_HOURS', '24')),
            fetch_comments=os.getenv('FETCH_COMMENTS', 'true').lower() == 'true',
            update_interval_hours=int(os.getenv('UPDATE_INTERVAL_HOURS', '6')),
            default_sort_by=os.getenv('DEFAULT_SORT_BY', 'new')
        )

        return cls(
            reddit=reddit_config,
            database=database_config,
            sentiment=sentiment_config,
            tracker=tracker_config
        )

    def validate(self) -> None:
        """Validate configuration, raise ValueError if invalid"""
        errors = []

        # Validate Reddit config
        if not self.reddit.client_id:
            errors.append("REDDIT_CLIENT_ID not found in environment")
        if not self.reddit.client_secret:
            errors.append("REDDIT_CLIENT_SECRET not found in environment")

        # Validate database config
        if not self.database.connection_string:
            errors.append("DATABASE_URL not found in environment")

        # Validate subreddits
        if not self.tracker.subreddits:
            errors.append("No subreddits configured")

        if errors:
            raise ValueError("Configuration errors:\n- " + "\n- ".join(errors))


# Constants moved from main.py
PAUSE_BEFORE_ANALYZE = 2  # seconds
SECONDS_PER_HOUR = 3600
DEFAULT_HOURS = 24
DEFAULT_BATCH_SIZE = 10
DEFAULT_UPDATE_INTERVAL = 6  # hours
DEFAULT_SORT_BY = 'new'
SUBREDDIT_DELIM = ','

# CLI command names
FETCH = 'fetch'
ANALYZE = 'analyze'
STATS = 'stats'
MONITOR = 'monitor'