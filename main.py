#!/usr/bin/env python3
"""
Reddit Sentiment Tracker - Main CLI Application
"""
import os
import sys
import logging
import argparse
import time
from datetime import datetime, timedelta
from typing import Optional
from dotenv import load_dotenv

from models import DatabaseConnection, Post, Comment
from data import RedditDataPipeline
from sentiment import SentimentAnalyzer, SentimentPipeline, get_analyzer
from repository import PostRepository, CommentRepository

PAUSE_BEFORE_ANALYZE = 2  # seconds
SECONDS_PER_HOUR = 3600
DEFAULT_HOURS = 24
DEFAULT_BATCH_SIZE = 10
DEFAULT_UPDATE_INTERVAL = 6  # hours
DEFAULT_SORT_BY = 'new'
SUBREDDIT_DELIM = ','
FETCH = 'fetch'
ANALYZE = 'analyze'
STATS = 'stats'
MONITOR = 'monitor'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reddit_tracker.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def parse_datetime(value: str) -> datetime:
    """Parse ISO 8601 datetime strings for CLI arguments."""
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid datetime '{value}'. Use ISO format, e.g. 2024-05-01T13:30"
        ) from exc


class RedditSentimentTracker:
    """Main application for Reddit sentiment tracking"""

    def __init__(self):
        """Initialize the tracker with configuration from environment"""
        # Load environment variables
        load_dotenv()

        # Reddit API configuration
        self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        self.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.reddit_user_agent = os.getenv('REDDIT_USER_AGENT', 'RedditSentimentTracker/1.0')

        # Database configuration
        self.db_url = os.getenv('DATABASE_URL')

        # Subreddits to track
        subreddits_str = os.getenv('SUBREDDITS', 'Anthropic,ClaudeCode,ClaudeAI')
        self.subreddits = [s.strip() for s in subreddits_str.split(SUBREDDIT_DELIM)]

        # Fetch configuration
        self.default_timeframe_hours = int(os.getenv('DEFAULT_TIMEFRAME_HOURS', '24'))
        self.fetch_comments = os.getenv('FETCH_COMMENTS', 'true').lower() == 'true'
        self.rate_limit_delay = float(os.getenv('RATE_LIMIT_DELAY', '1.0'))

        # Optional limits
        max_posts = os.getenv('MAX_POSTS_PER_SUBREDDIT')
        self.max_posts = int(max_posts) if max_posts else None
        max_comments = os.getenv('MAX_COMMENTS_PER_POST')
        self.max_comments = int(max_comments) if max_comments else None

        # Sentiment analysis configuration
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        self.use_local_model = os.getenv('USE_LOCAL_MODEL', 'false').lower() == 'true'
        self.enable_sentiment = bool(self.anthropic_api_key or self.use_local_model)

        # Validate configuration
        self._validate_config()

        # Initialize components
        self.pipeline = None
        self.sentiment_analyzer = None
        self._initialize_components()

    def _validate_config(self):
        """Validate required configuration"""
        if not self.reddit_client_id or not self.reddit_client_secret:
            raise ValueError("Reddit API credentials not found. Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env")

        if not self.db_url:
            raise ValueError("DATABASE_URL not found in .env file")

        logger.info(f"Configuration validated. Tracking subreddits: {', '.join(self.subreddits)}")

        if self.enable_sentiment:
            mode = 'API' if self.anthropic_api_key else 'Local Model'
            logger.info(f"Sentiment analysis enabled (Using {mode})")
        else:
            logger.info("Sentiment analysis disabled")

    def _initialize_components(self):
        """Initialize pipeline and sentiment analyzer"""
        try:
            # Initialize data pipeline
            self.pipeline = RedditDataPipeline(
                reddit_client_id=self.reddit_client_id,
                reddit_client_secret=self.reddit_client_secret,
                reddit_user_agent=self.reddit_user_agent,
                db_connection_string=self.db_url,
                enable_sentiment_analysis=self.enable_sentiment,
                sentiment_api_key=self.anthropic_api_key
            )

            # Configure fetch limits
            if self.max_posts is not None:
                self.pipeline.fetcher.config.max_posts = self.max_posts
            if self.max_comments is not None:
                self.pipeline.fetcher.config.max_comments_per_post = self.max_comments

            # Configure rate limiting
            self.pipeline.fetcher.config.rate_limit_delay = self.rate_limit_delay

            # Initialize sentiment analyzer if enabled
            if self.enable_sentiment:
                self.sentiment_analyzer = get_analyzer(
                    use_api=bool(self.anthropic_api_key),
                    api_key=self.anthropic_api_key
                )

            logger.info("Components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise

    def fetch_data(
        self,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        hours: Optional[int] = None,
        sort_by: str = DEFAULT_SORT_BY
    ) -> tuple[int, int]:
        """Fetch data from configured subreddits within a timeframe"""

        end_time = end or datetime.utcnow()
        if start is None:
            hours = hours or self.default_timeframe_hours
            start_time = end_time - timedelta(hours=hours)
        else:
            start_time = start

        if start_time >= end_time:
            raise ValueError("Start time must be before end time")

        window_hours = (end_time - start_time).total_seconds() / SECONDS_PER_HOUR
        print(
            f"\n📡 Fetching posts from {start_time.isoformat()} to {end_time.isoformat()} "
            f"(~{window_hours:.1f}h window)"
        )

        total_posts = 0
        total_comments = 0

        for subreddit in self.subreddits:
            try:
                print(f"\n📊 Processing r/{subreddit}...")
                result = self.pipeline.fetch_and_store_subreddit(
                    subreddit_name=subreddit,
                    start_time=start_time,
                    end_time=end_time,
                    fetch_comments=self.fetch_comments,
                    sort_by=sort_by
                )

                posts_count = result.get('posts_processed', 0)
                comments_count = result.get('comments_processed', 0)

                total_posts += posts_count
                total_comments += comments_count

                print(f"  ✓ Fetched {posts_count} posts and {comments_count} comments")

            except Exception as e:
                logger.error(f"Error fetching from r/{subreddit}: {e}")
                print(f"  ✗ Error: {e}")

        print(f"\n✅ Total fetched: {total_posts} posts, {total_comments} comments")
        return total_posts, total_comments

    def analyze_sentiment(self, hours: Optional[int] = None, force: bool = False):
        """Run sentiment analysis on unanalyzed content"""
        if not self.enable_sentiment:
            print("❌ Sentiment analysis is not enabled. Set ANTHROPIC_API_KEY or USE_LOCAL_MODEL=true")
            return

        hours = hours or DEFAULT_HOURS
        print(f"\n🤖 Running sentiment analysis on content from last {hours} hours...")

        sentiment_pipeline = SentimentPipeline(
            self.db_url,
            analyzer=self.sentiment_analyzer,
            batch_size=DEFAULT_BATCH_SIZE
        )

        try:
            post_stats = sentiment_pipeline.analyze_posts(
                hours_back=hours,
                force=force
            )
            comment_stats = sentiment_pipeline.analyze_comments(
                hours_back=hours,
                force=force
            )
        finally:
            sentiment_pipeline.close()

        print(
            f"✅ Analyzed {post_stats.posts_analyzed} posts and {comment_stats.comments_analyzed} comments"
        )

        total_errors = len(post_stats.errors) + len(comment_stats.errors)
        if total_errors:
            print(f"⚠️  {total_errors} errors occurred during analysis")

    def show_stats(self, hours: Optional[int] = None):
        """Show statistics for recent data"""
        hours = hours or DEFAULT_HOURS
        end_time = datetime.utcnow()
        since = end_time - timedelta(hours=hours)

        db = DatabaseConnection(self.db_url)
        session = db.get_session()

        try:
            post_repo = PostRepository(session)
            comment_repo = CommentRepository(session)

            # Get counts
            from sqlalchemy import func

            posts_count = session.query(func.count(Post.post_id)).filter(
                Post.created_utc >= since
            ).scalar()

            comments_count = session.query(func.count(Comment.comment_id)).filter(
                Comment.created_utc >= since
            ).scalar()

            # Get sentiment stats if available
            if self.enable_sentiment:
                from sqlalchemy import or_

                analyzed_posts = session.query(func.count(Post.post_id)).filter(
                    Post.created_utc >= since,
                    or_(
                        Post.title_sentiment_score.isnot(None),
                        Post.body_sentiment_score.isnot(None)
                    )
                ).scalar()

                analyzed_comments = session.query(func.count(Comment.comment_id)).filter(
                    Comment.created_utc >= since,
                    Comment.sentiment_score.isnot(None)
                ).scalar()

                avg_title_sentiment = session.query(func.avg(Post.title_sentiment_score)).filter(
                    Post.created_utc >= since,
                    Post.title_sentiment_score.isnot(None)
                ).scalar()

                avg_body_sentiment = session.query(func.avg(Post.body_sentiment_score)).filter(
                    Post.created_utc >= since,
                    Post.body_sentiment_score.isnot(None)
                ).scalar()

                avg_comment_sentiment = session.query(func.avg(Comment.sentiment_score)).filter(
                    Comment.created_utc >= since,
                    Comment.sentiment_score.isnot(None)
                ).scalar()

            print(f"\n📊 Statistics for last {hours} hours:")
            print(f"  Posts: {posts_count}")
            print(f"  Comments: {comments_count}")

            if self.enable_sentiment:
                print(f"\n  Sentiment Analysis:")
                print(f"    Analyzed Posts: {analyzed_posts}/{posts_count}")
                print(f"    Analyzed Comments: {analyzed_comments}/{comments_count}")
                if avg_title_sentiment is not None:
                    print(f"    Avg Title Sentiment: {avg_title_sentiment:.2f}")
                if avg_body_sentiment is not None:
                    print(f"    Avg Body Sentiment: {avg_body_sentiment:.2f}")
                if avg_comment_sentiment:
                    print(f"    Avg Comment Sentiment: {avg_comment_sentiment:.2f}")

        finally:
            session.close()

    def run_monitor(self, interval_hours: int = DEFAULT_UPDATE_INTERVAL):
        """Run continuous monitoring"""
        print(f"\n🔄 Starting continuous monitoring (interval: {interval_hours} hours)")
        print("Press Ctrl+C to stop\n")

        try:
            while True:
                # Fetch new data
                self.fetch_data(hours=interval_hours)

                # Run sentiment analysis if enabled
                if self.enable_sentiment:
                    time.sleep(PAUSE_BEFORE_ANALYZE)  # Brief pause
                    self.analyze_sentiment(hours=interval_hours)

                # Show stats
                self.show_stats(hours=interval_hours)

                # Wait for next interval
                next_run = datetime.now() + timedelta(hours=interval_hours)
                print(f"\n💤 Next run at {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
                time.sleep(interval_hours * SECONDS_PER_HOUR)

        except KeyboardInterrupt:
            print("\n\n👋 Monitoring stopped")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description='Reddit Sentiment Tracker')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Fetch command
    fetch_parser = subparsers.add_parser(FETCH, help='Fetch Reddit data')
    fetch_parser.add_argument('--hours', type=int, help='Hours of data to fetch')
    fetch_parser.add_argument('--start', type=parse_datetime, help='Start time (ISO 8601)')
    fetch_parser.add_argument('--end', type=parse_datetime, help='End time (ISO 8601)')

    # Analyze command
    analyze_parser = subparsers.add_parser(ANALYZE, help='Run sentiment analysis')
    analyze_parser.add_argument('--hours', type=int, help='Hours of data to analyze')
    analyze_parser.add_argument('--force', action='store_true', help='Force re-analysis')

    # Stats command
    stats_parser = subparsers.add_parser(STATS, help='Show statistics')
    stats_parser.add_argument('--hours', type=int, help='Hours of data to include')

    # Monitor command
    monitor_parser = subparsers.add_parser(MONITOR, help='Run continuous monitoring')
    monitor_parser.add_argument('--interval', type=int, default=DEFAULT_UPDATE_INTERVAL, help='Update interval in hours')

    args = parser.parse_args()

    # Initialize tracker
    try:
        tracker = RedditSentimentTracker()
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        sys.exit(1)

    # Execute command
    if args.command == FETCH:
        tracker.fetch_data(start=args.start, end=args.end, hours=args.hours)
    elif args.command == ANALYZE:
        tracker.analyze_sentiment(hours=args.hours, force=args.force)
    elif args.command == STATS:
        tracker.show_stats(hours=args.hours)
    elif args.command == MONITOR:
        tracker.run_monitor(interval_hours=args.interval)
    else:
        # Default: show help
        parser.print_help()
        print("\nExamples:")
        print("  python main.py fetch --hours 24")
        print("  python main.py fetch --start 2024-05-01T00:00 --end 2024-05-02T00:00")
        print("  python main.py analyze")
        print("  python main.py stats")
        print("  python main.py monitor --interval 6")


if __name__ == '__main__':
    main()
