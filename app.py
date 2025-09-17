"""
Core application logic for Reddit Sentiment Tracker
"""
import logging
import time
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any

from config import AppConfig, PAUSE_BEFORE_ANALYZE, SECONDS_PER_HOUR
from data import DatabaseConnection, Post, Comment, PostRepository, CommentRepository
from reddit import RedditDataPipeline
from sentiment import SentimentPipeline, get_analyzer

logger = logging.getLogger(__name__)


class RedditSentimentTracker:
    """Main application for Reddit sentiment tracking"""

    def __init__(self, config: AppConfig):
        """Initialize the tracker with provided configuration"""
        self.config = config
        self.pipeline = None
        self.sentiment_analyzer = None
        self._initialize_components()

    def _initialize_components(self):
        """Initialize pipeline and sentiment analyzer"""
        try:
            # Initialize data pipeline (without sentiment config)
            self.pipeline = RedditDataPipeline(
                reddit_client_id=self.config.reddit.client_id,
                reddit_client_secret=self.config.reddit.client_secret,
                reddit_user_agent=self.config.reddit.user_agent,
                db_connection_string=self.config.database.connection_string,
                enable_sentiment_analysis=False  # We'll handle sentiment separately
            )

            # Configure fetch limits
            if self.config.reddit.max_posts_per_subreddit is not None:
                self.pipeline.fetcher.config.max_posts = self.config.reddit.max_posts_per_subreddit
            if self.config.reddit.max_comments_per_post is not None:
                self.pipeline.fetcher.config.max_comments_per_post = self.config.reddit.max_comments_per_post

            # Configure rate limiting
            self.pipeline.fetcher.config.rate_limit_delay = self.config.reddit.rate_limit_delay

            # Initialize sentiment analyzer if enabled
            if self.config.sentiment.enabled:
                self.sentiment_analyzer = get_analyzer(
                    use_api=self.config.sentiment.use_api,
                    api_key=self.config.sentiment.api_key
                )
                logger.info(f"Sentiment analysis enabled (API: {self.config.sentiment.use_api})")

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
        sort_by: Optional[str] = None
    ) -> Tuple[int, int]:
        """
        Fetch data from configured subreddits within a timeframe

        Args:
            start: Start datetime
            end: End datetime
            hours: Hours of data to fetch (if start not provided)
            sort_by: Sort method for posts

        Returns:
            Tuple of (total_posts, total_comments)
        """
        sort_by = sort_by or self.config.tracker.default_sort_by
        end_time = end or datetime.utcnow()

        if start is None:
            hours = hours or self.config.tracker.default_timeframe_hours
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

        for subreddit in self.config.tracker.subreddits:
            try:
                print(f"\n📊 Processing r/{subreddit}...")
                result = self.pipeline.fetch_and_store_subreddit(
                    subreddit_name=subreddit,
                    start_time=start_time,
                    end_time=end_time,
                    fetch_comments=self.config.tracker.fetch_comments,
                    sort_by=sort_by
                )

                posts_count = result.get('posts_processed', 0)
                comments_count = result.get('comments_processed', 0)

                total_posts += posts_count
                total_comments += comments_count

                print(f"  ✓ Fetched {posts_count} posts and {comments_count} comments")

                # Run sentiment analysis if enabled
                if self.config.sentiment.enabled and self.sentiment_analyzer:
                    self._run_sentiment_for_subreddit(subreddit)

            except Exception as e:
                logger.error(f"Error fetching from r/{subreddit}: {e}")
                print(f"  ✗ Error: {e}")

        print(f"\n✅ Total fetched: {total_posts} posts, {total_comments} comments")
        return total_posts, total_comments

    def _run_sentiment_for_subreddit(self, subreddit: str):
        """Run sentiment analysis for a specific subreddit"""
        try:
            sentiment_pipeline = SentimentPipeline(
                self.config.database.connection_string,
                analyzer=self.sentiment_analyzer,
                batch_size=self.config.sentiment.batch_size
            )

            stats = sentiment_pipeline.analyze_subreddit(
                subreddit,
                hours_back=0,  # Only analyze new content
                force=False
            )

            if stats['posts_analyzed'] > 0 or stats['comments_analyzed'] > 0:
                print(f"  ✓ Analyzed {stats['posts_analyzed']} posts, {stats['comments_analyzed']} comments")

            sentiment_pipeline.close()

        except Exception as e:
            logger.warning(f"Sentiment analysis failed for r/{subreddit}: {e}")

    def analyze_sentiment(self, hours: Optional[int] = None, force: bool = False) -> Dict[str, Any]:
        """
        Run sentiment analysis on unanalyzed content

        Args:
            hours: Hours of data to analyze
            force: Force re-analysis of already analyzed content

        Returns:
            Dictionary with analysis statistics
        """
        if not self.config.sentiment.enabled:
            print("❌ Sentiment analysis is not enabled. Set ANTHROPIC_API_KEY or USE_LOCAL_MODEL=true")
            return {'error': 'Sentiment analysis not enabled'}

        hours = hours or self.config.tracker.default_timeframe_hours
        print(f"\n🤖 Running sentiment analysis on content from last {hours} hours...")

        sentiment_pipeline = SentimentPipeline(
            self.config.database.connection_string,
            analyzer=self.sentiment_analyzer,
            batch_size=self.config.sentiment.batch_size
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

            total_analyzed = post_stats.posts_analyzed + comment_stats.comments_analyzed
            total_errors = len(post_stats.errors) + len(comment_stats.errors)

            print(f"✅ Analyzed {post_stats.posts_analyzed} posts and {comment_stats.comments_analyzed} comments")

            if total_errors:
                print(f"⚠️  {total_errors} errors occurred during analysis")

            return {
                'posts_analyzed': post_stats.posts_analyzed,
                'comments_analyzed': comment_stats.comments_analyzed,
                'errors': total_errors
            }

        finally:
            sentiment_pipeline.close()

    def show_stats(self, hours: Optional[int] = None) -> Dict[str, Any]:
        """
        Show statistics for recent data

        Args:
            hours: Hours of data to include in stats

        Returns:
            Dictionary with statistics
        """
        hours = hours or self.config.tracker.default_timeframe_hours
        end_time = datetime.utcnow()
        since = end_time - timedelta(hours=hours)

        db = DatabaseConnection(self.config.database.connection_string)
        session = db.get_session()

        try:
            from sqlalchemy import func, or_

            # Get counts
            posts_count = session.query(func.count(Post.post_id)).filter(
                Post.created_utc >= since
            ).scalar()

            comments_count = session.query(func.count(Comment.comment_id)).filter(
                Comment.created_utc >= since
            ).scalar()

            stats = {
                'hours': hours,
                'posts': posts_count,
                'comments': comments_count
            }

            # Get sentiment stats if available
            if self.config.sentiment.enabled:
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

                stats.update({
                    'analyzed_posts': analyzed_posts,
                    'analyzed_comments': analyzed_comments,
                    'avg_title_sentiment': avg_title_sentiment,
                    'avg_body_sentiment': avg_body_sentiment,
                    'avg_comment_sentiment': avg_comment_sentiment
                })

            # Print stats
            print(f"\n📊 Statistics for last {hours} hours:")
            print(f"  Posts: {posts_count}")
            print(f"  Comments: {comments_count}")

            if self.config.sentiment.enabled:
                print(f"\n  Sentiment Analysis:")
                print(f"    Analyzed Posts: {analyzed_posts}/{posts_count}")
                print(f"    Analyzed Comments: {analyzed_comments}/{comments_count}")
                if avg_title_sentiment is not None:
                    print(f"    Avg Title Sentiment: {avg_title_sentiment:.2f}")
                if avg_body_sentiment is not None:
                    print(f"    Avg Body Sentiment: {avg_body_sentiment:.2f}")
                if avg_comment_sentiment:
                    print(f"    Avg Comment Sentiment: {avg_comment_sentiment:.2f}")

            return stats

        finally:
            session.close()

    def run_monitor(self, interval_hours: Optional[int] = None):
        """
        Run continuous monitoring

        Args:
            interval_hours: Update interval in hours
        """
        interval = interval_hours or self.config.tracker.update_interval_hours
        print(f"\n🔄 Starting continuous monitoring (interval: {interval} hours)")
        print("Press Ctrl+C to stop\n")

        try:
            while True:
                # Fetch new data
                self.fetch_data(hours=interval)

                # Run sentiment analysis if enabled
                if self.config.sentiment.enabled:
                    time.sleep(PAUSE_BEFORE_ANALYZE)  # Brief pause
                    self.analyze_sentiment(hours=interval)

                # Show stats
                self.show_stats(hours=interval)

                # Wait for next interval
                next_run = datetime.now() + timedelta(hours=interval)
                print(f"\n💤 Next run at {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
                time.sleep(interval * SECONDS_PER_HOUR)

        except KeyboardInterrupt:
            print("\n\n👋 Monitoring stopped")

    def update_stale_posts(self, hours_old: int = 6) -> Dict[str, Any]:
        """
        Update posts that haven't been scraped recently

        Args:
            hours_old: Hours since last scrape to consider stale

        Returns:
            Statistics dictionary
        """
        logger.info(f"Updating posts older than {hours_old} hours")
        stats = self.pipeline.update_stale_posts(hours_old)
        print(f"✅ Updated {stats.get('posts_updated', 0)} posts, added {stats.get('comments_added', 0)} comments")
        return stats

    def stream_subreddit(self, subreddit_name: str, fetch_comments: bool = True):
        """
        Stream new posts from a subreddit in real-time

        Args:
            subreddit_name: Name of the subreddit to stream
            fetch_comments: Whether to fetch comments for new posts
        """
        print(f"\n🔄 Starting real-time stream for r/{subreddit_name}")
        print("Press Ctrl+C to stop\n")

        try:
            self.pipeline.stream_and_store(subreddit_name, fetch_comments)
        except KeyboardInterrupt:
            print("\n\n👋 Streaming stopped")

    def close(self):
        """Clean up resources"""
        if self.pipeline:
            self.pipeline.close()