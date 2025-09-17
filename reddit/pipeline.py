"""
Data pipeline that integrates Reddit fetcher with database models
"""
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging
import os
import time
from .reddit_fetcher import RedditFetcher, FetchConfig
from prawcore.exceptions import NotFound, ResponseException
from data import (
    DatabaseConnection,
    SubredditRepository,
    AuthorRepository,
    PostRepository,
    CommentRepository,
)
from sentiment.pipeline import SentimentPipeline
from sentiment.analyzer import get_analyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RedditDataPipeline:
    """Pipeline for fetching Reddit data and storing it in the database"""

    def __init__(
        self,
        reddit_client_id: str,
        reddit_client_secret: str,
        reddit_user_agent: str,
        db_connection_string: str,
        enable_sentiment_analysis: bool = True,
        sentiment_api_key: Optional[str] = None
    ):
        """
        Initialize the data pipeline

        Args:
            reddit_client_id: Reddit API client ID
            reddit_client_secret: Reddit API client secret
            reddit_user_agent: User agent for Reddit API
            db_connection_string: PostgreSQL connection string
        """
        # Initialize Reddit fetcher
        self.fetcher = RedditFetcher(reddit_client_id, reddit_client_secret, reddit_user_agent)

        # Initialize database connection
        self.db = DatabaseConnection(db_connection_string)
        self.session = self.db.get_session()

        # Initialize repositories
        self.subreddit_repo = SubredditRepository(self.session)
        self.author_repo = AuthorRepository(self.session)
        self.post_repo = PostRepository(self.session)
        self.comment_repo = CommentRepository(self.session)

        # Initialize sentiment analysis if enabled
        self.enable_sentiment_analysis = enable_sentiment_analysis
        self.sentiment_pipeline = None
        if enable_sentiment_analysis:
            try:
                # Try to use API analyzer if key is available
                use_api = bool(sentiment_api_key or os.getenv('ANTHROPIC_API_KEY'))
                analyzer = get_analyzer(use_api=use_api, api_key=sentiment_api_key)
                self.sentiment_pipeline = SentimentPipeline(
                    db_connection_string,
                    analyzer=analyzer,
                    batch_size=10
                )
                logger.info(f"Sentiment analysis enabled (API: {use_api})")
            except Exception as e:
                logger.warning(f"Could not initialize sentiment analysis: {e}")
                self.enable_sentiment_analysis = False

        # Statistics
        self.stats = {
            'subreddits_processed': 0,
            'posts_processed': 0,
            'comments_processed': 0,
            'authors_processed': 0,
            'errors': []
        }

    def _process_subreddit(self, subreddit_data: Dict[str, Any]) -> Optional[str]:
        """Process and store subreddit information"""
        try:
            subreddit = self.subreddit_repo.create_or_update(
                subreddit_id=subreddit_data['subreddit_id'],
                name=subreddit_data['name'],
                subscriber_count=subreddit_data.get('subscriber_count'),
                created_utc=subreddit_data.get('created_utc')
            )
            self.stats['subreddits_processed'] += 1
            return subreddit.subreddit_id
        except Exception as e:
            logger.error(f"Error processing subreddit {subreddit_data.get('name')}: {e}")
            self.stats['errors'].append(f"Subreddit error: {e}")
            return None

    def _process_author(self, author_data: Dict[str, Any]) -> Optional[str]:
        """Process and store author information"""
        if not author_data or author_data.get('author_name') == '[deleted]':
            return None

        try:
            # Try to fetch full author data if we only have basic info
            if 'comment_karma' not in author_data:
                try:
                    author_obj = self.fetcher.reddit.redditor(author_data['author_name'])
                    author_data.update({
                        'comment_karma': author_obj.comment_karma,
                        'link_karma': author_obj.link_karma,
                        'created_utc': datetime.utcfromtimestamp(author_obj.created_utc)
                    })
                except:
                    # If we can't fetch full data, use what we have
                    pass

            author = self.author_repo.create_or_update(
                author_id=author_data['author_id'],
                name=author_data['author_name'],
                comment_karma=author_data.get('comment_karma'),
                link_karma=author_data.get('link_karma'),
                created_utc=author_data.get('created_utc')
            )
            self.stats['authors_processed'] += 1
            return author.author_id
        except Exception as e:
            logger.error(f"Error processing author {author_data.get('author_name')}: {e}")
            self.stats['errors'].append(f"Author error: {e}")
            # Rollback the session to clear the error state
            try:
                self.session.rollback()
            except:
                pass
            return None

    def _process_post(self, post_data: Dict[str, Any]) -> Optional[str]:
        """Process and store post information"""
        try:
            # Process author first if exists
            author_id = None
            if post_data.get('author_name') and post_data['author_name'] != '[deleted]':
                author_id = self._process_author({
                    'author_id': post_data.get('author_id'),
                    'author_name': post_data.get('author_name')
                })

            post = self.post_repo.create_or_update(
                post_id=post_data['post_id'],
                subreddit_id=post_data['subreddit_id'],
                author_id=author_id,
                title=post_data['title'],
                body=post_data.get('body'),
                url=post_data.get('url'),
                score=post_data.get('score'),
                num_comments=post_data.get('num_comments'),
                created_utc=post_data.get('created_utc'),
                last_scraped_utc=datetime.utcnow()
            )
            self.stats['posts_processed'] += 1
            return post.post_id
        except Exception as e:
            logger.error(f"Error processing post {post_data.get('post_id')}: {e}")
            self.stats['errors'].append(f"Post error: {e}")
            # Rollback the session to clear the error state
            try:
                self.session.rollback()
            except:
                pass
            return None

    def _process_comment(self, comment_data: Dict[str, Any]) -> Optional[str]:
        """Process and store comment information"""
        try:
            # Process author first if exists
            author_id = None
            if comment_data.get('author_name') and comment_data['author_name'] != '[deleted]':
                author_id = self._process_author({
                    'author_id': comment_data.get('author_id'),
                    'author_name': comment_data.get('author_name')
                })

            comment = self.comment_repo.create_or_update(
                comment_id=comment_data['comment_id'],
                post_id=comment_data['post_id'],
                author_id=author_id,
                parent_id=comment_data.get('parent_id'),
                body=comment_data['body'],
                score=comment_data.get('score'),
                created_utc=comment_data.get('created_utc'),
                last_scraped_utc=datetime.utcnow()
            )
            self.stats['comments_processed'] += 1
            return comment.comment_id
        except Exception as e:
            logger.error(f"Error processing comment {comment_data.get('comment_id')}: {e}")
            self.stats['errors'].append(f"Comment error: {e}")
            # Rollback the session to clear the error state
            try:
                self.session.rollback()
            except:
                pass
            return None

    def fetch_and_store_subreddit(
        self,
        subreddit_name: str,
        start_time: datetime,
        end_time: datetime,
        fetch_comments: bool = True,
        sort_by: str = 'new'
    ) -> Dict[str, Any]:
        """
        Fetch and store all posts and comments from a subreddit within a timeframe

        Args:
            subreddit_name: Name of the subreddit
            start_time: Start of the timeframe
            end_time: End of the timeframe
            fetch_comments: Whether to fetch comments
            sort_by: Sort method for posts

        Returns:
            Statistics dictionary
        """
        logger.info(f"Starting fetch for r/{subreddit_name} from {start_time} to {end_time}")

        # Reset stats
        self.stats = {
            'subreddits_processed': 0,
            'posts_processed': 0,
            'comments_processed': 0,
            'authors_processed': 0,
            'errors': []
        }

        try:
            # Fetch and store subreddit info
            subreddit_info = self.fetcher.fetch_subreddit_info(subreddit_name)
            subreddit_id = self._process_subreddit(subreddit_info)

            if not subreddit_id:
                logger.error(f"Failed to process subreddit {subreddit_name}")
                return self.stats

            # Fetch posts and comments with retry on rate limits
            attempts = 0
            max_attempts = 5
            while True:
                try:
                    posts, comments = self.fetcher.fetch_posts_and_comments(
                        subreddit_name, start_time, end_time, fetch_comments, sort_by
                    )
                    break
                except NotFound:
                    logger.warning(
                        f"Subreddit r/{subreddit_name} returned 404 during fetch; skipping posts"
                    )
                    posts, comments = [], []
                    break
                except ResponseException as e:
                    if getattr(e, 'response', None) and e.response.status_code == 404:
                        logger.warning(
                            f"Received 404 response when fetching r/{subreddit_name}; continuing with no posts"
                        )
                        posts, comments = [], []
                        break
                    error_message = str(e).lower()
                    if '429' in error_message or 'too many requests' in error_message:
                        attempts += 1
                        if attempts >= max_attempts:
                            logger.error(
                                f"Rate limit exceeded for r/{subreddit_name} after {attempts} attempts"
                            )
                            self.session.rollback()
                            self.stats['errors'].append(
                                f"Rate limit: r/{subreddit_name} after {attempts} attempts"
                            )
                            return self.stats
                        wait_seconds = max(self.fetcher.config.rate_limit_delay, 1.0) * (2 ** attempts)
                        logger.warning(
                            f"Rate limited when fetching r/{subreddit_name}. "
                            f"Retrying in {wait_seconds:.1f} seconds (attempt {attempts}/{max_attempts})"
                        )
                        time.sleep(wait_seconds)
                        continue
                    else:
                        raise
                except Exception as e:
                    error_message = str(e).lower()
                    if '404' in error_message or 'received 404 http response' in error_message:
                        logger.warning(
                            f"Encountered 404 while fetching r/{subreddit_name}: {e}. Continuing with no posts"
                        )
                        posts, comments = [], []
                        break
                    if '429' in error_message or 'too many requests' in error_message:
                        attempts += 1
                        if attempts >= max_attempts:
                            logger.error(
                                f"Rate limit exceeded for r/{subreddit_name} after {attempts} attempts"
                            )
                            self.session.rollback()
                            self.stats['errors'].append(
                                f"Rate limit: r/{subreddit_name} after {attempts} attempts"
                            )
                            return self.stats
                        wait_seconds = max(self.fetcher.config.rate_limit_delay, 1.0) * (2 ** attempts)
                        logger.warning(
                            f"Rate limited when fetching r/{subreddit_name}. "
                            f"Retrying in {wait_seconds:.1f} seconds (attempt {attempts}/{max_attempts})"
                        )
                        time.sleep(wait_seconds)
                        continue
                    else:
                        raise

            # Process posts
            for post_data in posts:
                # Ensure subreddit_id is set
                post_data['subreddit_id'] = subreddit_id
                self._process_post(post_data)

            # Process comments
            for comment_data in comments:
                self._process_comment(comment_data)

            # Commit all changes
            self.session.commit()

            # Run sentiment analysis if enabled
            if self.enable_sentiment_analysis and self.sentiment_pipeline:
                logger.info("Running sentiment analysis on new content")
                try:
                    sentiment_stats = self.sentiment_pipeline.analyze_subreddit(
                        subreddit_name,
                        hours_back=0,  # Only analyze new content
                        force=False
                    )
                    self.stats['sentiment_analysis'] = sentiment_stats
                    logger.info(f"Sentiment analysis completed: {sentiment_stats}")
                except Exception as e:
                    logger.error(f"Error in sentiment analysis: {e}")
                    self.stats['errors'].append(f"Sentiment error: {e}")

        except Exception as e:
            logger.error(f"Error in fetch_and_store_subreddit: {e}")
            self.stats['errors'].append(f"Pipeline error: {e}")
            self.session.rollback()

        logger.info(f"Completed: {self.stats}")
        return self.stats

    def fetch_multiple_subreddits(
        self,
        subreddit_names: List[str],
        start_time: datetime,
        end_time: datetime,
        fetch_comments: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch and store data from multiple subreddits

        Args:
            subreddit_names: List of subreddit names
            start_time: Start of the timeframe
            end_time: End of the timeframe
            fetch_comments: Whether to fetch comments

        Returns:
            Dictionary mapping subreddit names to their statistics
        """
        all_stats = {}

        for subreddit_name in subreddit_names:
            logger.info(f"Processing r/{subreddit_name}")
            stats = self.fetch_and_store_subreddit(
                subreddit_name, start_time, end_time, fetch_comments
            )
            all_stats[subreddit_name] = stats

        return all_stats

    def update_stale_posts(self, hours_old: int = 6) -> Dict[str, Any]:
        """
        Update posts that haven't been scraped recently

        Args:
            hours_old: Hours since last scrape to consider stale

        Returns:
            Statistics dictionary
        """
        logger.info(f"Updating posts older than {hours_old} hours")

        # Reset stats
        self.stats = {
            'posts_updated': 0,
            'comments_added': 0,
            'errors': []
        }

        try:
            # Get posts that need updating
            stale_posts = self.post_repo.needs_update(hours=hours_old)

            for post in stale_posts:
                try:
                    # Extract the post ID without prefix
                    post_id = post.post_id[3:] if post.post_id.startswith('t3_') else post.post_id

                    # Fetch updated post data
                    submission = self.fetcher.reddit.submission(id=post_id)
                    post_data = self.fetcher._extract_post_data(submission)

                    # Update post
                    self.post_repo.create_or_update(
                        post_id=post.post_id,
                        score=post_data['score'],
                        num_comments=post_data['num_comments'],
                        last_scraped_utc=datetime.utcnow()
                    )
                    self.stats['posts_updated'] += 1

                    # Fetch new comments
                    comments = self.fetcher.fetch_comments_for_post(post_id)
                    for comment_data in comments:
                        if self._process_comment(comment_data):
                            self.stats['comments_added'] += 1

                except Exception as e:
                    logger.error(f"Error updating post {post.post_id}: {e}")
                    self.stats['errors'].append(f"Update error for {post.post_id}: {e}")

            self.session.commit()

        except Exception as e:
            logger.error(f"Error in update_stale_posts: {e}")
            self.stats['errors'].append(f"Update pipeline error: {e}")
            self.session.rollback()

        logger.info(f"Update completed: {self.stats}")
        return self.stats

    def stream_and_store(
        self,
        subreddit_name: str,
        fetch_comments: bool = True
    ):
        """
        Stream new posts from a subreddit and store them in real-time

        Args:
            subreddit_name: Name of the subreddit to stream
            fetch_comments: Whether to fetch comments for new posts
        """
        # First, fetch and store subreddit info
        subreddit_info = self.fetcher.fetch_subreddit_info(subreddit_name)
        subreddit_id = self._process_subreddit(subreddit_info)

        def process_callback(post_data: Dict[str, Any], comments: List[Dict[str, Any]]):
            """Callback to process each new post"""
            try:
                # Ensure subreddit_id is set
                post_data['subreddit_id'] = subreddit_id

                # Process post
                post_id = self._process_post(post_data)

                if post_id:
                    logger.info(f"Stored new post: {post_data['title'][:50]}...")

                    # Process comments
                    for comment_data in comments:
                        self._process_comment(comment_data)

                    self.session.commit()
                    logger.info(f"Stored {len(comments)} comments for post {post_id}")

            except Exception as e:
                logger.error(f"Error processing stream data: {e}")
                self.session.rollback()

        # Start streaming
        logger.info(f"Starting stream for r/{subreddit_name}")
        self.fetcher.stream_subreddit_posts(subreddit_name, process_callback, fetch_comments)

    def close(self):
        """Close database session and sentiment pipeline"""
        self.session.close()
        if self.sentiment_pipeline:
            self.sentiment_pipeline.close()
