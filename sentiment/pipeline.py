"""
Batch processing pipeline for sentiment analysis
"""
import json
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

from data import DatabaseConnection, Post, Comment, PostRepository, CommentRepository
from .analyzer import SentimentAnalyzer, SentimentResult, get_analyzer
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AnalysisStats:
    """Statistics for analysis run"""
    posts_analyzed: int = 0
    comments_analyzed: int = 0
    posts_skipped: int = 0
    comments_skipped: int = 0
    errors: List[str] = None
    start_time: datetime = None
    end_time: datetime = None
    total_api_calls: int = 0
    cache_hits: int = 0

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.start_time is None:
            self.start_time = datetime.utcnow()

    def duration(self) -> timedelta:
        """Calculate duration of analysis"""
        end = self.end_time or datetime.utcnow()
        return end - self.start_time

    def __str__(self) -> str:
        return (
            f"Analysis Stats:\n"
            f"  Posts: {self.posts_analyzed} analyzed, {self.posts_skipped} skipped\n"
            f"  Comments: {self.comments_analyzed} analyzed, {self.comments_skipped} skipped\n"
            f"  API calls: {self.total_api_calls}, Cache hits: {self.cache_hits}\n"
            f"  Duration: {self.duration()}\n"
            f"  Errors: {len(self.errors)}"
        )


class SentimentPipeline:
    """Pipeline for batch sentiment analysis"""

    def __init__(
        self,
        db_connection_string: str,
        analyzer: Optional[SentimentAnalyzer] = None,
        batch_size: int = 10,
        cache_enabled: bool = True
    ):
        """
        Initialize sentiment pipeline

        Args:
            db_connection_string: Database connection string
            analyzer: Sentiment analyzer to use (defaults to API analyzer)
            batch_size: Number of items to analyze in one batch
            cache_enabled: Whether to cache results
        """
        self.db = DatabaseConnection(db_connection_string)
        self.session = self.db.get_session()
        self.post_repo = PostRepository(self.session)
        self.comment_repo = CommentRepository(self.session)

        self.analyzer = analyzer or get_analyzer(use_api=True)
        self.batch_size = batch_size
        self.cache_enabled = cache_enabled

        # In-memory cache for this session
        self.cache = {} if cache_enabled else None

        self.stats = AnalysisStats()

    def _get_cache_key(self, text: str, item_type: str) -> str:
        """Generate cache key for text"""
        # Simple hash-based key
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()[:16]
        return f"{item_type}_{text_hash}"

    def _should_analyze(self, item: Any, force: bool = False) -> bool:
        """Check if item needs sentiment analysis"""

        if force:
            return True

        # Check if already has sentiment
        if hasattr(item, 'sentiment_score') and item.sentiment_score is not None:
            return False
        if hasattr(item, 'title_sentiment_score') and item.title_sentiment_score is not None:
            return False

        # Skip deleted items
        if hasattr(item, 'body') and (item.body == '[deleted]' or item.body == '[removed]'):
            return False
        if hasattr(item, 'title') and item.title == '[deleted]':
            return False

        return True

    def analyze_posts(
        self,
        posts: Optional[List[Post]] = None,
        subreddit_id: Optional[str] = None,
        hours_back: Optional[int] = None,
        force: bool = False,
        limit: Optional[int] = None
    ) -> AnalysisStats:
        """
        Analyze sentiment for posts

        Args:
            posts: Specific posts to analyze (if None, query from database)
            subreddit_id: Filter by subreddit
            hours_back: Analyze posts from last N hours
            force: Force re-analysis even if sentiment exists
            limit: Maximum number of posts to analyze

        Returns:
            Analysis statistics
        """
        logger.info("Starting post sentiment analysis")
        self.stats = AnalysisStats()

        # Get posts if not provided
        if posts is None:
            query = self.session.query(Post)

            # Apply filters
            if not force:
                query = query.filter(
                    or_(
                        Post.title_sentiment_score.is_(None),
                        Post.body_sentiment_score.is_(None)
                    )
                )

            if subreddit_id:
                query = query.filter(Post.subreddit_id == subreddit_id)

            if hours_back:
                cutoff = datetime.utcnow() - timedelta(hours=hours_back)
                query = query.filter(Post.created_utc >= cutoff)

            # Order by engagement (score * num_comments) for priority
            query = query.order_by(
                text("(COALESCE(score, 0) * COALESCE(num_comments, 1)) DESC")
            )

            if limit:
                query = query.limit(limit)

            posts = query.all()

        logger.info(f"Found {len(posts)} posts to process")

        # Process in batches
        batch = []
        batch_texts = []

        for post in posts:
            if not self._should_analyze(post, force):
                self.stats.posts_skipped += 1
                continue

            # Check cache
            if self.cache_enabled:
                title_key = self._get_cache_key(post.title, 'post_title')
                if title_key in self.cache:
                    self._apply_post_sentiment(post, self.cache[title_key], is_title=True)
                    self.stats.cache_hits += 1

                if post.body:
                    body_key = self._get_cache_key(post.body, 'post_body')
                    if body_key in self.cache:
                        self._apply_post_sentiment(post, self.cache[body_key], is_title=False)
                        self.stats.cache_hits += 1

                # Skip if both cached
                if title_key in self.cache and (not post.body or body_key in self.cache):
                    self.stats.posts_analyzed += 1
                    continue

            batch.append(post)
            batch_texts.append(post.title)
            if post.body and post.body not in ['[deleted]', '[removed]', '']:
                batch_texts.append(post.body)

            # Process batch when full
            if len(batch_texts) >= self.batch_size:
                self._process_post_batch(batch, batch_texts)
                batch = []
                batch_texts = []

        # Process remaining
        if batch:
            self._process_post_batch(batch, batch_texts)

        # Commit changes
        try:
            self.session.commit()
            logger.info(f"Committed {self.stats.posts_analyzed} post sentiment updates")
        except Exception as e:
            logger.error(f"Error committing post sentiments: {e}")
            self.session.rollback()
            self.stats.errors.append(f"Commit error: {e}")

        self.stats.end_time = datetime.utcnow()
        return self.stats

    def _process_post_batch(self, posts: List[Post], texts: List[str]):
        """Process a batch of posts"""

        try:
            # Analyze batch
            results = self.analyzer.analyze_batch(texts)
            self.stats.total_api_calls += 1

            # Map results back to posts
            result_idx = 0
            for post in posts:
                # Title sentiment
                if result_idx < len(results):
                    title_result = results[result_idx]
                    self._apply_post_sentiment(post, title_result, is_title=True)
                    result_idx += 1

                    # Cache result
                    if self.cache_enabled:
                        cache_key = self._get_cache_key(post.title, 'post_title')
                        self.cache[cache_key] = title_result

                # Body sentiment (if exists)
                if post.body and post.body not in ['[deleted]', '[removed]', '']:
                    if result_idx < len(results):
                        body_result = results[result_idx]
                        self._apply_post_sentiment(post, body_result, is_title=False)
                        result_idx += 1

                        # Cache result
                        if self.cache_enabled:
                            cache_key = self._get_cache_key(post.body, 'post_body')
                            self.cache[cache_key] = body_result

                self.stats.posts_analyzed += 1

        except Exception as e:
            logger.error(f"Error processing post batch: {e}")
            self.stats.errors.append(f"Batch error: {e}")

    def _apply_post_sentiment(self, post: Post, result: SentimentResult, is_title: bool):
        """Apply sentiment result to post"""

        if is_title:
            post.title_sentiment_score = result.sentiment_score
            post.title_sentiment_label = result.sentiment_label
        else:
            post.body_sentiment_score = result.sentiment_score
            post.body_sentiment_label = result.sentiment_label

        # Store brand-aware sentiment
        post.brand_sentiment_score = result.brand_sentiment_score
        post.brand_sentiment_label = result.brand_sentiment_label.value
        post.brand_mentions = result.brand_mentions
        post.sentiment_metadata = {
            'confidence': result.confidence,
            'reasoning': result.reasoning,
            'metadata': result.metadata
        }
        post.sentiment_analyzed_at = datetime.utcnow()

    def analyze_comments(
        self,
        comments: Optional[List[Comment]] = None,
        post_id: Optional[str] = None,
        hours_back: Optional[int] = None,
        force: bool = False,
        limit: Optional[int] = None
    ) -> AnalysisStats:
        """
        Analyze sentiment for comments

        Args:
            comments: Specific comments to analyze
            post_id: Filter by post
            hours_back: Analyze comments from last N hours
            force: Force re-analysis
            limit: Maximum number to analyze

        Returns:
            Analysis statistics
        """
        logger.info("Starting comment sentiment analysis")
        self.stats = AnalysisStats()

        # Get comments if not provided
        if comments is None:
            query = self.session.query(Comment)

            if not force:
                query = query.filter(Comment.sentiment_score.is_(None))

            if post_id:
                query = query.filter(Comment.post_id == post_id)

            if hours_back:
                cutoff = datetime.utcnow() - timedelta(hours=hours_back)
                query = query.filter(Comment.created_utc >= cutoff)

            # Order by score for priority
            query = query.order_by(Comment.score.desc().nullslast())

            if limit:
                query = query.limit(limit)

            comments = query.all()

        logger.info(f"Found {len(comments)} comments to process")

        # Process in batches
        batch = []
        batch_texts = []

        for comment in comments:
            if not self._should_analyze(comment, force):
                self.stats.comments_skipped += 1
                continue

            # Check cache
            if self.cache_enabled:
                cache_key = self._get_cache_key(comment.body, 'comment')
                if cache_key in self.cache:
                    self._apply_comment_sentiment(comment, self.cache[cache_key])
                    self.stats.cache_hits += 1
                    self.stats.comments_analyzed += 1
                    continue

            batch.append(comment)
            batch_texts.append(comment.body)

            # Process batch when full
            if len(batch) >= self.batch_size:
                self._process_comment_batch(batch, batch_texts)
                batch = []
                batch_texts = []

        # Process remaining
        if batch:
            self._process_comment_batch(batch, batch_texts)

        # Commit changes
        try:
            self.session.commit()
            logger.info(f"Committed {self.stats.comments_analyzed} comment sentiment updates")
        except Exception as e:
            logger.error(f"Error committing comment sentiments: {e}")
            self.session.rollback()
            self.stats.errors.append(f"Commit error: {e}")

        self.stats.end_time = datetime.utcnow()
        return self.stats

    def _process_comment_batch(self, comments: List[Comment], texts: List[str]):
        """Process a batch of comments"""

        try:
            # Analyze batch
            results = self.analyzer.analyze_batch(texts)
            self.stats.total_api_calls += 1

            # Apply results
            for comment, result in zip(comments, results):
                self._apply_comment_sentiment(comment, result)

                # Cache result
                if self.cache_enabled:
                    cache_key = self._get_cache_key(comment.body, 'comment')
                    self.cache[cache_key] = result

                self.stats.comments_analyzed += 1

        except Exception as e:
            logger.error(f"Error processing comment batch: {e}")
            self.stats.errors.append(f"Batch error: {e}")

    def _apply_comment_sentiment(self, comment: Comment, result: SentimentResult):
        """Apply sentiment result to comment"""
        comment.sentiment_score = result.sentiment_score
        comment.sentiment_label = result.sentiment_label

        # Store brand-aware sentiment
        comment.brand_sentiment_score = result.brand_sentiment_score
        comment.brand_sentiment_label = result.brand_sentiment_label.value
        comment.brand_mentions = result.brand_mentions
        comment.sentiment_metadata = {
            'confidence': result.confidence,
            'reasoning': result.reasoning,
            'metadata': result.metadata
        }
        comment.sentiment_analyzed_at = datetime.utcnow()

    def analyze_subreddit(
        self,
        subreddit_name: str,
        hours_back: int = 24,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze all content from a subreddit

        Args:
            subreddit_name: Name of subreddit
            hours_back: How far back to analyze
            force: Force re-analysis

        Returns:
            Combined statistics
        """
        logger.info(f"Analyzing sentiment for r/{subreddit_name}")

        # Get subreddit
        from data import SubredditRepository
        subreddit_repo = SubredditRepository(self.session)
        subreddit = subreddit_repo.get_by_name(subreddit_name)

        if not subreddit:
            logger.error(f"Subreddit r/{subreddit_name} not found")
            return {"error": f"Subreddit not found"}

        # Analyze posts
        post_stats = self.analyze_posts(
            subreddit_id=subreddit.subreddit_id,
            hours_back=hours_back,
            force=force
        )

        # Get all posts from timeframe for comment analysis
        cutoff = datetime.utcnow() - timedelta(hours=hours_back)
        posts = self.session.query(Post).filter(
            and_(
                Post.subreddit_id == subreddit.subreddit_id,
                Post.created_utc >= cutoff
            )
        ).all()

        # Analyze comments for each post
        comment_stats = AnalysisStats()
        for post in posts:
            stats = self.analyze_comments(
                post_id=post.post_id,
                force=force
            )
            comment_stats.comments_analyzed += stats.comments_analyzed
            comment_stats.comments_skipped += stats.comments_skipped
            comment_stats.total_api_calls += stats.total_api_calls
            comment_stats.cache_hits += stats.cache_hits
            comment_stats.errors.extend(stats.errors)

        return {
            "subreddit": subreddit_name,
            "post_stats": str(post_stats),
            "comment_stats": str(comment_stats),
            "total_api_calls": post_stats.total_api_calls + comment_stats.total_api_calls,
            "total_cache_hits": post_stats.cache_hits + comment_stats.cache_hits
        }

    def analyze_all_unprocessed(self, limit: int = 1000) -> Dict[str, Any]:
        """
        Analyze all unprocessed content in database

        Args:
            limit: Maximum items to process

        Returns:
            Statistics
        """
        logger.info("Analyzing all unprocessed content")

        # Analyze posts
        post_stats = self.analyze_posts(limit=limit // 2)

        # Analyze comments
        comment_stats = self.analyze_comments(limit=limit // 2)

        return {
            "posts": str(post_stats),
            "comments": str(comment_stats),
            "total_processed": post_stats.posts_analyzed + comment_stats.comments_analyzed
        }

    def get_sentiment_summary(self, subreddit_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get sentiment summary statistics

        Args:
            subreddit_name: Optional subreddit filter

        Returns:
            Summary statistics
        """
        query = """
        WITH post_sentiment AS (
            SELECT
                s.name as subreddit,
                COUNT(*) as total_posts,
                AVG(p.title_sentiment_score) as avg_title_sentiment,
                AVG(p.body_sentiment_score) as avg_body_sentiment,
                COUNT(CASE WHEN p.title_sentiment_label = 'positive' THEN 1 END) as positive_titles,
                COUNT(CASE WHEN p.title_sentiment_label = 'negative' THEN 1 END) as negative_titles,
                COUNT(CASE WHEN p.title_sentiment_label = 'neutral' THEN 1 END) as neutral_titles
            FROM posts p
            JOIN subreddits s ON p.subreddit_id = s.subreddit_id
            WHERE p.title_sentiment_score IS NOT NULL
            {subreddit_filter}
            GROUP BY s.name
        ),
        comment_sentiment AS (
            SELECT
                s.name as subreddit,
                COUNT(*) as total_comments,
                AVG(c.sentiment_score) as avg_comment_sentiment,
                COUNT(CASE WHEN c.sentiment_label = 'positive' THEN 1 END) as positive_comments,
                COUNT(CASE WHEN c.sentiment_label = 'negative' THEN 1 END) as negative_comments,
                COUNT(CASE WHEN c.sentiment_label = 'neutral' THEN 1 END) as neutral_comments
            FROM comments c
            JOIN posts p ON c.post_id = p.post_id
            JOIN subreddits s ON p.subreddit_id = s.subreddit_id
            WHERE c.sentiment_score IS NOT NULL
            {subreddit_filter_comment}
            GROUP BY s.name
        )
        SELECT
            COALESCE(ps.subreddit, cs.subreddit) as subreddit,
            ps.total_posts,
            ps.avg_title_sentiment,
            ps.avg_body_sentiment,
            ps.positive_titles,
            ps.negative_titles,
            ps.neutral_titles,
            cs.total_comments,
            cs.avg_comment_sentiment,
            cs.positive_comments,
            cs.negative_comments,
            cs.neutral_comments
        FROM post_sentiment ps
        FULL OUTER JOIN comment_sentiment cs ON ps.subreddit = cs.subreddit
        ORDER BY COALESCE(ps.total_posts, 0) + COALESCE(cs.total_comments, 0) DESC
        """

        # Add subreddit filter if needed
        if subreddit_name:
            subreddit_filter = f"AND s.name = '{subreddit_name}'"
            subreddit_filter_comment = f"AND s.name = '{subreddit_name}'"
        else:
            subreddit_filter = ""
            subreddit_filter_comment = ""

        query = query.format(
            subreddit_filter=subreddit_filter,
            subreddit_filter_comment=subreddit_filter_comment
        )

        result = self.session.execute(text(query))
        rows = result.fetchall()

        summaries = []
        for row in rows:
            summaries.append({
                "subreddit": row[0],
                "posts": {
                    "total": row[1] or 0,
                    "avg_title_sentiment": float(row[2]) if row[2] else None,
                    "avg_body_sentiment": float(row[3]) if row[3] else None,
                    "positive": row[4] or 0,
                    "negative": row[5] or 0,
                    "neutral": row[6] or 0
                },
                "comments": {
                    "total": row[7] or 0,
                    "avg_sentiment": float(row[8]) if row[8] else None,
                    "positive": row[9] or 0,
                    "negative": row[10] or 0,
                    "neutral": row[11] or 0
                }
            })

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "summaries": summaries
        }

    def close(self):
        """Close database session"""
        self.session.close()
