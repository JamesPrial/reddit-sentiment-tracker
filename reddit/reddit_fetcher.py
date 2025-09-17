"""
Reddit data fetcher using PRAW (Python Reddit API Wrapper)
"""
import praw
from praw.models import Subreddit as PrawSubreddit, Submission, Comment as PrawComment, MoreComments
from prawcore.exceptions import (
    ResponseException,
    RequestException,
    NotFound,
    PrawcoreException,
)
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Generator, Tuple
import logging
import time
from dataclasses import dataclass
from collections import deque

from .enums import (
    PostDataKey,
    CommentDataKey,
    AuthorDataKey,
    SubredditDataKey,
    RedditMarker,
    RedditAttribute,
    RedditIdPrefix,
    SortMethod,
    TimeFilter,
    ExceptionAttr,
    HttpHeader,
    HttpStatus,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_RETRIES = 5
BASE_WAIT = 5
EXP_BASE = 2




@dataclass
class FetchConfig:
    """Configuration for fetching Reddit data"""
    max_posts: Optional[int] = None  # None means fetch all posts
    max_comments_per_post: Optional[int] = None  # None means fetch all comments
    max_comment_depth: int = 100  # Increased depth for deep threads
    include_deleted: bool = False
    include_removed: bool = False
    rate_limit_delay: float = 1.0  # Increased to 1 second to avoid rate limits
    replace_more_limit: Optional[int] = None  # None means replace all "More Comments"


class RedditFetcher:
    """Fetches posts and comments from Reddit using PRAW"""

    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        """
        Initialize Reddit API connection

        Args:
            client_id: Reddit API client ID
            client_secret: Reddit API client secret
            user_agent: User agent string for API requests
        """
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        self.config = FetchConfig()
        self._last_request_time = 0

    def _rate_limit(self):
        """Implement rate limiting between API calls"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.config.rate_limit_delay:
            time.sleep(self.config.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    def _retry_delay_seconds(self, exc: Exception, fallback: float) -> Optional[float]:
        """Determine how long to wait after encountering a 429."""
        sleep_time = getattr(exc, ExceptionAttr.SLEEP_TIME.value, None)
        if sleep_time:
            try:
                return float(sleep_time)
            except (TypeError, ValueError):
                pass

        response = getattr(exc, ExceptionAttr.RESPONSE.value, None)
        if response is not None:
            status_code = getattr(response, ExceptionAttr.STATUS_CODE.value, None)
            if status_code != HttpStatus.RATE_LIMITED.value:
                return None

            headers = getattr(response, "headers", None)
            retry_after = None
            if headers:
                retry_after = headers.get(HttpHeader.RETRY_AFTER.value) or headers.get(HttpHeader.RETRY_AFTER_LOWER.value)

            if retry_after is not None:
                try:
                    return float(retry_after)
                except (TypeError, ValueError):
                    logger.debug("Invalid Retry-After header value: %s", retry_after)

            return float(fallback)

        message = str(exc)
        if str(HttpStatus.RATE_LIMITED.value) in message:
            return float(fallback)

        return None

    def _is_not_found_error(self, exc: Exception) -> bool:
        """Return True when the exception represents a 404/Not Found."""
        if isinstance(exc, NotFound):
            return True

        response = getattr(exc, ExceptionAttr.RESPONSE.value, None)
        if response is not None and getattr(response, ExceptionAttr.STATUS_CODE.value, None) == HttpStatus.NOT_FOUND.value:
            return True

        status_code = getattr(exc, ExceptionAttr.STATUS_CODE.value, None)
        if status_code == HttpStatus.NOT_FOUND.value:
            return True

        if isinstance(exc, RequestException):
            resp = getattr(exc, ExceptionAttr.RESPONSE.value, None)
            if resp is not None and getattr(resp, ExceptionAttr.STATUS_CODE.value, None) == HttpStatus.NOT_FOUND.value:
                return True

        message = str(exc)
        return str(HttpStatus.NOT_FOUND.value) in message

    def _handle_rate_limit_retry(self, func, *args, **kwargs):
        """Handle rate limiting with exponential backoff"""
        max_retries = MAX_RETRIES
        base_wait = BASE_WAIT

        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except (PrawcoreException, RequestException) as exc:
                wait_time = self._retry_delay_seconds(exc, base_wait * (EXP_BASE ** attempt))
            except Exception as exc:
                wait_time = self._retry_delay_seconds(exc, base_wait * (EXP_BASE ** attempt))

            if wait_time is None:
                raise

            logger.warning(
                "Rate limited (%s). Waiting %s seconds (attempt %s/%s)",
                HttpStatus.RATE_LIMITED.value,
                wait_time,
                attempt + 1,
                max_retries,
            )
            time.sleep(wait_time)

        raise Exception(f"Max retries ({max_retries}) exceeded for rate limiting")

    def _extract_post_data(self, submission: Submission) -> Dict[str, Any]:
        """Extract relevant data from a Reddit submission"""
        return {
            PostDataKey.POST_ID.value: f"{RedditIdPrefix.POST.value}{submission.id}",
            PostDataKey.SUBREDDIT_ID.value: submission.subreddit.id if hasattr(submission.subreddit, RedditAttribute.ID.value) else None,
            PostDataKey.SUBREDDIT_NAME.value: submission.subreddit.display_name,
            PostDataKey.AUTHOR_ID.value: (
                f"{RedditIdPrefix.AUTHOR.value}{submission.author.id}"
                if submission.author and hasattr(submission.author, RedditAttribute.ID.value)
                else None
            ),
            PostDataKey.AUTHOR_NAME.value: submission.author.name if submission.author else RedditMarker.DELETED.value,
            PostDataKey.TITLE.value: submission.title,
            PostDataKey.BODY.value: submission.selftext if submission.is_self else None,
            PostDataKey.URL.value: f'https://reddit.com{submission.permalink}',
            PostDataKey.SCORE.value: submission.score,
            PostDataKey.NUM_COMMENTS.value: submission.num_comments,
            PostDataKey.CREATED_UTC.value: datetime.utcfromtimestamp(submission.created_utc),
            PostDataKey.IS_DELETED.value: submission.author is None,
            PostDataKey.IS_REMOVED.value: submission.selftext == RedditMarker.REMOVED.value if submission.is_self else False,
            PostDataKey.AWARDS.value: len(submission.all_awardings) if hasattr(submission, RedditAttribute.ALL_AWARDINGS.value) else 0,
            PostDataKey.UPVOTE_RATIO.value: submission.upvote_ratio if hasattr(submission, RedditAttribute.UPVOTE_RATIO.value) else None,
            PostDataKey.IS_STICKIED.value: submission.stickied,
            PostDataKey.IS_LOCKED.value: submission.locked,
            PostDataKey.IS_NSFW.value: submission.over_18
        }

    def _extract_comment_data(self, comment: PrawComment, post_id: str) -> Dict[str, Any]:
        """Extract relevant data from a Reddit comment"""
        return {
            CommentDataKey.COMMENT_ID.value: f"{RedditIdPrefix.COMMENT.value}{comment.id}",
            CommentDataKey.POST_ID.value: post_id,
            CommentDataKey.AUTHOR_ID.value: (
                f"{RedditIdPrefix.AUTHOR.value}{comment.author.id}"
                if comment.author and hasattr(comment.author, RedditAttribute.ID.value)
                else None
            ),
            CommentDataKey.AUTHOR_NAME.value: comment.author.name if comment.author else RedditMarker.DELETED.value,
            CommentDataKey.PARENT_ID.value: comment.parent_id,
            CommentDataKey.BODY.value: comment.body,
            CommentDataKey.SCORE.value: comment.score,
            CommentDataKey.CREATED_UTC.value: datetime.utcfromtimestamp(comment.created_utc),
            CommentDataKey.IS_DELETED.value: comment.author is None,
            CommentDataKey.IS_REMOVED.value: comment.body == RedditMarker.REMOVED.value,
            CommentDataKey.AWARDS.value: len(comment.all_awardings) if hasattr(comment, RedditAttribute.ALL_AWARDINGS.value) else 0,
            CommentDataKey.IS_STICKIED.value: comment.stickied,
            CommentDataKey.DEPTH.value: comment.depth if hasattr(comment, RedditAttribute.DEPTH.value) else 0
        }

    def _extract_author_data(self, author) -> Optional[Dict[str, Any]]:
        """Extract relevant data from a Reddit author"""
        if not author or author.name == RedditMarker.DELETED.value:
            return None

        try:
            return {
                AuthorDataKey.AUTHOR_ID.value: f"{RedditIdPrefix.AUTHOR.value}{author.id}",
                AuthorDataKey.NAME.value: author.name,
                AuthorDataKey.COMMENT_KARMA.value: author.comment_karma,
                AuthorDataKey.LINK_KARMA.value: author.link_karma,
                AuthorDataKey.CREATED_UTC.value: datetime.utcfromtimestamp(author.created_utc),
                AuthorDataKey.IS_MOD.value: author.is_mod if hasattr(author, RedditAttribute.IS_MOD.value) else False,
                AuthorDataKey.IS_GOLD.value: author.is_gold if hasattr(author, RedditAttribute.IS_GOLD.value) else False,
                AuthorDataKey.VERIFIED.value: author.verified if hasattr(author, RedditAttribute.VERIFIED.value) else False
            }
        except Exception as e:
            logger.warning(f"Could not fetch author data for {author.name}: {e}")
            return None

    def fetch_subreddit_info(self, subreddit_name: str) -> Dict[str, Any]:
        """Fetch information about a subreddit with rate limit handling"""
        self._rate_limit()

        def _fetch():
            subreddit = self.reddit.subreddit(subreddit_name)
            return {
                SubredditDataKey.SUBREDDIT_ID.value: f"{RedditIdPrefix.SUBREDDIT.value}{subreddit.id}",
                SubredditDataKey.NAME.value: subreddit.display_name,
                SubredditDataKey.SUBSCRIBER_COUNT.value: subreddit.subscribers,
                SubredditDataKey.CREATED_UTC.value: datetime.utcfromtimestamp(subreddit.created_utc),
                SubredditDataKey.DESCRIPTION.value: subreddit.public_description,
                SubredditDataKey.IS_NSFW.value: subreddit.over18
            }

        return self._handle_rate_limit_retry(_fetch)

    def fetch_posts_by_timeframe(
        self,
        subreddit_name: str,
        start_time: datetime,
        end_time: datetime,
        sort_by: str = SortMethod.NEW.value
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Fetch posts from a subreddit within a specific timeframe

        Args:
            subreddit_name: Name of the subreddit
            start_time: Start of the timeframe
            end_time: End of the timeframe
            sort_by: Sort method ('new', 'hot', 'top', 'rising')

        Yields:
            Post data dictionaries
        """
        subreddit = self.reddit.subreddit(subreddit_name)
        posts_fetched = 0

        # Convert datetimes to timestamps
        start_timestamp = start_time.timestamp()
        end_timestamp = end_time.timestamp()

        logger.info(f"Fetching posts from r/{subreddit_name} between {start_time} and {end_time}")

        def _select_post_stream():
            if sort_by == SortMethod.NEW.value:
                return subreddit.new(limit=None)
            if sort_by == SortMethod.HOT.value:
                return subreddit.hot(limit=None)
            if sort_by == SortMethod.TOP.value:
                return subreddit.top(limit=None, time_filter=TimeFilter.ALL.value)
            if sort_by == SortMethod.RISING.value:
                return subreddit.rising(limit=None)
            raise ValueError(f"Invalid sort_by value: {sort_by}")

        post_stream = self._handle_rate_limit_retry(_select_post_stream)

        for submission in post_stream:
            self._rate_limit()

            def _touch_submission():
                submission.created_utc
                submission.author

            try:
                self._handle_rate_limit_retry(_touch_submission)
            except Exception as e:
                if self._is_not_found_error(e):
                    logger.warning(
                        "Skipping a submission in r/%s: %s",
                        subreddit_name,
                        e,
                    )
                    continue
                raise

            # Check if post is within timeframe
            if submission.created_utc < start_timestamp:
                # Since posts are sorted by time, we can stop here
                break

            if submission.created_utc > end_timestamp:
                continue

            # Skip deleted/removed posts if configured
            if not self.config.include_deleted and submission.author is None:
                continue
            if not self.config.include_removed and submission.selftext == RedditMarker.REMOVED.value:
                continue

            post_data = self._extract_post_data(submission)
            posts_fetched += 1

            yield post_data

            if self.config.max_posts and posts_fetched >= self.config.max_posts:
                logger.info(f"Reached max posts limit ({self.config.max_posts})")
                break

        logger.info(f"Fetched {posts_fetched} posts from r/{subreddit_name}")

    def fetch_comments_for_post(
        self,
        post_id: str,
        max_comments: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch all comments for a specific post

        Args:
            post_id: Reddit post ID (with or without 't3_' prefix)
            max_comments: Maximum number of comments to fetch

        Returns:
            List of comment data dictionaries
        """
        # Clean post_id
        post_prefix = RedditIdPrefix.POST.value
        if post_id.startswith(post_prefix):
            post_id = post_id[len(post_prefix):]

        self._rate_limit()

        def _load_submission():
            submission = self.reddit.submission(id=post_id)
            replace_limit = (
                self.config.replace_more_limit
                if self.config.replace_more_limit is not None
                else None
            )
            submission.comments.replace_more(limit=replace_limit)
            return submission

        try:
            submission = self._handle_rate_limit_retry(_load_submission)
        except Exception as exc:
            if self._is_not_found_error(exc):
                logger.warning(
                    "Post %s not found (%s). Skipping comment fetch.",
                    post_id,
                    HttpStatus.NOT_FOUND.value,
                )
                return []
            raise

        comments = []
        max_to_fetch = max_comments if max_comments is not None else self.config.max_comments_per_post

        # Use BFS to traverse comment tree
        comment_queue = deque(submission.comments[:])

        while comment_queue and (max_to_fetch is None or len(comments) < max_to_fetch):
            comment = comment_queue.popleft()

            # Skip MoreComments objects that couldn't be replaced
            if isinstance(comment, MoreComments):
                continue

            # Skip deleted/removed comments if configured
            if not self.config.include_deleted and comment.author is None:
                continue
            if not self.config.include_removed and comment.body == RedditMarker.REMOVED.value:
                continue

            comment_data = self._extract_comment_data(comment, f"{RedditIdPrefix.POST.value}{post_id}")
            comments.append(comment_data)

            # Add replies to queue if within depth limit
            if comment_data[CommentDataKey.DEPTH.value] < self.config.max_comment_depth:
                comment_queue.extend(comment.replies)

        logger.info(f"Fetched {len(comments)} comments for post {post_id}")
        return comments

    def fetch_posts_and_comments(
        self,
        subreddit_name: str,
        start_time: datetime,
        end_time: datetime,
        fetch_comments: bool = True,
        sort_by: str = SortMethod.NEW.value
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Fetch both posts and their comments from a subreddit within a timeframe

        Args:
            subreddit_name: Name of the subreddit
            start_time: Start of the timeframe
            end_time: End of the timeframe
            fetch_comments: Whether to fetch comments for each post
            sort_by: Sort method for posts

        Returns:
            Tuple of (posts_list, comments_list)
        """
        posts = []
        all_comments = []

        # Fetch posts
        for post_data in self.fetch_posts_by_timeframe(subreddit_name, start_time, end_time, sort_by):
            posts.append(post_data)

            # Fetch comments for this post if requested
            if fetch_comments and post_data[PostDataKey.NUM_COMMENTS.value] > 0:
                try:
                    comments = self.fetch_comments_for_post(post_data[PostDataKey.POST_ID.value])
                    all_comments.extend(comments)
                except Exception as exc:
                    if self._is_not_found_error(exc):
                        logger.warning(
                            "Skipping comments for post %s (%s response)",
                            post_data[PostDataKey.POST_ID.value],
                            HttpStatus.NOT_FOUND.value,
                        )
                    else:
                        logger.error(
                            "Error fetching comments for post %s: %s",
                            post_data[PostDataKey.POST_ID.value],
                            exc,
                        )

        logger.info(f"Total fetched: {len(posts)} posts and {len(all_comments)} comments")
        return posts, all_comments

    def fetch_user_activity(
        self,
        username: str,
        start_time: datetime,
        end_time: datetime
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Fetch a user's posts and comments within a timeframe

        Args:
            username: Reddit username
            start_time: Start of the timeframe
            end_time: End of the timeframe

        Returns:
            Tuple of (posts_list, comments_list)
        """
        user = self.reddit.redditor(username)
        posts = []
        comments = []

        start_timestamp = start_time.timestamp()
        end_timestamp = end_time.timestamp()

        # Fetch user's posts
        for submission in user.submissions.new(limit=None):
            self._rate_limit()

            if submission.created_utc < start_timestamp:
                break
            if submission.created_utc > end_timestamp:
                continue

            posts.append(self._extract_post_data(submission))

        # Fetch user's comments
        for comment in user.comments.new(limit=None):
            self._rate_limit()

            if comment.created_utc < start_timestamp:
                break
            if comment.created_utc > end_timestamp:
                continue

            # Get the post ID from the comment's link_id
            post_id = comment.link_id if hasattr(comment, RedditAttribute.LINK_ID.value) else None
            comments.append(self._extract_comment_data(comment, post_id))

        logger.info(f"Fetched {len(posts)} posts and {len(comments)} comments for user {username}")
        return posts, comments

    def fetch_multiple_subreddits(
        self,
        subreddit_names: List[str],
        start_time: datetime,
        end_time: datetime,
        fetch_comments: bool = True
    ) -> Dict[str, Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]:
        """
        Fetch data from multiple subreddits

        Args:
            subreddit_names: List of subreddit names
            start_time: Start of the timeframe
            end_time: End of the timeframe
            fetch_comments: Whether to fetch comments

        Returns:
            Dictionary mapping subreddit names to (posts, comments) tuples
        """
        results = {}

        for subreddit_name in subreddit_names:
            try:
                logger.info(f"Fetching data from r/{subreddit_name}")
                posts, comments = self.fetch_posts_and_comments(
                    subreddit_name, start_time, end_time, fetch_comments
                )
                results[subreddit_name] = (posts, comments)
            except Exception as e:
                logger.error(f"Error fetching from r/{subreddit_name}: {e}")
                results[subreddit_name] = ([], [])

        return results

    def stream_subreddit_posts(
        self,
        subreddit_name: str,
        process_callback: callable,
        fetch_comments: bool = True
    ):
        """
        Stream new posts from a subreddit in real-time

        Args:
            subreddit_name: Name of the subreddit
            process_callback: Function to call for each new post
            fetch_comments: Whether to fetch comments for new posts
        """
        subreddit = self.reddit.subreddit(subreddit_name)

        logger.info(f"Starting stream for r/{subreddit_name}")

        for submission in subreddit.stream.submissions(skip_existing=True):
            try:
                self._rate_limit()

                post_data = self._extract_post_data(submission)
                comments = []

                if fetch_comments and submission.num_comments > 0:
                    comments = self.fetch_comments_for_post(submission.id)

                # Call the processing callback
                process_callback(post_data, comments)

            except Exception as e:
                logger.error(f"Error processing submission {submission.id}: {e}")
                continue
