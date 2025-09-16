"""Enum definitions for Reddit data keys and related constants."""
from enum import Enum


class PostDataKey(str, Enum):
    POST_ID = "post_id"
    SUBREDDIT_ID = "subreddit_id"
    SUBREDDIT_NAME = "subreddit_name"
    AUTHOR_ID = "author_id"
    AUTHOR_NAME = "author_name"
    TITLE = "title"
    BODY = "body"
    URL = "url"
    SCORE = "score"
    NUM_COMMENTS = "num_comments"
    CREATED_UTC = "created_utc"
    IS_DELETED = "is_deleted"
    IS_REMOVED = "is_removed"
    AWARDS = "awards"
    UPVOTE_RATIO = "upvote_ratio"
    IS_STICKIED = "is_stickied"
    IS_LOCKED = "is_locked"
    IS_NSFW = "is_nsfw"


class CommentDataKey(str, Enum):
    COMMENT_ID = "comment_id"
    POST_ID = "post_id"
    AUTHOR_ID = "author_id"
    AUTHOR_NAME = "author_name"
    PARENT_ID = "parent_id"
    BODY = "body"
    SCORE = "score"
    CREATED_UTC = "created_utc"
    IS_DELETED = "is_deleted"
    IS_REMOVED = "is_removed"
    AWARDS = "awards"
    IS_STICKIED = "is_stickied"
    DEPTH = "depth"


class AuthorDataKey(str, Enum):
    AUTHOR_ID = "author_id"
    NAME = "name"
    COMMENT_KARMA = "comment_karma"
    LINK_KARMA = "link_karma"
    CREATED_UTC = "created_utc"
    IS_MOD = "is_mod"
    IS_GOLD = "is_gold"
    VERIFIED = "verified"


class SubredditDataKey(str, Enum):
    SUBREDDIT_ID = "subreddit_id"
    NAME = "name"
    SUBSCRIBER_COUNT = "subscriber_count"
    CREATED_UTC = "created_utc"
    DESCRIPTION = "description"
    IS_NSFW = "is_nsfw"


class RedditMarker(str, Enum):
    DELETED = "[deleted]"
    REMOVED = "[removed]"


class RedditAttribute(str, Enum):
    ID = "id"
    ALL_AWARDINGS = "all_awardings"
    UPVOTE_RATIO = "upvote_ratio"
    DEPTH = "depth"
    LINK_ID = "link_id"
    IS_MOD = "is_mod"
    IS_GOLD = "is_gold"
    VERIFIED = "verified"


class RedditIdPrefix(str, Enum):
    COMMENT = "t1_"
    POST = "t3_"
    AUTHOR = "t2_"
    SUBREDDIT = "t5_"


class SortMethod(str, Enum):
    NEW = "new"
    HOT = "hot"
    TOP = "top"
    RISING = "rising"


class TimeFilter(str, Enum):
    ALL = "all"


class ExceptionAttr(str, Enum):
    SLEEP_TIME = "sleep_time"
    RESPONSE = "response"
    STATUS_CODE = "status_code"


class HttpHeader(str, Enum):
    RETRY_AFTER = "Retry-After"
    RETRY_AFTER_LOWER = "retry-after"


class HttpStatus(int, Enum):
    RATE_LIMITED = 429
    NOT_FOUND = 404
