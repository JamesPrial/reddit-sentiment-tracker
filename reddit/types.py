"""Typed representations of Reddit data and related constants."""
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


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


@dataclass
class RawSubreddit:
    """Structured data for a subreddit returned by the fetcher."""

    subreddit_id: str
    name: str
    subscriber_count: Optional[int] = None
    created_utc: Optional[datetime] = None
    description: Optional[str] = None
    is_nsfw: Optional[bool] = None


@dataclass
class RawAuthor:
    """Structured data for a Reddit author."""

    author_id: Optional[str]
    author_name: str
    comment_karma: Optional[int] = None
    link_karma: Optional[int] = None
    created_utc: Optional[datetime] = None
    is_mod: Optional[bool] = None
    is_gold: Optional[bool] = None
    verified: Optional[bool] = None


@dataclass
class RawPost:
    """Structured data for a Reddit submission."""

    post_id: str
    subreddit_id: Optional[str]
    subreddit_name: str
    author_id: Optional[str]
    author_name: str
    title: str
    body: Optional[str]
    url: Optional[str]
    score: Optional[int]
    num_comments: Optional[int]
    created_utc: Optional[datetime]
    is_deleted: bool = False
    is_removed: bool = False
    awards: Optional[int] = None
    upvote_ratio: Optional[float] = None
    is_stickied: Optional[bool] = None
    is_locked: Optional[bool] = None
    is_nsfw: Optional[bool] = None


@dataclass
class RawComment:
    """Structured data for a Reddit comment."""

    comment_id: str
    post_id: str
    author_id: Optional[str]
    author_name: str
    parent_id: Optional[str]
    body: str
    score: Optional[int]
    created_utc: Optional[datetime]
    is_deleted: bool = False
    is_removed: bool = False
    awards: Optional[int] = None
    is_stickied: Optional[bool] = None
    depth: int = 0


__all__ = [
    "RedditMarker",
    "RedditAttribute",
    "RedditIdPrefix",
    "SortMethod",
    "TimeFilter",
    "ExceptionAttr",
    "HttpHeader",
    "HttpStatus",
    "RawSubreddit",
    "RawAuthor",
    "RawPost",
    "RawComment",
]
