"""
SQLAlchemy models for Reddit sentiment tracking database
"""
from datetime import datetime
from typing import Optional, List
from sqlalchemy import create_engine, Column, String, Integer, Float, Text, TIMESTAMP, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session
from sqlalchemy.sql import func

Base = declarative_base()


class Subreddit(Base):
    __tablename__ = 'subreddits'

    subreddit_id = Column(String(20), primary_key=True)  # e.g., 't5_2qh1i'
    name = Column(String(255), nullable=False, unique=True)  # e.g., 'wallstreetbets'
    subscriber_count = Column(Integer)
    created_utc = Column(TIMESTAMP)
    date_added = Column(TIMESTAMP, server_default=func.current_timestamp())

    # Relationships
    posts = relationship("Post", back_populates="subreddit", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Subreddit(name='{self.name}', id='{self.subreddit_id}')>"

    @property
    def post_count(self) -> int:
        """Get the number of posts in this subreddit"""
        return len(self.posts) if self.posts else 0

    def get_recent_posts(self, limit: int = 10) -> List["Post"]:
        """Get the most recent posts from this subreddit"""
        return sorted(self.posts, key=lambda p: p.created_utc or datetime.min, reverse=True)[:limit]


class Author(Base):
    __tablename__ = 'authors'

    author_id = Column(String(20), primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    comment_karma = Column(Integer)
    link_karma = Column(Integer)
    created_utc = Column(TIMESTAMP)
    date_added = Column(TIMESTAMP, server_default=func.current_timestamp())

    # Relationships
    posts = relationship("Post", back_populates="author")
    comments = relationship("Comment", back_populates="author")

    def __repr__(self) -> str:
        return f"<Author(name='{self.name}', id='{self.author_id}')>"

    @property
    def total_karma(self) -> int:
        """Calculate total karma (comment + link)"""
        return (self.comment_karma or 0) + (self.link_karma or 0)

    @property
    def activity_count(self) -> int:
        """Get total number of posts and comments"""
        return len(self.posts or []) + len(self.comments or [])


class Post(Base):
    __tablename__ = 'posts'

    post_id = Column(String(20), primary_key=True)  # e.g., 't3_wri09b'
    subreddit_id = Column(String(20), ForeignKey('subreddits.subreddit_id'), nullable=False)
    author_id = Column(String(20), ForeignKey('authors.author_id'))
    title = Column(Text, nullable=False)
    body = Column(Text)
    url = Column(String(2048))
    score = Column(Integer)
    num_comments = Column(Integer)
    created_utc = Column(TIMESTAMP)
    title_sentiment_score = Column(Float)  # -1.0 to 1.0
    title_sentiment_label = Column(String(10))  # 'positive', 'negative', 'neutral'
    body_sentiment_score = Column(Float)
    body_sentiment_label = Column(String(10))
    brand_sentiment_score = Column(Float)  # Brand-aware sentiment score
    brand_sentiment_label = Column(String(30))  # Brand-aware sentiment category
    brand_mentions = Column(JSONB)  # Which brands are mentioned
    sentiment_metadata = Column(JSONB)  # Additional sentiment analysis data
    sentiment_analyzed_at = Column(TIMESTAMP)  # When sentiment was analyzed
    last_scraped_utc = Column(TIMESTAMP)

    # Relationships
    subreddit = relationship("Subreddit", back_populates="posts")
    author = relationship("Author", back_populates="posts")
    comments = relationship("Comment", back_populates="post", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Post(title='{self.title[:50]}...', id='{self.post_id}')>"

    @property
    def overall_sentiment_score(self) -> Optional[float]:
        """Calculate average sentiment score from title and body"""
        scores = [s for s in [self.title_sentiment_score, self.body_sentiment_score] if s is not None]
        return sum(scores) / len(scores) if scores else None

    @property
    def overall_sentiment_label(self) -> str:
        """Determine overall sentiment label based on score"""
        score = self.overall_sentiment_score
        if score is None:
            return "unknown"
        elif score > 0.1:
            return "positive"
        elif score < -0.1:
            return "negative"
        else:
            return "neutral"

    def get_top_comments(self, limit: int = 5) -> List["Comment"]:
        """Get top-scored comments for this post"""
        return sorted(self.comments, key=lambda c: c.score or 0, reverse=True)[:limit]

    @property
    def engagement_rate(self) -> float:
        """Calculate engagement rate (comments per score point)"""
        if self.score and self.score > 0:
            return (self.num_comments or 0) / self.score
        return 0.0


class Comment(Base):
    __tablename__ = 'comments'

    comment_id = Column(String(20), primary_key=True)  # e.g., 't1_ikx45l5'
    post_id = Column(String(20), ForeignKey('posts.post_id'), nullable=False)
    author_id = Column(String(20), ForeignKey('authors.author_id'))
    parent_id = Column(String(20))  # Can be post_id or another comment_id
    body = Column(Text, nullable=False)
    score = Column(Integer)
    created_utc = Column(TIMESTAMP)
    sentiment_score = Column(Float)  # -1.0 to 1.0
    sentiment_label = Column(String(10))
    brand_sentiment_score = Column(Float)  # Brand-aware sentiment score
    brand_sentiment_label = Column(String(30))  # Brand-aware sentiment category
    brand_mentions = Column(JSONB)  # Which brands are mentioned
    sentiment_metadata = Column(JSONB)  # Additional sentiment analysis data
    sentiment_analyzed_at = Column(TIMESTAMP)  # When sentiment was analyzed
    last_scraped_utc = Column(TIMESTAMP)

    # Relationships
    post = relationship("Post", back_populates="comments")
    author = relationship("Author", back_populates="comments")

    def __repr__(self) -> str:
        return f"<Comment(body='{self.body[:50]}...', id='{self.comment_id}')>"

    @property
    def is_top_level(self) -> bool:
        """Check if this comment is a direct reply to the post"""
        return self.parent_id == self.post_id if self.parent_id else False

    def get_sentiment_emoji(self) -> str:
        """Return an emoji based on sentiment"""
        if self.sentiment_label == "positive":
            return "😊"
        elif self.sentiment_label == "negative":
            return "😔"
        elif self.sentiment_label == "neutral":
            return "😐"
        return "❓"


# Database connection helper
class DatabaseConnection:
    """Helper class for database connections and sessions"""

    def __init__(self, connection_string: str):
        """
        Initialize database connection

        Args:
            connection_string: PostgreSQL connection string
                              e.g., 'postgresql://user:password@localhost/reddit'
        """
        self.engine = create_engine(connection_string)
        Base.metadata.create_all(self.engine)

    def get_session(self) -> Session:
        """Get a new database session"""
        from sqlalchemy.orm import sessionmaker
        SessionLocal = sessionmaker(bind=self.engine)
        return SessionLocal()

    def create_tables(self):
        """Create all tables if they don't exist"""
        Base.metadata.create_all(self.engine)

    def drop_tables(self):
        """Drop all tables - use with caution!"""
        Base.metadata.drop_all(self.engine)