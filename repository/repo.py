"""
Repository pattern for Reddit sentiment tracking database operations
"""
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import desc, asc, and_, or_, func
from models import Subreddit, Author, Post, Comment


class SubredditRepository:
    """Repository for Subreddit operations"""

    def __init__(self, session: Session):
        self.session = session

    def get_by_id(self, subreddit_id: str) -> Optional[Subreddit]:
        """Get subreddit by ID"""
        return self.session.query(Subreddit).filter_by(subreddit_id=subreddit_id).first()

    def get_by_name(self, name: str) -> Optional[Subreddit]:
        """Get subreddit by name"""
        return self.session.query(Subreddit).filter_by(name=name).first()

    def create_or_update(self, **kwargs) -> Subreddit:
        """Create or update a subreddit"""
        subreddit = self.get_by_id(kwargs.get('subreddit_id'))
        if not subreddit:
            subreddit = Subreddit(**kwargs)
            self.session.add(subreddit)
        else:
            for key, value in kwargs.items():
                setattr(subreddit, key, value)
        self.session.commit()
        return subreddit

    def get_trending(self, days: int = 7, limit: int = 10) -> List[Subreddit]:
        """Get subreddits with most posts in recent days"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        return (self.session.query(Subreddit)
                .join(Post)
                .filter(Post.created_utc >= cutoff_date)
                .group_by(Subreddit.subreddit_id)
                .order_by(desc(func.count(Post.post_id)))
                .limit(limit)
                .all())

    def get_all(self, limit: Optional[int] = None) -> List[Subreddit]:
        """Get all subreddits"""
        query = self.session.query(Subreddit).order_by(desc(Subreddit.subscriber_count))
        if limit:
            query = query.limit(limit)
        return query.all()


class AuthorRepository:
    """Repository for Author operations"""

    def __init__(self, session: Session):
        self.session = session

    def get_by_id(self, author_id: str) -> Optional[Author]:
        """Get author by ID"""
        return self.session.query(Author).filter_by(author_id=author_id).first()

    def get_by_name(self, name: str) -> Optional[Author]:
        """Get author by username"""
        return self.session.query(Author).filter_by(name=name).first()

    def create_or_update(self, **kwargs) -> Author:
        """Create or update an author"""
        author = self.get_by_id(kwargs.get('author_id'))
        if not author:
            author = Author(**kwargs)
            self.session.add(author)
        else:
            for key, value in kwargs.items():
                setattr(author, key, value)
        self.session.commit()
        return author

    def get_top_by_karma(self, limit: int = 10) -> List[Author]:
        """Get top authors by total karma"""
        return (self.session.query(Author)
                .order_by(desc(Author.comment_karma + Author.link_karma))
                .limit(limit)
                .all())

    def get_most_active(self, days: int = 30, limit: int = 10) -> List[Tuple[Author, int]]:
        """Get most active authors by post/comment count in recent days"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        post_counts = (self.session.query(Author.author_id, func.count(Post.post_id).label('post_count'))
                      .join(Post)
                      .filter(Post.created_utc >= cutoff_date)
                      .group_by(Author.author_id)
                      .subquery())

        comment_counts = (self.session.query(Author.author_id, func.count(Comment.comment_id).label('comment_count'))
                         .join(Comment)
                         .filter(Comment.created_utc >= cutoff_date)
                         .group_by(Author.author_id)
                         .subquery())

        return (self.session.query(Author,
                                  (func.coalesce(post_counts.c.post_count, 0) +
                                   func.coalesce(comment_counts.c.comment_count, 0)).label('activity_count'))
                .outerjoin(post_counts, Author.author_id == post_counts.c.author_id)
                .outerjoin(comment_counts, Author.author_id == comment_counts.c.author_id)
                .order_by(desc('activity_count'))
                .limit(limit)
                .all())


class PostRepository:
    """Repository for Post operations"""

    def __init__(self, session: Session):
        self.session = session

    def get_by_id(self, post_id: str) -> Optional[Post]:
        """Get post by ID"""
        return self.session.query(Post).filter_by(post_id=post_id).first()

    def create_or_update(self, **kwargs) -> Post:
        """Create or update a post"""
        post = self.get_by_id(kwargs.get('post_id'))
        if not post:
            post = Post(**kwargs)
            self.session.add(post)
        else:
            for key, value in kwargs.items():
                setattr(post, key, value)
        self.session.commit()
        return post

    def get_by_subreddit(self, subreddit_id: str, limit: int = 100) -> List[Post]:
        """Get posts from a specific subreddit"""
        return (self.session.query(Post)
                .filter_by(subreddit_id=subreddit_id)
                .order_by(desc(Post.created_utc))
                .limit(limit)
                .all())

    def get_by_sentiment(self, sentiment_label: str, limit: int = 100) -> List[Post]:
        """Get posts by sentiment label"""
        return (self.session.query(Post)
                .filter(or_(Post.title_sentiment_label == sentiment_label,
                           Post.body_sentiment_label == sentiment_label))
                .order_by(desc(Post.created_utc))
                .limit(limit)
                .all())

    def get_trending(self, hours: int = 24, min_score: int = 100) -> List[Post]:
        """Get trending posts based on score and recency"""
        cutoff_date = datetime.utcnow() - timedelta(hours=hours)
        return (self.session.query(Post)
                .filter(and_(Post.created_utc >= cutoff_date,
                           Post.score >= min_score))
                .order_by(desc(Post.score))
                .all())

    def get_controversial(self, days: int = 7, min_comments: int = 50) -> List[Post]:
        """Get controversial posts (high engagement, mixed sentiment)"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        return (self.session.query(Post)
                .filter(and_(Post.created_utc >= cutoff_date,
                           Post.num_comments >= min_comments,
                           Post.title_sentiment_score.between(-0.3, 0.3)))
                .order_by(desc(Post.num_comments))
                .all())

    def needs_update(self, hours: int = 6) -> List[Post]:
        """Get posts that need to be re-scraped"""
        update_cutoff = datetime.utcnow() - timedelta(hours=hours)
        return (self.session.query(Post)
                .filter(or_(Post.last_scraped_utc.is_(None),
                           Post.last_scraped_utc < update_cutoff))
                .order_by(asc(Post.last_scraped_utc))
                .all())


class CommentRepository:
    """Repository for Comment operations"""

    def __init__(self, session: Session):
        self.session = session

    def get_by_id(self, comment_id: str) -> Optional[Comment]:
        """Get comment by ID"""
        return self.session.query(Comment).filter_by(comment_id=comment_id).first()

    def create_or_update(self, **kwargs) -> Comment:
        """Create or update a comment"""
        comment = self.get_by_id(kwargs.get('comment_id'))
        if not comment:
            comment = Comment(**kwargs)
            self.session.add(comment)
        else:
            for key, value in kwargs.items():
                setattr(comment, key, value)
        self.session.commit()
        return comment

    def get_by_post(self, post_id: str, limit: Optional[int] = None) -> List[Comment]:
        """Get comments for a specific post"""
        query = (self.session.query(Comment)
                .filter_by(post_id=post_id)
                .order_by(desc(Comment.score)))
        if limit:
            query = query.limit(limit)
        return query.all()

    def get_by_author(self, author_id: str, limit: int = 100) -> List[Comment]:
        """Get comments by a specific author"""
        return (self.session.query(Comment)
                .filter_by(author_id=author_id)
                .order_by(desc(Comment.created_utc))
                .limit(limit)
                .all())

    def get_by_sentiment(self, sentiment_label: str, limit: int = 100) -> List[Comment]:
        """Get comments by sentiment label"""
        return (self.session.query(Comment)
                .filter_by(sentiment_label=sentiment_label)
                .order_by(desc(Comment.created_utc))
                .limit(limit)
                .all())

    def get_top_level(self, post_id: str) -> List[Comment]:
        """Get only top-level comments (direct replies to post)"""
        return (self.session.query(Comment)
                .filter(and_(Comment.post_id == post_id,
                           Comment.parent_id == post_id))
                .order_by(desc(Comment.score))
                .all())


class SentimentAnalyzer:
    """Helper class for sentiment analysis queries"""

    def __init__(self, session: Session):
        self.session = session

    def get_subreddit_sentiment(self, subreddit_id: str, days: int = 30) -> Dict[str, Any]:
        """Get sentiment statistics for a subreddit"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        posts = (self.session.query(Post)
                .filter(and_(Post.subreddit_id == subreddit_id,
                           Post.created_utc >= cutoff_date))
                .all())

        if not posts:
            return {"error": "No posts found"}

        sentiment_scores = [p.overall_sentiment_score for p in posts if p.overall_sentiment_score is not None]

        return {
            "subreddit_id": subreddit_id,
            "period_days": days,
            "total_posts": len(posts),
            "average_sentiment": sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else None,
            "positive_posts": sum(1 for p in posts if p.overall_sentiment_label == "positive"),
            "negative_posts": sum(1 for p in posts if p.overall_sentiment_label == "negative"),
            "neutral_posts": sum(1 for p in posts if p.overall_sentiment_label == "neutral"),
        }

    def get_sentiment_trend(self, subreddit_id: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get daily sentiment trend for a subreddit"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        daily_sentiment = (self.session.query(
                            func.date(Post.created_utc).label('date'),
                            func.avg(Post.title_sentiment_score).label('avg_sentiment'),
                            func.count(Post.post_id).label('post_count'))
                          .filter(and_(Post.subreddit_id == subreddit_id,
                                     Post.created_utc >= cutoff_date,
                                     Post.title_sentiment_score.isnot(None)))
                          .group_by(func.date(Post.created_utc))
                          .order_by(asc('date'))
                          .all())

        return [{"date": str(row.date),
                "average_sentiment": float(row.avg_sentiment) if row.avg_sentiment else None,
                "post_count": row.post_count}
               for row in daily_sentiment]

    def get_author_sentiment(self, author_id: str) -> Dict[str, Any]:
        """Get sentiment statistics for an author"""
        posts = self.session.query(Post).filter_by(author_id=author_id).all()
        comments = self.session.query(Comment).filter_by(author_id=author_id).all()

        post_sentiments = [p.overall_sentiment_score for p in posts if p.overall_sentiment_score is not None]
        comment_sentiments = [c.sentiment_score for c in comments if c.sentiment_score is not None]

        all_sentiments = post_sentiments + comment_sentiments

        return {
            "author_id": author_id,
            "total_posts": len(posts),
            "total_comments": len(comments),
            "average_post_sentiment": sum(post_sentiments) / len(post_sentiments) if post_sentiments else None,
            "average_comment_sentiment": sum(comment_sentiments) / len(comment_sentiments) if comment_sentiments else None,
            "overall_sentiment": sum(all_sentiments) / len(all_sentiments) if all_sentiments else None,
        }