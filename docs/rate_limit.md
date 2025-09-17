# Rate Limit Strategies

The Reddit API enforces strict quotas. The tracker defends against them in several layers:

- `FetchConfig.rate_limit_delay` adds a pause between requests (default 1s). Increase this value if you hit 429 responses frequently.
- `_handle_rate_limit_retry` in `reddit/reddit_fetcher.py` backs off exponentially when Reddit returns 429s. You can tune `max_retries` or the base wait at the top of that helper.
- Batch large jobs across subreddits by staggering start times or constraining `max_posts`/`max_comments_per_post` in the config when experimenting.
- When running continuous monitoring, choose an interval that significantly exceeds the time required for a full cycle so successive runs do not overlap.

For manual testing, keep an eye on the logs (`reddit_tracker.log`). Repeated 429 entries are a signal to slow the rate or temporarily disable comment fetching.
