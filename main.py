#!/usr/bin/env python3
"""
Reddit Sentiment Tracker - CLI Entry Point
"""
import sys
import logging
import argparse
from datetime import datetime

from config import AppConfig, FETCH, ANALYZE, STATS, MONITOR, DEFAULT_UPDATE_INTERVAL
from app import RedditSentimentTracker

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


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description='Reddit Sentiment Tracker')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Fetch command
    fetch_parser = subparsers.add_parser(FETCH, help='Fetch Reddit data')
    fetch_parser.add_argument('--hours', type=int, help='Hours of data to fetch')
    fetch_parser.add_argument('--start', type=parse_datetime, help='Start time (ISO 8601)')
    fetch_parser.add_argument('--end', type=parse_datetime, help='End time (ISO 8601)')
    fetch_parser.add_argument('--sort', choices=['new', 'hot', 'top', 'rising'], help='Sort method')

    # Analyze command
    analyze_parser = subparsers.add_parser(ANALYZE, help='Run sentiment analysis')
    analyze_parser.add_argument('--hours', type=int, help='Hours of data to analyze')
    analyze_parser.add_argument('--force', action='store_true', help='Force re-analysis')

    # Stats command
    stats_parser = subparsers.add_parser(STATS, help='Show statistics')
    stats_parser.add_argument('--hours', type=int, help='Hours of data to include')

    # Monitor command
    monitor_parser = subparsers.add_parser(MONITOR, help='Run continuous monitoring')
    monitor_parser.add_argument('--interval', type=int, default=DEFAULT_UPDATE_INTERVAL,
                                help='Update interval in hours')

    # Update command (new)
    parser_update = subparsers.add_parser('update', help='Update stale posts')
    parser_update.add_argument('--hours', type=int, default=6, help='Hours old to consider stale')

    # Stream command (new)
    parser_stream = subparsers.add_parser('stream', help='Stream real-time data from a subreddit')
    parser_stream.add_argument('subreddit', help='Subreddit to stream')
    parser_stream.add_argument('--no-comments', action='store_true', help='Skip fetching comments')

    args = parser.parse_args()

    # Load and validate configuration
    try:
        config = AppConfig.from_env()
        config.validate()
        logger.info(f"Configuration loaded. Tracking: {', '.join(config.tracker.subreddits)}")
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        sys.exit(1)

    # Initialize tracker
    try:
        tracker = RedditSentimentTracker(config)
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        sys.exit(1)

    # Execute command
    try:
        if args.command == FETCH:
            tracker.fetch_data(
                start=args.start,
                end=args.end,
                hours=args.hours,
                sort_by=args.sort if hasattr(args, 'sort') else None
            )

        elif args.command == ANALYZE:
            tracker.analyze_sentiment(hours=args.hours, force=args.force)

        elif args.command == STATS:
            tracker.show_stats(hours=args.hours)

        elif args.command == MONITOR:
            tracker.run_monitor(interval_hours=args.interval)

        elif args.command == 'update':
            tracker.update_stale_posts(hours_old=args.hours)

        elif args.command == 'stream':
            tracker.stream_subreddit(
                args.subreddit,
                fetch_comments=not args.no_comments
            )

        else:
            # Default: show help
            parser.print_help()
            print("\nExamples:")
            print("  python main.py fetch --hours 24")
            print("  python main.py fetch --start 2024-05-01T00:00 --end 2024-05-02T00:00")
            print("  python main.py analyze")
            print("  python main.py stats")
            print("  python main.py monitor --interval 6")
            print("  python main.py update --hours 12")
            print("  python main.py stream wallstreetbets")

    except KeyboardInterrupt:
        print("\n\n👋 Operation cancelled")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Command failed: {e}", exc_info=True)
        print(f"❌ Error: {e}")
        sys.exit(1)
    finally:
        tracker.close()


if __name__ == '__main__':
    main()