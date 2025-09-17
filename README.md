# Reddit Sentiment Tracker

A comprehensive Reddit sentiment tracking and analysis system that monitors specified subreddits, fetches posts and comments, and performs brand-aware sentiment analysis.

## Features

- 📡 Real-time Reddit data fetching via PRAW API
- 🤖 Advanced sentiment analysis using Anthropic Claude API
- 🏢 Brand-aware sentiment tracking (distinguishes brand vs competitor mentions)
- 📊 PostgreSQL database for persistent storage
- 🔄 Batch processing with rate limiting
- 📈 Statistical analysis and reporting
- 🐳 Docker support for easy deployment

## Quick Start

1. Download the project:
```bash
git clone <repo-url>
cd reddit-sentiment-tracker
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies inside the virtual environment:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API credentials and database info
```

5. Initialise the database schema:
```bash
psql -U your_user -d your_database < migrations/schema_latest.sql
```

   If you are upgrading an existing deployment, run the numbered migrations in order instead.

6. Run the tracker:
```bash
# Fetch Reddit data (last 24 hours)
python main.py fetch --hours 24

# Fetch Reddit data for an explicit window (ISO 8601 timestamps)
python main.py fetch --start 2024-05-01T00:00 --end 2024-05-02T00:00

# Run sentiment analysis
python main.py analyze

# View statistics
python main.py stats

# Continuous monitoring
python main.py monitor --interval 6
```

## Project Structure

```
reddit-sentiment-tracker/
├── main.py                  # Main CLI application
├── data/                    # Database models and repositories
├── reddit/                  # Reddit data fetching modules
├── sentiment/               # Sentiment analysis
├── config/                  # Configuration files
├── migrations/              # Database migrations
├── scripts/                 # Utility scripts
├── examples/                # Usage examples
└── docs/                    # Documentation
```

## Documentation

See the `docs/` directory for detailed documentation:
- [Documentation Index](docs/README.md)
- [Data Fetching & Pipeline](docs/data_fetching.md)
- [Brand-Aware Sentiment Analysis](docs/sentiment_analysis.md)
- [Rate Limit Strategies](docs/rate_limit.md)

## License

This project is licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later). See the LICENSE file for full terms.
