# Brand-Aware Sentiment Analysis

## Overview

This Reddit Sentiment Tracker now includes sophisticated brand-aware sentiment analysis that understands competitive dynamics. When analyzing a brand's subreddit, it correctly interprets that positive sentiment about competitors is actually a negative signal for the tracked brand.

## Key Features

### 1. Brand-Aware Categories
- **positive_brand**: Positive sentiment about your brand (good signal)
- **negative_brand**: Negative sentiment about your brand (bad signal)
- **positive_competitor**: Positive sentiment about competitors (bad signal for your brand)
- **negative_competitor**: Negative sentiment about competitors (good signal for your brand)
- **mixed_comparison**: Complex comparisons between brands
- **neutral**: Neutral sentiment
- **irrelevant**: No brand mentions detected

### 2. Intelligent Context Understanding
- Detects migration patterns ("switched from ChatGPT to Claude")
- Understands comparisons ("X is better than Y")
- Recognizes sarcasm and irony
- Handles technical critiques vs general complaints

### 3. Flexible Analysis Options
- **Anthropic Claude API**: High accuracy, understands nuance
- **Local Models (Ollama)**: Free, private, runs locally
- **Batch Processing**: Efficient API usage
- **Caching**: Avoids re-analyzing identical content

## Setup

### Option 1: Using Anthropic API (Recommended)
```bash
# Add to .env file
ANTHROPIC_API_KEY=your_api_key_here
```

### Option 2: Using Local Models
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama3

# Add to .env file
USE_LOCAL_MODEL=true
```

### Database Migration
Ensure your database is created with `migrations/schema_latest.sql`. For legacy installs, apply any numbered migrations you have not yet run.

## Configuration

### Brand Configuration (`config/brand_config.json`)
```json
{
  "primary_brand": {
    "name": "Anthropic",
    "products": ["Claude", "Claude Code"],
    "keywords": ["anthropic", "claude", "claude.ai"]
  },
  "competitors": [
    {
      "name": "OpenAI",
      "products": ["ChatGPT", "GPT-4"],
      "keywords": ["openai", "chatgpt", "gpt-4"]
    }
  ]
}
```

The default file lives at `config/brand_config.json`. Copy it, adjust brand/competitor terms, and restart the app so the new patterns are loaded.

## Usage

### CLI Workflow
Sentiment analysis hooks into the main CLI. Typical commands:

```bash
# Analyze the most recent 24 hours (default window)
python main.py analyze

# Analyze a specific time range
python main.py analyze --hours 6

# Force a full re-run even if sentiment already exists
python main.py analyze --force

# View coverage and averages
python main.py stats --hours 24
```

### Programmatic Usage
```python
from sentiment import get_analyzer

# Initialize analyzer
analyzer = get_analyzer(use_api=True)  # or False for local

# Analyze single text
result = analyzer.analyze("Claude is better than ChatGPT!")
print(f"Brand sentiment: {result.brand_sentiment_label}")
print(f"Score: {result.brand_sentiment_score}")

# Batch analysis
texts = ["text1", "text2", "text3"]
results = analyzer.analyze_batch(texts)
```

### Integration with Data Pipeline
When `enable_sentiment_analysis=True` the data pipeline automatically runs the sentiment batcher after storing new content:

```python
from data.pipeline import RedditDataPipeline

pipeline = RedditDataPipeline(
    reddit_client_id="...",
    reddit_client_secret="...",
    reddit_user_agent="...",
    db_connection_string="...",
    enable_sentiment_analysis=True,
    sentiment_api_key="..."  # Optional if using Anthropic
)
```

## Testing

Add regression coverage in `tests/` by instantiating the analyzer with a fake brand config and asserting on `SentimentResult`. Automated tests are not shipped yet, so add them before relying on custom modifications.

## Example Analysis

### Input
"Just switched from ChatGPT to Claude and the difference is incredible!"

### Traditional Analysis
- Sentiment: Positive (0.8)
- Label: "positive"

### Brand-Aware Analysis
- Brand Sentiment: positive_brand
- Score: 0.8 (positive for Claude)
- Brands: {"Anthropic": ["Claude"], "OpenAI": ["ChatGPT"]}
- Reasoning: "User migrated from competitor to primary brand with positive experience"

## Database Schema

New columns added to posts and comments tables:
- `brand_sentiment_score` (FLOAT): Adjusted sentiment score
- `brand_sentiment_label` (VARCHAR): Brand category
- `brand_mentions` (JSON): Detected brand mentions
- `sentiment_metadata` (JSON): Analysis details
- `sentiment_analyzed_at` (TIMESTAMP): Analysis timestamp

## Performance Considerations

### API Costs
- Claude 3 Haiku: ~$0.25 per million input tokens
- Batch processing reduces API calls by 10x
- Caching prevents re-analysis

### Processing Speed
- API: ~100-200ms per item (batched)
- Local (Llama 3): ~500-1000ms per item
- Cache hits: <1ms

### Recommendations
1. Use batch size of 10-20 for optimal API efficiency
2. Enable caching for duplicate content
3. Prioritize high-engagement content
4. Run analysis during off-peak hours

## Monitoring

View sentiment trends:
```sql
-- Average sentiment by day
SELECT
    DATE(created_utc) as day,
    AVG(brand_sentiment_score) as avg_sentiment,
    COUNT(*) as posts
FROM posts
WHERE brand_sentiment_score IS NOT NULL
GROUP BY DATE(created_utc)
ORDER BY day DESC;

-- Top mentioned competitors
SELECT
    brand_mentions,
    COUNT(*) as mentions,
    AVG(brand_sentiment_score) as avg_sentiment
FROM posts
WHERE brand_mentions IS NOT NULL
GROUP BY brand_mentions
ORDER BY mentions DESC;
```

## Troubleshooting

### No sentiment scores appearing
1. Check ANTHROPIC_API_KEY is set correctly
2. Verify Ollama is running if using local model
3. Run database migration for new columns
4. Check logs for API errors

### Incorrect sentiment categorization
1. Review brand_config.json keywords
2. Adjust sentiment thresholds if needed
3. Consider using API instead of local model

### Slow processing
1. Increase batch size (up to 50)
2. Enable caching
3. Use API instead of local model
4. Process during off-peak hours
