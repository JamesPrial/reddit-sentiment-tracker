"""
Brand-aware sentiment analysis for Reddit posts and comments
"""
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import re
from pathlib import Path

logger = logging.getLogger(__name__)


class BrandSentiment(Enum):
    """Brand-aware sentiment categories"""
    POSITIVE_BRAND = "positive_brand"  # Positive about our brand
    NEGATIVE_BRAND = "negative_brand"  # Negative about our brand
    POSITIVE_COMPETITOR = "positive_competitor"  # Positive about competitor (bad for us)
    NEGATIVE_COMPETITOR = "negative_competitor"  # Negative about competitor (good for us)
    MIXED_COMPARISON = "mixed_comparison"  # Complex comparison
    NEUTRAL = "neutral"  # Neutral sentiment
    IRRELEVANT = "irrelevant"  # Not brand-related


@dataclass
class SentimentResult:
    """Result of sentiment analysis"""
    text: str
    sentiment_score: float  # -1.0 to 1.0
    sentiment_label: str  # Traditional sentiment
    brand_sentiment_score: float  # Adjusted for brand context
    brand_sentiment_label: BrandSentiment
    brand_mentions: Dict[str, List[str]]  # Brand -> mentions found
    confidence: float  # 0.0 to 1.0
    reasoning: str  # Explanation of the analysis
    metadata: Dict[str, Any]  # Additional analysis data


DEFAULT_BRAND_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "brand_config.json"


class BrandConfig:
    """Manages brand configuration and keyword matching"""

    def __init__(self, config_path: Optional[str] = None):
        """Load brand configuration from file"""
        path = Path(config_path) if config_path else DEFAULT_BRAND_CONFIG_PATH
        if not path.exists():
            raise FileNotFoundError(f"Brand configuration file not found at {path}")

        with open(path, 'r') as f:
            self.config = json.load(f)
        self.config_path = str(path)

        # Compile regex patterns for efficient matching
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for all brands and products"""
        self.brand_patterns = {}

        # Primary brand patterns
        primary = self.config['primary_brand']
        all_terms = (
            primary['keywords'] +
            primary['products'] +
            primary.get('variations', [])
        )
        self.brand_patterns[primary['name']] = self._create_pattern(all_terms)

        # Competitor patterns
        for competitor in self.config['competitors']:
            all_terms = (
                competitor['keywords'] +
                competitor['products'] +
                competitor.get('variations', [])
            )
            self.brand_patterns[competitor['name']] = self._create_pattern(all_terms)

    def _create_pattern(self, terms: List[str]) -> re.Pattern:
        """Create regex pattern from list of terms"""
        # Escape special characters and create word boundary pattern
        escaped_terms = [re.escape(term) for term in terms]
        pattern = r'\b(' + '|'.join(escaped_terms) + r')\b'
        return re.compile(pattern, re.IGNORECASE)

    def detect_brands(self, text: str) -> Dict[str, List[str]]:
        """Detect brand mentions in text"""
        mentions = {}

        for brand_name, pattern in self.brand_patterns.items():
            matches = pattern.findall(text)
            if matches:
                # Deduplicate while preserving order
                unique_matches = []
                seen = set()
                for match in matches:
                    match_lower = match.lower()
                    if match_lower not in seen:
                        seen.add(match_lower)
                        unique_matches.append(match)
                mentions[brand_name] = unique_matches

        return mentions

    def is_primary_brand(self, brand_name: str) -> bool:
        """Check if brand is the primary brand"""
        return brand_name == self.config['primary_brand']['name']

    def get_sentiment_context(self) -> str:
        """Get context string for LLM prompts"""
        primary = self.config['primary_brand']
        competitors = [c['name'] for c in self.config['competitors']]

        return f"""
        Primary brand: {primary['name']} (products: {', '.join(primary['products'])})
        Competitors: {', '.join(competitors)}

        When analyzing sentiment:
        - Positive sentiment about {primary['name']} = positive for brand
        - Negative sentiment about {primary['name']} = negative for brand
        - Positive sentiment about competitors ({', '.join(competitors)}) = negative for {primary['name']}
        - Negative sentiment about competitors = positive for {primary['name']}
        - Comparisons need careful analysis of which brand is favored
        """


class SentimentAnalyzer(ABC):
    """Abstract base class for sentiment analyzers"""

    def __init__(self, brand_config: Optional[BrandConfig] = None):
        """Initialize analyzer with brand configuration"""
        self.brand_config = brand_config or BrandConfig()

    @abstractmethod
    def analyze(self, text: str, context: Optional[str] = None) -> SentimentResult:
        """Analyze sentiment of text with optional context"""
        pass

    @abstractmethod
    def analyze_batch(self, texts: List[str], context: Optional[str] = None) -> List[SentimentResult]:
        """Analyze sentiment of multiple texts"""
        pass

    def _adjust_for_brand_context(
        self,
        sentiment_score: float,
        sentiment_label: str,
        brand_mentions: Dict[str, List[str]],
        text: str
    ) -> Tuple[float, BrandSentiment]:
        """Adjust sentiment based on brand context"""

        if not brand_mentions:
            # No brand mentions, neutral from brand perspective
            return 0.0, BrandSentiment.IRRELEVANT

        primary_mentioned = any(
            self.brand_config.is_primary_brand(brand)
            for brand in brand_mentions.keys()
        )
        competitors_mentioned = any(
            not self.brand_config.is_primary_brand(brand)
            for brand in brand_mentions.keys()
        )

        # Simple heuristic adjustment (can be refined with ML)
        if primary_mentioned and competitors_mentioned:
            # Comparison detected
            return sentiment_score * 0.5, BrandSentiment.MIXED_COMPARISON
        elif primary_mentioned:
            # About our brand
            if sentiment_score > 0.1:
                return sentiment_score, BrandSentiment.POSITIVE_BRAND
            elif sentiment_score < -0.1:
                return sentiment_score, BrandSentiment.NEGATIVE_BRAND
            else:
                return sentiment_score, BrandSentiment.NEUTRAL
        elif competitors_mentioned:
            # About competitors - invert sentiment
            if sentiment_score > 0.1:
                return -sentiment_score, BrandSentiment.POSITIVE_COMPETITOR
            elif sentiment_score < -0.1:
                return -sentiment_score, BrandSentiment.NEGATIVE_COMPETITOR
            else:
                return 0.0, BrandSentiment.NEUTRAL

        return 0.0, BrandSentiment.IRRELEVANT


class AnthropicAnalyzer(SentimentAnalyzer):
    """Sentiment analyzer using Anthropic's Claude API"""

    def __init__(self, api_key: Optional[str] = None, brand_config: Optional[BrandConfig] = None):
        """Initialize with API key and brand config"""
        super().__init__(brand_config)

        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("Anthropic API key not provided. Set ANTHROPIC_API_KEY environment variable.")

        # Initialize Anthropic client
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

    def _create_prompt(self, text: str, context: Optional[str] = None) -> str:
        """Create analysis prompt for Claude"""
        brand_context = self.brand_config.get_sentiment_context()

        prompt = f"""Analyze the sentiment of the following Reddit text with brand awareness.

{brand_context}

Text to analyze:
"{text}"

{f"Additional context: {context}" if context else ""}

Provide a JSON response with:
{{
    "sentiment_score": float between -1 (very negative) and 1 (very positive),
    "sentiment_label": "positive" | "negative" | "neutral",
    "brand_sentiment_score": float adjusted for brand context,
    "brand_sentiment_category": "positive_brand" | "negative_brand" | "positive_competitor" | "negative_competitor" | "mixed_comparison" | "neutral" | "irrelevant",
    "detected_brands": {{brand_name: [mentions]}},
    "confidence": float between 0 and 1,
    "reasoning": "Brief explanation of the analysis"
}}

Consider:
- Sarcasm and irony
- Comparisons between brands
- Feature discussions
- Migration mentions (switching from/to)
- Technical critiques vs general complaints"""

        return prompt

    def analyze(self, text: str, context: Optional[str] = None) -> SentimentResult:
        """Analyze sentiment using Claude API"""

        # Detect brand mentions locally first
        brand_mentions = self.brand_config.detect_brands(text)

        # Create and send prompt
        prompt = self._create_prompt(text, context)

        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",  # Fast and cost-effective
                max_tokens=500,
                temperature=0.1,  # Low temperature for consistency
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse JSON response
            import json
            result_data = json.loads(response.content[0].text)

            return SentimentResult(
                text=text,
                sentiment_score=result_data['sentiment_score'],
                sentiment_label=result_data['sentiment_label'],
                brand_sentiment_score=result_data['brand_sentiment_score'],
                brand_sentiment_label=BrandSentiment(result_data['brand_sentiment_category']),
                brand_mentions=result_data.get('detected_brands', brand_mentions),
                confidence=result_data['confidence'],
                reasoning=result_data['reasoning'],
                metadata={'model': 'claude-3-haiku', 'api_response': result_data}
            )

        except Exception as e:
            logger.error(f"Error calling Anthropic API: {e}")
            # Fallback to local analysis
            return self._fallback_analysis(text, brand_mentions)

    def analyze_batch(self, texts: List[str], context: Optional[str] = None) -> List[SentimentResult]:
        """Analyze multiple texts efficiently"""

        # Sanitize and prepare texts for prompt
        texts_for_prompt = []
        for i, text in enumerate(texts):
            # Clean problematic characters
            cleaned = text.replace('\\', '\\\\')  # Escape backslashes first
            cleaned = cleaned.replace('"', '\\"')  # Escape quotes
            cleaned = cleaned.replace('\n', ' ')  # Replace newlines with spaces
            cleaned = cleaned.replace('\r', ' ')  # Remove carriage returns
            cleaned = cleaned.replace('\t', ' ')  # Replace tabs with spaces

            # Remove or replace problematic Unicode characters
            import unicodedata
            cleaned = unicodedata.normalize('NFKD', cleaned)
            cleaned = ''.join(char if ord(char) < 127 else ' ' for char in cleaned)

            # Truncate if too long
            if len(cleaned) > 400:
                cleaned = cleaned[:400] + "..."

            texts_for_prompt.append({"id": i, "text": cleaned})

        # Create batch prompt with system instruction
        batch_prompt = f"""Analyze sentiment for {len(texts_for_prompt)} Reddit texts.

Brand context:
- Primary: Anthropic/Claude
- Competitors: OpenAI/ChatGPT, Google/Gemini, Meta/Llama

Categories: positive_brand, negative_brand, positive_competitor, negative_competitor, mixed_comparison, neutral, irrelevant

Texts: {json.dumps(texts_for_prompt)}

Respond with JSON array only:"""

        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=2000,
                temperature=0,  # Use 0 for most deterministic JSON output
                system="You are a JSON API that only returns valid JSON arrays. Never include explanatory text.",
                messages=[
                    {"role": "user", "content": batch_prompt}
                ]
            )

            response_text = response.content[0].text

            response_text = response.content[0].text

            # Parse the JSON response with comprehensive cleaning
            try:
                cleaned_text = response_text.strip()

                # Remove any text before the first [ and after last ]
                import re
                json_match = re.search(r'\[.*\]', cleaned_text, re.DOTALL)
                if json_match:
                    cleaned_text = json_match.group()

                # Comprehensive JSON fixing
                # Fix unescaped colons (simplified regex to avoid look-behind issues)
                # This is less precise but avoids the regex error
                cleaned_text = cleaned_text.replace('": "', '": "')  # Preserve valid JSON colons

                # Fix missing commas between objects
                cleaned_text = re.sub(r'}\s*{', '},{', cleaned_text)

                # Fix missing commas between array elements
                cleaned_text = re.sub(r'"\s*\n\s*"', '","', cleaned_text)
                cleaned_text = re.sub(r'}\s*\n\s*{', '},{', cleaned_text)

                # Remove trailing commas
                cleaned_text = re.sub(r',\s*([\]}])', r'\1', cleaned_text)

                # Fix newlines in string values
                cleaned_text = re.sub(r'\n(?=[^"]*"(?:[^"]*"[^"]*")*[^"]*$)', ' ', cleaned_text)

                # Remove any control characters
                cleaned_text = ''.join(char for char in cleaned_text if ord(char) >= 32 or char in '\n\r\t')

                # Parse the JSON
                results_data = json.loads(cleaned_text)

                # Validate it's a list
                if not isinstance(results_data, list):
                    raise ValueError("Response is not a JSON array")

            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"JSON parse error in batch: {e}")
                logger.debug(f"Response was: {response_text[:500]}...")

                # Try to salvage partial results
                try:
                    # Try to parse up to the error position
                    partial = cleaned_text[:cleaned_text.rfind('}') + 1] + ']'
                    results_data = json.loads(partial)
                    logger.info(f"Salvaged {len(results_data)} results from partial JSON")
                except:
                    # Complete failure - fall back to individual analysis
                    logger.warning("Batch analysis failed, falling back to individual analysis")
                    return [self.analyze(text, context) for text in texts]

            results = []
            for i, text in enumerate(texts):
                result_data = next((r for r in results_data if r.get('id') == i), None)
                if result_data:
                    try:
                        # Validate and extract fields with defaults
                        # Handle variant category names that Claude might return
                        category = result_data.get('brand_sentiment_category', 'irrelevant')
                        if category == 'mixed_brand':
                            category = 'mixed_comparison'
                        elif category not in ['positive_brand', 'negative_brand', 'positive_competitor',
                                             'negative_competitor', 'mixed_comparison', 'neutral', 'irrelevant']:
                            logger.warning(f"Unknown category '{category}', using 'neutral'")
                            category = 'neutral'

                        results.append(SentimentResult(
                            text=text,
                            sentiment_score=float(result_data.get('sentiment_score', 0)),
                            sentiment_label=result_data.get('sentiment_label', 'neutral'),
                            brand_sentiment_score=float(result_data.get('brand_sentiment_score', 0)),
                            brand_sentiment_label=BrandSentiment(category),
                            brand_mentions=result_data.get('detected_brands', {}),
                            confidence=float(result_data.get('confidence', 0.5)),
                            reasoning=result_data.get('reasoning', 'Batch analysis'),
                            metadata={'model': 'claude-3-haiku', 'batch': True}
                        ))
                    except (KeyError, ValueError, TypeError) as e:
                        logger.error(f"Error parsing result {i}: {e}")
                        # Use local brand detection as fallback
                        brand_mentions = self.brand_config.detect_brands(text)
                        results.append(self._fallback_analysis(text, brand_mentions))
                else:
                    # Fallback for missing results
                    brand_mentions = self.brand_config.detect_brands(text)
                    results.append(self._fallback_analysis(text, brand_mentions))

            return results

        except Exception as e:
            logger.error(f"Error in batch analysis: {e}")
            # Always fall back to individual analysis on any error
            logger.info("Falling back to individual analysis due to batch error")
            results = []
            for i, text in enumerate(texts):
                try:
                    results.append(self.analyze(text, context))
                except Exception as ind_error:
                    logger.error(f"Individual analysis failed for text {i}: {ind_error}")
                    # Use basic fallback
                    brand_mentions = self.brand_config.detect_brands(text)
                    results.append(self._fallback_analysis(text, brand_mentions))
            return results

    def _fallback_analysis(self, text: str, brand_mentions: Dict[str, List[str]]) -> SentimentResult:
        """Simple fallback analysis when API fails"""

        # Use basic sentiment indicators
        positive_words = ['good', 'great', 'excellent', 'best', 'love', 'amazing']
        negative_words = ['bad', 'worst', 'hate', 'terrible', 'awful', 'broken']

        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)

        if pos_count > neg_count:
            sentiment_score = min(0.5 + 0.1 * (pos_count - neg_count), 1.0)
            sentiment_label = "positive"
        elif neg_count > pos_count:
            sentiment_score = max(-0.5 - 0.1 * (neg_count - pos_count), -1.0)
            sentiment_label = "negative"
        else:
            sentiment_score = 0.0
            sentiment_label = "neutral"

        # Adjust for brand context
        brand_score, brand_label = self._adjust_for_brand_context(
            sentiment_score, sentiment_label, brand_mentions, text
        )

        return SentimentResult(
            text=text,
            sentiment_score=sentiment_score,
            sentiment_label=sentiment_label,
            brand_sentiment_score=brand_score,
            brand_sentiment_label=brand_label,
            brand_mentions=brand_mentions,
            confidence=0.3,  # Low confidence for fallback
            reasoning="Fallback analysis using keyword matching",
            metadata={'fallback': True}
        )


class LocalModelAnalyzer(SentimentAnalyzer):
    """Sentiment analyzer using local models via Ollama"""

    def __init__(self, model_name: str = "llama3", brand_config: Optional[BrandConfig] = None):
        """Initialize with local model name"""
        super().__init__(brand_config)
        self.model_name = model_name

        # Check if Ollama is available
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code != 200:
                raise ConnectionError("Ollama not running on localhost:11434")

            models = response.json().get('models', [])
            if not any(m['name'].startswith(model_name) for m in models):
                logger.warning(f"Model {model_name} not found in Ollama. Available models: {[m['name'] for m in models]}")
        except Exception as e:
            logger.warning(f"Could not connect to Ollama: {e}")

    def _create_prompt(self, text: str, context: Optional[str] = None) -> str:
        """Create analysis prompt for local model"""
        brand_context = self.brand_config.get_sentiment_context()

        # Simpler prompt for local models
        prompt = f"""You are a sentiment analysis expert. Analyze this Reddit text considering brand context.

{brand_context}

Text: "{text}"

Respond with ONLY valid JSON:
{{
    "sentiment_score": -1 to 1,
    "sentiment_label": "positive" or "negative" or "neutral",
    "brand_sentiment_score": -1 to 1 adjusted for brand,
    "brand_sentiment_category": "positive_brand" or "negative_brand" or "positive_competitor" or "negative_competitor" or "mixed_comparison" or "neutral" or "irrelevant",
    "confidence": 0 to 1,
    "reasoning": "one sentence explanation"
}}"""

        return prompt

    def analyze(self, text: str, context: Optional[str] = None) -> SentimentResult:
        """Analyze sentiment using local model"""

        # Detect brand mentions
        brand_mentions = self.brand_config.detect_brands(text)

        try:
            import requests

            prompt = self._create_prompt(text, context)

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.1,
                    "format": "json"
                }
            )

            if response.status_code == 200:
                result_text = response.json()['response']
                result_data = json.loads(result_text)

                return SentimentResult(
                    text=text,
                    sentiment_score=float(result_data['sentiment_score']),
                    sentiment_label=result_data['sentiment_label'],
                    brand_sentiment_score=float(result_data['brand_sentiment_score']),
                    brand_sentiment_label=BrandSentiment(result_data['brand_sentiment_category']),
                    brand_mentions=brand_mentions,
                    confidence=float(result_data['confidence']),
                    reasoning=result_data['reasoning'],
                    metadata={'model': self.model_name, 'local': True}
                )
            else:
                raise Exception(f"Ollama returned status {response.status_code}")

        except Exception as e:
            logger.error(f"Error with local model: {e}")

            # Simple fallback
            sentiment_score = 0.0
            sentiment_label = "neutral"

            # Check for obvious positive/negative words
            if any(word in text.lower() for word in ['great', 'excellent', 'best', 'love']):
                sentiment_score = 0.5
                sentiment_label = "positive"
            elif any(word in text.lower() for word in ['bad', 'worst', 'hate', 'terrible']):
                sentiment_score = -0.5
                sentiment_label = "negative"

            brand_score, brand_label = self._adjust_for_brand_context(
                sentiment_score, sentiment_label, brand_mentions, text
            )

            return SentimentResult(
                text=text,
                sentiment_score=sentiment_score,
                sentiment_label=sentiment_label,
                brand_sentiment_score=brand_score,
                brand_sentiment_label=brand_label,
                brand_mentions=brand_mentions,
                confidence=0.2,
                reasoning="Fallback analysis",
                metadata={'error': str(e), 'fallback': True}
            )

    def analyze_batch(self, texts: List[str], context: Optional[str] = None) -> List[SentimentResult]:
        """Analyze multiple texts"""
        # Local models typically don't support true batching, so process sequentially
        return [self.analyze(text, context) for text in texts]


def get_analyzer(use_api: bool = True, api_key: Optional[str] = None) -> SentimentAnalyzer:
    """Factory function to get appropriate analyzer"""

    if use_api:
        try:
            return AnthropicAnalyzer(api_key=api_key)
        except (ImportError, ValueError) as e:
            logger.warning(f"Could not initialize Anthropic analyzer: {e}")
            logger.info("Falling back to local model analyzer")
            return LocalModelAnalyzer()
    else:
        return LocalModelAnalyzer()
