"""Finance command handlers for MOTHER."""
from __future__ import annotations

from typing import Optional, Dict, Any, Tuple
import httpx

from mother.core.logging_config import get_logger

logger = get_logger("commands.finance")

# Symbol resolution mapping
NAME_TO_SYMBOL = {
    # Crypto
    "bitcoin": "BTC-USD", "btc": "BTC-USD",
    "ethereum": "ETH-USD", "eth": "ETH-USD",
    # Big equities
    "apple": "AAPL", "tesla": "TSLA", "microsoft": "MSFT", "amazon": "AMZN",
    "google": "GOOGL", "alphabet": "GOOGL", "nvidia": "NVDA", "meta": "META",
    "facebook": "META", "netflix": "NFLX", "amd": "AMD", "intel": "INTC",
    "paypal": "PYPL", "square": "SQ", "block": "SQ", "shopify": "SHOP",
    "coca cola": "KO", "coke": "KO",
}

# Number to words conversion
_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
_TEENS = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_SCALES = [(1_000_000_000, "billion"), (1_000_000, "million"), (1_000, "thousand"), (1, "")]


def _hundreds_to_words(n: int) -> str:
    words: list[str] = []
    if n >= 100:
        words.append(_ONES[n // 100])
        words.append("hundred")
        n %= 100
    if n >= 20:
        words.append(_TENS[n // 10])
        if n % 10:
            words.append(_ONES[n % 10])
    elif n >= 10:
        words.append(_TEENS[n - 10])
    elif n > 0:
        words.append(_ONES[n])
    return " ".join(words)


def _int_to_words(n: int) -> str:
    if n == 0:
        return "zero"
    words: list[str] = []
    for scale, name in _SCALES:
        if n >= scale:
            chunk = n // scale
            if chunk:
                words.append(_hundreds_to_words(chunk))
                if name:
                    words.append(name)
            n %= scale
    return " ".join(words)


def money_to_words(amount: float, currency_code: str = "USD") -> str:
    """Convert money amount to spoken words."""
    dollars = int(amount)
    cents = int(round((amount - dollars) * 100))
    if cents == 100:
        dollars += 1
        cents = 0
    curr_map = {"USD": "dollars", "EUR": "euros", "GBP": "pounds", "JPY": "yen"}
    curr = curr_map.get((currency_code or "USD").upper(), currency_code or "USD")
    parts: list[str] = []
    parts.append(f"{_int_to_words(dollars)} {curr}" if dollars != 1 else f"one {curr[:-1] if curr.endswith('s') else curr}")
    if cents:
        cent_word = "cent" if cents == 1 else "cents"
        parts.append(f"and {_int_to_words(cents)} {cent_word}")
    return " ".join(parts)


def resolve_symbol_from_text(user_input: str) -> Optional[str]:
    """Resolve stock/crypto symbol from natural language."""
    low = (user_input or "").lower()
    
    # Multi-word keys first
    for key, sym in NAME_TO_SYMBOL.items():
        if " " in key and key in low:
            return sym
    
    # Token pass
    tokens = [t.strip(".,!?:;()[]{}\"'") for t in low.split()]
    for t in tokens:
        if t in NAME_TO_SYMBOL:
            return NAME_TO_SYMBOL[t]
    
    # Ticker heuristic: ALL-CAPS 1-5 chars
    for raw in (user_input or "").split():
        tok = raw.strip(".,!?:;()[]{}\"'")
        if tok.isupper() and 1 <= len(tok) <= 5:
            return tok
    
    return None


def handle_finance_command(
    user_input: str,
    rag_api_base: str = "http://127.0.0.1:8123",
    local_quote_func=None
) -> Tuple[bool, Optional[str]]:
    """Handle finance/price query.
    
    Args:
        user_input: User's text input
        rag_api_base: RAG API base URL
        local_quote_func: Optional local quote function fallback
        
    Returns:
        (handled, response_text) - handled is True if this was a finance query
    """
    low = (user_input or "").lower()
    
    # Check if this is a finance query
    if not any(kw in low for kw in ["price", "quote", "trading at", "stock price"]):
        return False, None
    
    symbol = resolve_symbol_from_text(user_input)
    if not symbol:
        logger.debug("Finance query but no symbol found")
        return True, "Which symbol did you mean?"
    
    logger.info(f"Finance query for symbol: {symbol}")
    
    # Try RAG API first
    data = {}
    try:
        with httpx.Client(timeout=0.5) as hc:
            resp = hc.get(f"{rag_api_base}/finance/quote", params={"symbol": symbol})
            if resp.status_code == 200:
                data = resp.json()
    except httpx.TimeoutException:
        logger.warning(f"RAG API timeout for {symbol}")
    except httpx.HTTPError as e:
        logger.warning(f"RAG API error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error fetching quote: {e}")
    
    # Fallback to local
    if (not data or data.get("regularMarketPrice") is None) and local_quote_func:
        try:
            data = local_quote_func(symbol) or {}
            logger.debug(f"Used local quote fallback for {symbol}")
        except Exception as e:
            logger.warning(f"Local quote fallback failed: {e}")
    
    price = data.get("regularMarketPrice")
    curr = data.get("currency") or "USD"
    name = data.get("shortName") or symbol
    
    if price is not None:
        response = f"{name} is {money_to_words(float(price), curr)}."
    else:
        response = f"Price unavailable for {symbol}."
    
    return True, response


def handle_finance_news(
    user_input: str,
    rag_api_base: str = "http://127.0.0.1:8123",
    local_news_func=None
) -> Tuple[bool, Optional[str]]:
    """Handle finance news query.
    
    Returns:
        (handled, response_text)
    """
    low = (user_input or "").lower()
    
    if not any(kw in low for kw in ["finance news", "stock news", "market news", "financial news"]):
        return False, None
    
    logger.info("Finance news query")
    
    titles = []
    
    # Try RAG API
    try:
        with httpx.Client(timeout=1.0) as hc:
            resp = hc.get(f"{rag_api_base}/finance/news")
            if resp.status_code == 200:
                news = resp.json()
                titles = [item.get("title", "") for item in news[:5] if item.get("title")]
    except httpx.TimeoutException:
        logger.warning("News API timeout")
    except Exception as e:
        logger.warning(f"News API error: {e}")
    
    # Fallback to local
    if not titles and local_news_func:
        try:
            news = local_news_func() or []
            titles = [item.get("title", "") for item in news[:5] if item.get("title")]
        except Exception as e:
            logger.warning(f"Local news fallback failed: {e}")
    
    if titles:
        response = "; ".join(titles[:3])
    else:
        response = "No finance news available."
    
    return True, response

