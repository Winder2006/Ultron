"""Tests for command handlers."""
import pytest
from src.commands.finance import (
    resolve_symbol_from_text,
    money_to_words,
    handle_finance_command,
)
from src.commands.weather import (
    resolve_location,
    handle_weather_command,
)
from src.commands.info_search import (
    is_info_query,
    extract_query_topic,
    is_lore_query,
    shorten_summary,
)
from src.commands.memory_commands import (
    is_remember_command,
    is_memory_query,
    extract_remember_content,
    extract_memory_query,
)


class TestFinanceCommands:
    """Tests for finance command handlers."""
    
    def test_resolve_bitcoin(self):
        """Test Bitcoin symbol resolution."""
        assert resolve_symbol_from_text("What is Bitcoin price") == "BTC-USD"
        assert resolve_symbol_from_text("btc price") == "BTC-USD"
    
    def test_resolve_stock(self):
        """Test stock symbol resolution."""
        assert resolve_symbol_from_text("Apple stock price") == "AAPL"
        assert resolve_symbol_from_text("What is TSLA trading at") == "TSLA"
    
    def test_resolve_ticker_heuristic(self):
        """Test uppercase ticker detection."""
        assert resolve_symbol_from_text("What is NVDA at") == "NVDA"
    
    def test_no_symbol(self):
        """Test no symbol found."""
        assert resolve_symbol_from_text("How is the weather") is None
    
    def test_money_to_words_simple(self):
        """Test simple money conversion."""
        assert "one hundred" in money_to_words(100.00).lower()
        assert "dollar" in money_to_words(100.00).lower()
    
    def test_money_to_words_with_cents(self):
        """Test money with cents."""
        result = money_to_words(10.50)
        assert "ten" in result.lower()
        assert "fifty" in result.lower()
        assert "cent" in result.lower()
    
    def test_handle_finance_not_finance_query(self):
        """Test non-finance query returns not handled."""
        handled, response = handle_finance_command("How is the weather")
        assert handled is False
        assert response is None


class TestWeatherCommands:
    """Tests for weather command handlers."""
    
    def test_resolve_milwaukee(self):
        """Test Milwaukee location resolution."""
        lat, lon = resolve_location("weather in milwaukee")
        assert abs(lat - 43.0389) < 0.01
    
    def test_resolve_default(self):
        """Test default location."""
        lat, lon = resolve_location("what's the weather")
        # Should return default (Milwaukee)
        assert lat is not None
        assert lon is not None
    
    def test_handle_weather_not_weather_query(self):
        """Test non-weather query returns not handled."""
        handled, response = handle_weather_command("What is bitcoin price")
        assert handled is False
        assert response is None


class TestInfoSearchCommands:
    """Tests for info search command handlers."""
    
    def test_is_info_query_who(self):
        """Test 'who is' queries."""
        assert is_info_query("Who is Albert Einstein")
        assert is_info_query("who is the president")
    
    def test_is_info_query_what(self):
        """Test 'what is' queries."""
        assert is_info_query("What is Python")
        assert is_info_query("what are neural networks")
    
    def test_is_info_query_negative(self):
        """Test non-info queries."""
        assert not is_info_query("Set a timer for 5 minutes")
        assert not is_info_query("Play some music")
    
    def test_extract_topic(self):
        """Test topic extraction."""
        assert extract_query_topic("Who is Albert Einstein") == "Albert Einstein"
        assert extract_query_topic("What is machine learning") == "machine learning"
    
    def test_is_lore_query(self):
        """Test lore query detection."""
        assert is_lore_query("Tell me about the xenomorph")
        assert is_lore_query("What is LV-426")
        assert is_lore_query("Weyland-Yutani corporation")
    
    def test_is_not_lore_query(self):
        """Test non-lore queries."""
        assert not is_lore_query("Who is Albert Einstein")
        assert not is_lore_query("What is Python")
    
    def test_shorten_summary(self):
        """Test summary shortening."""
        long_text = "This is a sentence. " * 50
        result = shorten_summary(long_text, max_chars=100)
        assert len(result) <= 110  # Allow for sentence boundary
    
    def test_shorten_summary_short(self):
        """Test short text unchanged."""
        short_text = "This is short."
        assert shorten_summary(short_text, max_chars=100) == short_text


class TestMemoryCommands:
    """Tests for memory command handlers."""
    
    def test_is_remember_command(self):
        """Test remember command detection."""
        assert is_remember_command("Remember my birthday is March 15")
        assert is_remember_command("remember that I like coffee")
        assert is_remember_command("Don't forget I work at Google")
    
    def test_is_not_remember_command(self):
        """Test non-remember commands."""
        assert not is_remember_command("What is my birthday")
        assert not is_remember_command("Tell me about Python")
    
    def test_extract_remember_content(self):
        """Test remember content extraction."""
        content = extract_remember_content("Remember my birthday is March 15")
        assert content is not None
        assert "birthday" in content.lower() or "march" in content.lower()
    
    def test_is_memory_query(self):
        """Test memory query detection."""
        assert is_memory_query("What is my birthday")
        assert is_memory_query("What's my favorite color")
        assert is_memory_query("Do you know my name")
    
    def test_is_not_memory_query(self):
        """Test non-memory queries."""
        assert not is_memory_query("What is the weather")
        assert not is_memory_query("Who is Albert Einstein")
    
    def test_extract_memory_query_topic(self):
        """Test memory query topic extraction."""
        topic = extract_memory_query("What is my birthday")
        assert topic is not None
        assert "birthday" in topic.lower()

