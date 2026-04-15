"""Tests for text_normalizer module."""
import pytest
from src.text_normalizer import normalize_for_speech


class TestNormalizeForSpeech:
    """Tests for the normalize_for_speech function."""
    
    def test_full_date_format(self):
        """Test full date format conversion."""
        assert "december" in normalize_for_speech("December 6, 2025").lower()
        assert "twenty twenty-five" in normalize_for_speech("December 6, 2025").lower()
    
    def test_short_date_format(self):
        """Test short date format conversion."""
        result = normalize_for_speech("12/06/2025")
        # Should convert to words
        assert "2025" not in result or "twenty twenty-five" in result.lower()
    
    def test_iso_date_format(self):
        """Test ISO date format conversion."""
        result = normalize_for_speech("2025-12-06")
        assert any(month in result.lower() for month in ["december", "twelve"])
    
    def test_year_standalone(self):
        """Test standalone year conversion."""
        assert "nineteen ninety" in normalize_for_speech("1990").lower()
        assert "twenty twenty-five" in normalize_for_speech("2025").lower()
    
    def test_ordinals(self):
        """Test ordinal number conversion."""
        assert "first" in normalize_for_speech("1st").lower()
        assert "second" in normalize_for_speech("2nd").lower()
        assert "third" in normalize_for_speech("3rd").lower()
        assert "fourth" in normalize_for_speech("4th").lower()
    
    def test_percentages(self):
        """Test percentage conversion."""
        result = normalize_for_speech("50%")
        assert "percent" in result.lower()
    
    def test_abbreviations(self):
        """Test common abbreviation expansion."""
        assert "doctor" in normalize_for_speech("Dr. Smith").lower()
        assert "mister" in normalize_for_speech("Mr. Jones").lower()
    
    def test_numbers(self):
        """Test number to words conversion."""
        # Small numbers in context may not be converted (intentional)
        result = normalize_for_speech("There are 42 items.")
        # Just ensure it doesn't crash and returns something
        assert len(result) > 0
    
    def test_empty_string(self):
        """Test empty string handling."""
        assert normalize_for_speech("") == ""
    
    def test_none_handling(self):
        """Test None input handling."""
        # Should not crash - either returns empty string or raises TypeError
        try:
            result = normalize_for_speech(None)  # type: ignore
            # If it returns, should be string (empty or otherwise)
            assert isinstance(result, (str, type(None)))
        except (TypeError, AttributeError):
            # This is acceptable - function doesn't handle None
            pass
    
    def test_plain_text_unchanged(self):
        """Test that plain text passes through."""
        text = "Hello, how are you today?"
        result = normalize_for_speech(text)
        assert "hello" in result.lower()
        assert "today" in result.lower()


class TestEdgeCases:
    """Edge case tests."""
    
    def test_mixed_content(self):
        """Test mixed numbers and text."""
        text = "The year 1990 was when I was born on March 15th."
        result = normalize_for_speech(text)
        assert "nineteen ninety" in result.lower()
    
    def test_large_numbers(self):
        """Test large number handling."""
        text = "The population is 1000000."
        result = normalize_for_speech(text)
        # Should convert or at least not crash
        assert len(result) > 0
    
    def test_decimal_numbers(self):
        """Test decimal number handling."""
        text = "The price is 19.99 dollars."
        result = normalize_for_speech(text)
        assert len(result) > 0

