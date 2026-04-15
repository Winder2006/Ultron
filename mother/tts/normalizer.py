"""Text normalization for TTS - converts dates, numbers, and symbols to spoken form."""
from __future__ import annotations

import re
from typing import Optional

# -------- Number word mappings --------
_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
         "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
         "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_ORDINALS = {
    1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
    6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth",
    11: "eleventh", 12: "twelfth", 13: "thirteenth", 14: "fourteenth", 15: "fifteenth",
    16: "sixteenth", 17: "seventeenth", 18: "eighteenth", 19: "nineteenth",
    20: "twentieth", 21: "twenty-first", 22: "twenty-second", 23: "twenty-third",
    24: "twenty-fourth", 25: "twenty-fifth", 26: "twenty-sixth", 27: "twenty-seventh",
    28: "twenty-eighth", 29: "twenty-ninth", 30: "thirtieth", 31: "thirty-first",
}
_MONTHS = {
    1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
    7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December",
}
_MONTH_NAMES = {v.lower(): v for v in _MONTHS.values()}


def _two_digit_to_words(n: int) -> str:
    """Convert 0-99 to words."""
    if n < 20:
        return _ONES[n]
    tens, ones = divmod(n, 10)
    if ones == 0:
        return _TENS[tens]
    return f"{_TENS[tens]}-{_ONES[ones]}"


def _number_under_1000(n: int) -> str:
    """Convert 0-999 to words."""
    if n < 100:
        return _two_digit_to_words(n)
    hundreds, remainder = divmod(n, 100)
    if remainder == 0:
        return f"{_ONES[hundreds]} hundred"
    return f"{_ONES[hundreds]} hundred {_two_digit_to_words(remainder)}"


def year_to_words(year: int) -> str:
    """Convert a year to spoken form.
    
    Examples:
        1990 → "nineteen ninety"
        2001 → "two thousand one"
        2023 → "twenty twenty-three"
        1800 → "eighteen hundred"
        2000 → "two thousand"
    """
    if year < 0 or year > 9999:
        return str(year)
    
    # Special cases for 2000s
    if year == 2000:
        return "two thousand"
    if 2001 <= year <= 2009:
        return f"two thousand {_ONES[year - 2000]}"
    if 2010 <= year <= 2019:
        return f"twenty {_ONES[year - 2010]}" if year > 2010 else "twenty ten"
    
    # Standard century-style (nineteen ninety, twenty twenty-three)
    if year >= 1000:
        first_half = year // 100
        second_half = year % 100
        
        # Handle years like 1800, 1900
        if second_half == 0:
            return f"{_two_digit_to_words(first_half)} hundred"
        
        # Handle years like 1901, 1805
        if second_half < 10:
            return f"{_two_digit_to_words(first_half)} oh {_ONES[second_half]}"
        
        return f"{_two_digit_to_words(first_half)} {_two_digit_to_words(second_half)}"
    
    # Years < 1000
    return _number_under_1000(year)


def ordinal_to_words(n: int) -> str:
    """Convert a number to its ordinal form (1 → first, 22 → twenty-second)."""
    if n in _ORDINALS:
        return _ORDINALS[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        if ones == 0:
            base = _TENS[tens]
            return base[:-1] + "ieth" if base.endswith("y") else base + "th"
        return f"{_TENS[tens]}-{_ORDINALS.get(ones, _ONES[ones] + 'th')}"
    return f"{_number_under_1000(n)}th"


def _normalize_year_in_text(text: str) -> str:
    """Find standalone 4-digit years and convert them."""
    def replace_year(m: re.Match) -> str:
        year = int(m.group(1))
        # Only convert plausible years (1000-2099)
        if 1000 <= year <= 2099:
            return year_to_words(year)
        return m.group(0)
    
    # Match 4-digit numbers that look like years (not part of larger numbers)
    # Negative lookbehind to avoid matching after digits; allow periods/punctuation after
    pattern = r'(?<![0-9])\b(1[0-9]{3}|20[0-9]{2})\b(?![0-9])'
    return re.sub(pattern, replace_year, text)


def _normalize_ordinal_suffix(text: str) -> str:
    """Convert 1st, 2nd, 3rd, etc. to spoken form."""
    def replace_ordinal(m: re.Match) -> str:
        num = int(m.group(1))
        return ordinal_to_words(num)
    
    # Match numbers with ordinal suffixes
    pattern = r'\b(\d{1,2})(st|nd|rd|th)\b'
    return re.sub(pattern, replace_ordinal, text, flags=re.IGNORECASE)


def _normalize_date_formats(text: str) -> str:
    """Convert various date formats to spoken form."""
    
    # MM/DD/YYYY or M/D/YYYY
    def replace_slash_date(m: re.Match) -> str:
        month, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31:
            month_name = _MONTHS.get(month, str(month))
            return f"{month_name} {ordinal_to_words(day)}, {year_to_words(year)}"
        return m.group(0)
    
    text = re.sub(r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b', replace_slash_date, text)
    
    # YYYY-MM-DD (ISO format)
    def replace_iso_date(m: re.Match) -> str:
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31:
            month_name = _MONTHS.get(month, str(month))
            return f"{month_name} {ordinal_to_words(day)}, {year_to_words(year)}"
        return m.group(0)
    
    text = re.sub(r'\b(\d{4})-(\d{2})-(\d{2})\b', replace_iso_date, text)
    
    # Month DD, YYYY or Month DDth, YYYY
    def replace_written_date(m: re.Match) -> str:
        month_name = m.group(1).capitalize()
        day = int(m.group(2))
        year = int(m.group(3))
        return f"{month_name} {ordinal_to_words(day)}, {year_to_words(year)}"
    
    months_pattern = '|'.join(_MONTH_NAMES.keys())
    text = re.sub(
        rf'\b({months_pattern})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s*(\d{{4}})\b',
        replace_written_date, text, flags=re.IGNORECASE
    )
    
    return text


def _normalize_time(text: str) -> str:
    """Convert times like 3:30 PM to spoken form."""
    def replace_time(m: re.Match) -> str:
        hour, minute = int(m.group(1)), int(m.group(2))
        period = (m.group(3) or "").upper()
        
        hour_word = _two_digit_to_words(hour)
        if minute == 0:
            time_str = f"{hour_word} o'clock"
        elif minute == 30:
            time_str = f"half past {hour_word}"
        elif minute == 15:
            time_str = f"quarter past {hour_word}"
        elif minute == 45:
            next_hour = hour + 1 if hour < 12 else 1
            time_str = f"quarter to {_two_digit_to_words(next_hour)}"
        else:
            min_word = _two_digit_to_words(minute) if minute >= 10 else f"oh {_ONES[minute]}"
            time_str = f"{hour_word} {min_word}"
        
        if period:
            time_str += f" {period.replace('.', '')}"
        return time_str
    
    # Match times like 3:30, 12:45 PM, 9:05 a.m.
    pattern = r'\b(\d{1,2}):(\d{2})\s*([APap]\.?[Mm]\.?)?\b'
    return re.sub(pattern, replace_time, text)


def _normalize_percentages(text: str) -> str:
    """Convert 50% to 'fifty percent'."""
    def replace_pct(m: re.Match) -> str:
        num = m.group(1)
        if '.' in num:
            whole, dec = num.split('.')
            return f"{_number_under_1000(int(whole))} point {' '.join(_ONES[int(d)] for d in dec)} percent"
        return f"{_number_under_1000(int(num))} percent"
    
    return re.sub(r'\b(\d+(?:\.\d+)?)\s*%', replace_pct, text)


def _normalize_abbreviations(text: str) -> str:
    """Expand common abbreviations."""
    abbrevs = {
        r'\bDr\.': 'Doctor',
        r'\bMr\.': 'Mister',
        r'\bMrs\.': 'Missus',
        r'\bMs\.': 'Miss',
        r'\bSt\.': 'Saint',
        r'\bvs\.?': 'versus',
        r'\betc\.': 'etcetera',
        r'\be\.g\.': 'for example',
        r'\bi\.e\.': 'that is',
        r'\bw/': 'with',
        r'\bw/o': 'without',
        r'\bLV-426\b': 'L V four twenty-six',
        r'\bAI\b': 'A I',
        r'\bUI\b': 'U I',
        r'\bAPI\b': 'A P I',
        r'\bURL\b': 'U R L',
        r'\bUSA\b': 'U S A',
        r'\bUK\b': 'U K',
    }
    for pattern, replacement in abbrevs.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def _strip_markdown(text: str) -> str:
    """Remove markdown formatting that would get read aloud by TTS.

    TTS engines literally speak asterisks as "star", underscores as "underscore",
    backticks confusingly, etc. Strip them before synthesis.
    """
    # Bold/italic: **word** or __word__ → word
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    # Single emphasis: *word* or _word_ → word
    text = re.sub(r'(?<!\w)\*([^*\n]+?)\*(?!\w)', r'\1', text)
    text = re.sub(r'(?<!\w)_([^_\n]+?)_(?!\w)', r'\1', text)
    # Inline code: `word` → word
    text = re.sub(r'`([^`\n]+)`', r'\1', text)
    # Bullet points: "- " or "* " at line start
    text = re.sub(r'(?m)^[\s]*[-*]\s+', '', text)
    # Numbered lists: "1. " at line start
    text = re.sub(r'(?m)^[\s]*\d+\.\s+', '', text)
    # Headers: "# " at line start
    text = re.sub(r'(?m)^[\s]*#+\s+', '', text)
    # Any stray asterisks or underscores that didn't match a pair
    text = text.replace('*', '').replace('`', '')
    return text


def _normalize_symbols(text: str) -> str:
    """Convert symbols to spoken form."""
    symbols = {
        '&': ' and ',
        '@': ' at ',
        '#': ' number ',
        '+': ' plus ',
        '=': ' equals ',
        '...': ', ',
        '—': ', ',
        '–': ', ',
    }
    for sym, word in symbols.items():
        text = text.replace(sym, word)
    return text


def normalize_for_speech(text: str) -> str:
    """Apply all text normalizations for natural TTS output.
    
    This function processes text to make it sound natural when spoken:
    - Years: 1990 → "nineteen ninety"
    - Dates: March 15, 2023 → "March fifteenth, twenty twenty-three"
    - Ordinals: 1st, 2nd, 3rd → "first", "second", "third"
    - Times: 3:30 PM → "three thirty PM"
    - Percentages: 50% → "fifty percent"
    - Abbreviations: Dr., Mr., etc.
    - Symbols: &, @, #
    """
    if not text:
        return text

    # Strip markdown FIRST so formatting symbols don't get read aloud
    text = _strip_markdown(text)

    # Apply normalizations in order (order matters!)
    text = _normalize_abbreviations(text)
    text = _normalize_date_formats(text)
    text = _normalize_ordinal_suffix(text)
    text = _normalize_year_in_text(text)
    text = _normalize_time(text)
    text = _normalize_percentages(text)
    text = _normalize_symbols(text)
    
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

