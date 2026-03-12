"""Grade sorting utilities.

Maps climbing grades from different systems to numeric sort keys.
Supports Hueco (V-scale), French, YDS, and Font systems.
"""

import re
from typing import Tuple

# Hueco V-scale: V0, V1, ... V17
_HUECO_RE = re.compile(r"^V(\d+)$")

# French sport: 4a, 5a, 5b, 5c, 6a, 6a+, 6b, 6b+, ... 9c+
_FRENCH_LETTER_MAP = {"a": 0, "b": 1, "c": 2}
_FRENCH_RE = re.compile(r"^(\d)([abc])(\+?)$")

# YDS: 5.0, 5.1, ... 5.9, 5.10a, 5.10b, ... 5.15d
_YDS_RE = re.compile(r"^5\.(\d+)([a-d]?)$")
_YDS_LETTER_MAP = {"": 0, "a": 0, "b": 1, "c": 2, "d": 3}

# Font (bouldering): 4, 4+, 5, 5+, 6A, 6A+, 6B, 6B+, 6C, 6C+, 7A, ... 8C+
_FONT_LETTER_MAP = {"": 0, "a": 0, "b": 1, "c": 2}
_FONT_RE = re.compile(r"^(\d)([ABC]?)(\+?)$", re.IGNORECASE)

_FALLBACK = (999, 999)


def grade_sort_key(grade: str, system: str) -> Tuple[int, int]:
    """Return a (major, minor) sort key for a grade string."""
    try:
        if system == "hueco":
            m = _HUECO_RE.match(grade)
            if m:
                return (int(m.group(1)), 0)
        elif system == "french":
            m = _FRENCH_RE.match(grade)
            if m:
                num, letter, plus = m.groups()
                minor = _FRENCH_LETTER_MAP[letter] * 2 + (1 if plus else 0)
                return (int(num), minor)
        elif system == "yds":
            m = _YDS_RE.match(grade)
            if m:
                num, letter = m.groups()
                minor = _YDS_LETTER_MAP.get(letter, 0)
                return (int(num), minor)
        elif system == "font":
            m = _FONT_RE.match(grade)
            if m:
                num, letter, plus = m.groups()
                letter_val = _FONT_LETTER_MAP.get(letter.lower(), 0)
                minor = letter_val * 2 + (1 if plus else 0)
                return (int(num), minor)
    except (ValueError, KeyError):
        pass
    return _FALLBACK
