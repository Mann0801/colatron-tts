"""Text cleaning + a FIXED character-to-id mapping (no randomness anywhere).

clean_text("Dr. Smith paid $42.50 in 1984.")
    -> "doctor smith paid forty two dollars and fifty cents in nineteen eighty four."
"""
import re
import unicodedata

# ----------------------------------------------------------------------------- vocabulary
# The order below is the mapping. It never changes between runs or machines.
PAD = "<pad>"
SYMBOLS = [PAD, " "] + list("abcdefghijklmnopqrstuvwxyz") + list(",.?!'-;:\"")
SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}
ID_TO_SYMBOL = {i: s for s, i in SYMBOL_TO_ID.items()}
VOCAB_SIZE = len(SYMBOLS)
_ALLOWED = set(SYMBOLS) - {PAD}


def encode(text):
    """Characters -> fixed ids. Characters outside the vocabulary are skipped."""
    return [SYMBOL_TO_ID[c] for c in text if c in SYMBOL_TO_ID]


def decode(ids):
    return "".join(ID_TO_SYMBOL[i] for i in ids if i != 0)


# ----------------------------------------------------------------------------- numbers
_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_SCALES = [(10 ** 9, "billion"), (10 ** 6, "million"), (10 ** 3, "thousand")]
_ORD_IRREGULAR = {
    "one": "first", "two": "second", "three": "third", "five": "fifth",
    "eight": "eighth", "nine": "ninth", "twelve": "twelfth",
}


def number_to_words(n):
    """42 -> 'forty two', 1305 -> 'one thousand three hundred five'."""
    if n < 0:
        return "minus " + number_to_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _TENS[tens] + ("" if ones == 0 else " " + _ONES[ones])
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        return _ONES[hundreds] + " hundred" + ("" if rest == 0 else " " + number_to_words(rest))
    if n >= 10 ** 12:  # absurdly long: read digit by digit
        return " ".join(_ONES[int(d)] for d in str(n))
    for scale, name in _SCALES:
        if n >= scale:
            q, r = divmod(n, scale)
            return number_to_words(q) + " " + name + ("" if r == 0 else " " + number_to_words(r))
    raise ValueError(n)  # unreachable


def ordinal_to_words(n):
    """3 -> 'third', 21 -> 'twenty first', 40 -> 'fortieth'."""
    words = number_to_words(n).split(" ")
    last = words[-1]
    if last in _ORD_IRREGULAR:
        last = _ORD_IRREGULAR[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last += "th"
    words[-1] = last
    return " ".join(words)


def year_to_words(n):
    """1984 -> 'nineteen eighty four', 1905 -> 'nineteen oh five', 1900 -> 'nineteen hundred'."""
    hi, lo = divmod(n, 100)
    if lo == 0:
        return number_to_words(hi) + " hundred"
    if lo < 10:
        return number_to_words(hi) + " oh " + _ONES[lo]
    return number_to_words(hi) + " " + number_to_words(lo)


# ----------------------------------------------------------------------------- abbreviations
_ABBREVIATIONS = {
    "mr": "mister", "mrs": "missus", "ms": "miss", "dr": "doctor", "st": "saint",
    "jr": "junior", "sr": "senior", "rev": "reverend", "hon": "honorable",
    "gen": "general", "col": "colonel", "capt": "captain", "lt": "lieutenant",
    "sgt": "sergeant", "gov": "governor", "maj": "major", "messrs": "messieurs",
    "mt": "mount", "ft": "fort", "co": "company", "corp": "corporation",
    "ltd": "limited", "inc": "incorporated", "vs": "versus", "etc": "et cetera",
    "esq": "esquire", "prof": "professor",
}
_ABBR_RE = re.compile(
    r"\b(" + "|".join(sorted((re.escape(k) for k in _ABBREVIATIONS), key=len, reverse=True)) + r")\.",
    re.IGNORECASE,
)
_NO_RE = re.compile(r"\bno\.\s*(?=\d)", re.IGNORECASE)

_QUOTE_MAP = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "…": ".",
})


def _money(m):
    symbol, whole, cents = m.group(1), m.group(2).replace(",", ""), m.group(3)
    unit = "dollar" if symbol == "$" else "pound"
    n = int(whole)
    words = number_to_words(n) + " " + (unit if n == 1 else unit + "s")
    if cents and int(cents) > 0:
        c = int(cents)
        if symbol == "$":
            small = "cent" if c == 1 else "cents"
        else:
            small = "penny" if c == 1 else "pence"
        words += " and " + number_to_words(c) + " " + small
    return " " + words + " "


def _decimal(m):
    return " " + number_to_words(int(m.group(1))) + " point " + " ".join(_ONES[int(d)] for d in m.group(2)) + " "


def clean_text(text):
    """Lowercase, expand abbreviations and numbers, drop unsupported characters."""
    text = text.translate(_QUOTE_MAP)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower()

    text = _NO_RE.sub("number ", text)
    text = _ABBR_RE.sub(lambda m: _ABBREVIATIONS[m.group(1).lower()] + " ", text)

    text = re.sub(r"([$£])(\d+(?:,\d{3})*)(?:\.(\d{2})(?!\d))?", _money, text)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"\1 percent", text)
    text = re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", lambda m: " " + ordinal_to_words(int(m.group(1))) + " ", text)
    text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)  # 1,234 -> 1234
    text = re.sub(r"\b(\d+)\.(\d+)\b", _decimal, text)
    text = re.sub(r"\b(1[1-9]\d\d|20[1-9]\d)\b", lambda m: " " + year_to_words(int(m.group(1))) + " ", text)
    text = re.sub(r"\d+", lambda m: " " + number_to_words(int(m.group(0))) + " ", text)

    for symbol, word in (("&", " and "), ("@", " at "), ("+", " plus "), ("=", " equals "), ("#", " number ")):
        text = text.replace(symbol, word)

    text = "".join(c if c in _ALLOWED else " " for c in text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.?!;:])", r"\1", text)
    return text


if __name__ == "__main__":
    for sample in (
        "Dr. Smith paid $42.50 on March 3rd, 1984.",
        "Mr. and Mrs. Jones bought 1,200 books (about 12%).",
        "No. 7 Main St. was built in 1905.",
    ):
        cleaned = clean_text(sample)
        print(sample, "->", cleaned)
        print("   ids:", encode(cleaned)[:20], "...")
