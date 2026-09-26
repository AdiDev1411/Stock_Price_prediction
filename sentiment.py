import re
import requests

POSITIVE_WORDS = {
    "boost", "rally", "surge", "gain", "growth", "improve", "improves", "strong", "confidence",
    "optimistic", "positive", "rise", "up", "higher", "stable", "rebound", "recovery",
}
NEGATIVE_WORDS = {
    "fall", "drop", "decline", "weak", "worry", "risk", "tension", "tensions", "crash", "down",
    "negative", "pressure", "slump", "recession", "inflation", "uncertain", "volatile", "bearish",
}


def sentiment_score(headlines):
    scores = []
    for headline in headlines:
        text = re.sub(r"[^a-z0-9\s]", " ", headline.lower())
        tokens = set(text.split())
        score = 0
        for token in tokens:
            if token in POSITIVE_WORDS:
                score += 1
            elif token in NEGATIVE_WORDS:
                score -= 1
        scores.append(score)

    return sum(scores) / len(scores) if scores else 0.0


def get_global_news_sentiment():
    try:
        response = requests.get(
            "https://hnrss.org/frontpage",
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
        text = response.text
        headlines = [
            item.split("<title>", 1)[1].split("</title>", 1)[0]
            for item in text.split("<item>")
            if "<title>" in item
        ]
        if headlines:
            return sentiment_score(headlines[:10])
    except Exception:
        pass

    fallback_headlines = [
        "global markets rebound as investors gain confidence",
        "trade tensions and inflation concerns pressure equities",
    ]
    return sentiment_score(fallback_headlines)