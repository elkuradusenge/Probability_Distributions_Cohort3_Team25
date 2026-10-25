"""Bayes on IMDb reviews with stdlib only (csv + re)."""
import csv
import re

WORD_RE = re.compile(r"[a-z']+")


def load_reviews(path):
    # pull each review into a word set + its sentiment label
    reviews = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text = row["review"].lower().replace("<br />", " ")
            reviews.append((set(WORD_RE.findall(text)), row["sentiment"]))
    return reviews


def count_keyword(reviews, keyword):
    # count docs with the word, and how many of those are positive
    total = positive = 0
    for words, sentiment in reviews:
        if keyword in words:
            total += 1
            positive += sentiment == "positive"
    return total, positive
