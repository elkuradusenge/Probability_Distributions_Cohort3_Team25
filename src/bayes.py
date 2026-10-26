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


def bayes_posterior(prior, likelihood, marginal):
    # P(Pos|kw) = P(kw|Pos) * P(Pos) / P(kw)
    return likelihood * prior / marginal


def keyword_table(reviews, keywords):
    # one row per keyword: prior, likelihood, marginal, posterior (+ direct check)
    n = len(reviews)
    n_pos = sum(s == "positive" for _, s in reviews)
    prior = n_pos / n
    rows = []
    for kw in keywords:
        n_kw, n_kw_pos = count_keyword(reviews, kw)
        likelihood = n_kw_pos / n_pos
        marginal = n_kw / n
        rows.append(dict(
            keyword=kw,
            prior=prior,
            likelihood=likelihood,
            marginal=marginal,
            posterior=bayes_posterior(prior, likelihood, marginal),
            direct=n_kw_pos / n_kw,
            n_docs=n_kw,
        ))
    return rows
