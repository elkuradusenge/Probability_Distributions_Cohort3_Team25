"""Bayes on IMDb reviews with stdlib only (csv + re)."""
import csv
import re

WORD_RE = re.compile(r"[a-z']+")
