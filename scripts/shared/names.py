#!/usr/bin/env python3
"""
Player name normalization utilities.

Provides functions for normalizing and matching player names across
different data sources (auction, IPL performance, T20I, etc.).
"""

import pandas as pd


def normalize_name(name):
    """
    Normalize player name for matching.

    Converts to lowercase, removes punctuation, normalizes whitespace.

    Parameters
    ----------
    name : str
        Player name to normalize

    Returns
    -------
    str
        Normalized name
    """
    if pd.isna(name):
        return ""
    name = str(name).strip().lower()
    name = name.replace(".", " ").replace("-", " ").replace("'", "").replace("'", "")
    name = " ".join(name.split())
    return name


def get_initials_last(name):
    """
    Convert 'Virat Kohli' to 'v kohli' or 'MS Dhoni' to 'ms dhoni'.

    Parameters
    ----------
    name : str
        Normalized player name

    Returns
    -------
    str
        Name in initials-last format
    """
    parts = name.split()
    if len(parts) == 1:
        return name
    last = parts[-1]
    first_initials = "".join([p[0] if len(p) > 0 else "" for p in parts[:-1]])
    return f"{first_initials} {last}"


def get_last_name(name):
    """
    Get last name from a normalized name.

    Parameters
    ----------
    name : str
        Normalized player name

    Returns
    -------
    str
        Last name
    """
    parts = name.split()
    if len(parts) == 0:
        return ""
    return parts[-1]


def get_first_initial(name):
    """
    Get first initial from a normalized name.

    Parameters
    ----------
    name : str
        Normalized player name

    Returns
    -------
    str
        First initial
    """
    parts = name.split()
    if len(parts) == 0:
        return ""
    return parts[0][0] if parts[0] else ""


def convert_full_to_initial_format(name):
    """
    Convert 'virat kohli' to 'v kohli' format.

    Parameters
    ----------
    name : str
        Normalized player name

    Returns
    -------
    str
        Name in initial-last format
    """
    parts = name.split()
    if len(parts) < 2:
        return name
    first_initial = parts[0][0] if parts[0] else ""
    middle_initials = "".join([p[0] for p in parts[1:-1]])
    last = parts[-1]
    if middle_initials:
        return f"{first_initial}{middle_initials} {last}"
    return f"{first_initial} {last}"


def names_compatible(name1, name2):
    """
    Check if two normalized names could represent the same person.

    Compares last names exactly, and checks if first name/initials
    are compatible.

    Parameters
    ----------
    name1 : str
        First normalized name
    name2 : str
        Second normalized name

    Returns
    -------
    bool
        True if names could be same person
    """
    parts1 = name1.split()
    parts2 = name2.split()

    if len(parts1) == 0 or len(parts2) == 0:
        return False

    if parts1[-1] != parts2[-1]:
        return False

    first1 = parts1[:-1]
    first2 = parts2[:-1]

    if len(first1) == 0 or len(first2) == 0:
        return True

    f1 = first1[0]
    f2 = first2[0]

    if len(f1) <= 2:
        return f2[0] == f1[0]
    if len(f2) <= 2:
        return f1[0] == f2[0]

    return f1 == f2
