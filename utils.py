"""
===========================
Utility functions.
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2019
---------------------------
"""


def unique(ls: list) -> list:
    """
    Returns an ordred list of the entries in `ls`, without repetitions.
    Only preserves order in Python 3.7+/
    Thanks https://stackoverflow.com/a/7961390/2883198
    :param ls:
    :return:
    """
    return list(dict.fromkeys(ls))
