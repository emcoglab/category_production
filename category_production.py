"""
===========================
Data for Experiment 1.4 - Category Production.
===========================

Dr. Briony Banks
Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2018
---------------------------
"""
from functools import partial
from logging import getLogger
from os import path
from typing import List

import yaml
from numpy import mean, nan
from pandas import DataFrame, read_csv

logger = getLogger(__name__)


class ColNames(object):
    """Column names used in the data files."""
    # The category
    Category             = "Category"
    # The response (linguistic version)
    Response             = "Response"
    # Production frequency
    ProductionFrequency  = "Production.frequency"
    # The response (sensorimotor version)
    ResponseSensorimotor = "Sensorimotor.version"
    # Mean rank
    MeanRank             = "Mean.rank"
    # First-rank frequency
    FirstRankFrequency   = "First.rank.frequency"
    # Mean reaction time for first responses
    MeanRT               = "Mean RT"
    # Mean standardised reaction time for first responses
    MeanZRT              = "Mean zRT"


class CategoryProduction(object):
    """
    Represents the dataset for Experiment 1.4 - Category Production.
    In particular, the BrEng version to be used for the linguistic model.
    """

    _ignored_words = {
        # articles
        "a", "the",
        # prepositions
        "of",
        # other
        "and", "'s",
        # punctuation
        ".",
    }

    _data_filename = 'Category Production Data (osf_v1).csv'
    _rt_data_filename = '1.4_RT data_ALL.csv'

    _specific_substutitions_filename = 'specific_substitutions.yaml'

    @classmethod
    def _default_word_tokenise(cls, x):
        """Default tokeniser to use if none is provided."""
        return x.split(" ")

    def __init__(self,
                 word_tokenise: callable = None):
        """
        :param word_tokenise:
        (Optional.)
        If provided and not None: A function which maps strings (strings) to lists of strings (token substrings).
        Default: s ↦ s.split(" ")
        """

        # If no tokeniser given, use default
        if word_tokenise is None:
            word_tokenise = CategoryProduction._default_word_tokenise

        # Prepare substitution dictionaries

        # Specific substitutions to correct typos etc.
        with open(path.join(path.dirname(path.realpath(__file__)), CategoryProduction._specific_substutitions_filename), mode="r", encoding="utf-8") as specific_substitutions_file:
            self._specific_substitutions = yaml.load(specific_substitutions_file, yaml.SafeLoader)

        # Load and prepare data

        self.data: DataFrame = read_csv(path.join(path.dirname(path.realpath(__file__)), CategoryProduction._data_filename), index_col=None, header=0)
        rt_data: DataFrame = read_csv(path.join(path.dirname(path.realpath(__file__)), CategoryProduction._rt_data_filename), index_col=0, header=0)

        # Only consider unique category–response pairs
        self.data.drop_duplicates(
            subset=[ColNames.Category, ColNames.Response],
            inplace=True)

        # Drop columns which disambiguated duplicate entries
        self.data.drop(['Item.number', 'Participant', 'Trial.no.', 'Rank'], axis=1, inplace=True)

        # Hide those with production frequency 1
        self.data = self.data[self.data[ColNames.ProductionFrequency] != 1]

        # A nan in the FRF column means the first-rank frequency is zero
        # Set FRF=NAN rows to FRF=0 and convert to int
        self.data[ColNames.FirstRankFrequency] = self.data[ColNames.FirstRankFrequency].fillna(0).astype(int)

        # Trim whitespace and convert all words to lower case
        self.data[ColNames.Category] = self.data[ColNames.Category].str.strip()
        self.data[ColNames.Category] = self.data[ColNames.Category].str.lower()
        self.data[ColNames.Response] = self.data[ColNames.Response].str.strip()
        self.data[ColNames.Response] = self.data[ColNames.Response].str.lower()

        # Apply specific substitutions.
        self.data.replace(self._specific_substitutions, inplace=True)
        rt_data.replace(self._specific_substitutions, inplace=True)

        self.data[ColNames.MeanRT]  = self.data.apply(partial(_get_mean_rt, rt_data=rt_data, use_zrt=False), axis=1)
        self.data[ColNames.MeanZRT] = self.data.apply(partial(_get_mean_rt, rt_data=rt_data, use_zrt=True), axis=1)

        # Build lists

        self.category_labels = sorted({category for category in self.data[ColNames.Category]})
        self.response_labels = sorted({response for response in self.data[ColNames.Response]})

        # Build vocab lists

        # All multi-word tokens in the dataset
        self.vocabulary_multi_word  = sorted(set(self.category_labels)
                                             | set(self.response_labels))
        # All single-word tokens in the dataset
        self.vocabulary_single_word = sorted(set(word
                                                 for vocab_item in self.vocabulary_multi_word
                                                 for word in word_tokenise(vocab_item)
                                                 if word not in CategoryProduction._ignored_words))

    def responses_for_category(self,
                               category: str,
                               single_word_only: bool = False,
                               sort_by: 'ColNames'=None) -> List[str]:
        """
        Responses for a provided category.
        :param category:
        :param single_word_only:
        :param sort_by: ColNames
            Default: ColNames.MeanRank
        :return:
        """
        # Set default values
        if sort_by is None:
            sort_by = ColNames.MeanRank

        # Check validity
        if category not in self.category_labels:
            raise CategoryNotFoundError(category)

        filtered_data = self.data[self.data[ColNames.Category] == category]
        filtered_data = filtered_data.sort_values(by=sort_by, ascending=True)
        filtered_data = filtered_data[ColNames.Response]

        if single_word_only:
            filtered_data = [r for r in filtered_data if " " not in r]
        else:
            filtered_data = [r for r in filtered_data]

        return filtered_data

    def data_for_category_response_pair(self,
                                        category: str,
                                        response: str,
                                        col_name: 'ColNames'):
        """Data for a category–response pair."""
        if category not in self.category_labels:
            raise CategoryNotFoundError(category)
        if response not in self.responses_for_category(category):
            raise ResponseNotFoundError(response)

        filtered_data = self.data[
            (self.data[ColNames.Category] == category)
            & (self.data[ColNames.Response] == response)]

        # We should already have dropped duplicated on loading, but just to be safe we check here
        if filtered_data.shape[0] > 1:
            logger.warning(f"Found multiple entries for {category}–{response} pair. Just using the first.")

        return filtered_data.iloc[0][col_name]


class TermNotFoundError(Exception):
    pass


class CategoryNotFoundError(TermNotFoundError):
    pass


class ResponseNotFoundError(TermNotFoundError):
    pass


def _get_mean_rt(row, rt_data: DataFrame, use_zrt: bool):
    filtered_rt_data = rt_data[(rt_data[ColNames.Category] == row[ColNames.Category])
                               & (rt_data[ColNames.Response] == row[ColNames.Response])]

    if use_zrt:
        rts = list(filtered_rt_data["zscore_per_pt"])
    else:
        rts = list(filtered_rt_data["RT"])
    if rts:
        return mean(rts)
    else:
        return nan


# For debug
if __name__ == '__main__':
    cp = CategoryProduction()
    pass
