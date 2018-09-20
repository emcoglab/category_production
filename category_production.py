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

from typing import List, Set
from logging import getLogger

from pandas import DataFrame, read_csv

from .category_production_preferences import Preferences

logger = getLogger(__name__)


def _split_on_spaces(str_in: str) -> List[str]:
    return str_in.split(" ")


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

    def __init__(self,
                 word_tokenise: callable = None):
        """
        :param word_tokenise:
        (Optional.)
        If provided and not None: A function which maps strings (strings) to lists of strings (token substrings).
        Default: s ↦ s.split(" ")
        """

        if word_tokenise is None:
            word_tokenise = _split_on_spaces

        # Load and prepare data

        self.data: DataFrame = read_csv(Preferences.linguistic_wordlist_csv_path, index_col=0, header=0)
        self.rt_data: DataFrame = read_csv(Preferences.linguistic_wordlist_rt_csv_path, index_col=0, header=0)

        # Only consider unique category–response pairs
        self.data.drop_duplicates(
            subset=[CategoryProduction.ColNames.Category, CategoryProduction.ColNames.Response],
            inplace=True)

        # Hide those with production frequency 1
        self.data = self.data[self.data[CategoryProduction.ColNames.ProductionFrequency] != 1]

        # A nan in the FRF column means the first-rank frequency is zero
        # Set FRF=NAN rows to FRF=0
        self.data[CategoryProduction.ColNames.FirstRankFrequency] = self.data[CategoryProduction.ColNames.FirstRankFrequency].fillna(0)

        # Trim whitespace and convert all words to lower case
        self.data[CategoryProduction.ColNames.Category] = self.data[CategoryProduction.ColNames.Category].str.strip()
        self.data[CategoryProduction.ColNames.Category] = self.data[CategoryProduction.ColNames.Category].str.lower()
        self.data[CategoryProduction.ColNames.Response] = self.data[CategoryProduction.ColNames.Response].str.strip()
        self.data[CategoryProduction.ColNames.Response] = self.data[CategoryProduction.ColNames.Response].str.lower()

        # Apply specific substitutions.
        self.data.replace(Preferences.specific_substitutions, inplace=True)
        self.rt_data.replace(Preferences.specific_substitutions, inplace=True)

        # Build lists

        self.category_labels = sorted({category for category in self.data[CategoryProduction.ColNames.Category]})
        self.response_labels = sorted({response for response in self.data[CategoryProduction.ColNames.Response]})

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
                               sort_by: 'CategoryProduction.ColNames'=None) -> List[str]:
        """
        Responses for a provided category.
        :param category:
        :param single_word_only:
        :param sort_by: CategoryProduction.ColNames
            Default: CategoryProduction.ColNames.MeanRank
        :return:
        """
        # Set default values
        if sort_by is None:
            sort_by = CategoryProduction.ColNames.MeanRank

        # Check validity
        if category not in self.category_labels:
            raise CategoryNotFoundError(category)

        filtered_data = self.data[self.data[CategoryProduction.ColNames.Category] == category]
        filtered_data = filtered_data.sort_values(by=sort_by, ascending=True)
        filtered_data = filtered_data[CategoryProduction.ColNames.Response]

        if single_word_only:
            filtered_data = [r for r in filtered_data if " " not in r]
        else:
            filtered_data = [r for r in filtered_data]

        return filtered_data

    def data_for_category_response_pair(self,
                                        category: str,
                                        response: str,
                                        col_name: 'CategoryProduction.ColNames'):
        """Data for a category–response pair."""
        if category not in self.category_labels:
            raise CategoryNotFoundError(category)
        if response not in self.responses_for_category(category):
            raise ResponseNotFoundError(response)

        filtered_data = self.data[
            (self.data[CategoryProduction.ColNames.Category] == category)
            & (self.data[CategoryProduction.ColNames.Response] == response)]

        # We should already have dropped duplicated on loading, but just to be safe we check here
        if filtered_data.shape[0] > 1:
            logger.warning(f"Found multiple entries for {category}–{response} pair. Just using the first.")

        return filtered_data.iloc[0][col_name]

    def rts_for_category_response_pair(self,
                                       category: str,
                                       response: str) -> Set[float]:
        filtered_rt_data = self.rt_data[(self.rt_data[CategoryProduction.ColNames.Category] == category) & (self.rt_data[CategoryProduction.ColNames.Response] == response)]
        return set(filtered_rt_data["RT"])

    class ColNames(object):
        """Column names used in the data files."""
        # The category
        Category             = "Category"
        # The response (linguistic version)
        Response             = "Response"
        # The response (sensorimotor version)
        ResponseSensorimotor = "SM_term"
        # Production frequency
        ProductionFrequency  = "ProdFreq"
        # Mean rank
        MeanRank             = "MeanRank"
        # First-rank frequency
        FirstRankFrequency   = "FRF"


class TermNotFoundError(Exception):
    pass


class CategoryNotFoundError(TermNotFoundError):
    pass


class ResponseNotFoundError(TermNotFoundError):
    pass


# For debug
if __name__ == '__main__':
    cp = CategoryProduction()
    pass
