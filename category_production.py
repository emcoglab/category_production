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
from typing import List

from numpy import mean, nan
from pandas import DataFrame, read_csv

from preferences import Preferences

logger = getLogger(__name__)


class ColNames(object):
    """Column names used in the data files."""
    
    # Columns from main data file

    # The category
    Category             = "Category"
    # The response (linguistic version)
    Response             = "Response"
    # The category (sensorimotor version)
    CategorySensorimotor = "SM_category"
    # The response (sensorimotor version)
    ResponseSensorimotor = "SM_term"
    # Production frequency
    ProductionFrequency  = "ProdFreq"
    # Mean rank
    MeanRank             = "MeanRank"
    # First-rank frequency
    FirstRankFrequency   = "FRF"

    # Columns for RT data

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

    @classmethod
    def _default_word_tokenise(cls, x):
        """Default tokeniser to use if none is provided."""
        return x.split(" ")

    def __init__(self,
                 minimum_production_frequency: int = 2,
                 word_tokenise: callable = None):
        """
        :param minimum_production_frequency:
            (Optional.)
            Filters all data by that which has production frequency < this value.
            For example, a value to 2 will exclude all idiosyncratic responses.
            Use 1 to include all data.
            Default: 2 (i.e. all non-idiosyncratic responses).
        :param word_tokenise:
            (Optional.)
            If provided and not None: A function which maps strings (strings) to lists of strings (token substrings).
            Default: s ↦ s.split(" ")
        """

        # Validate arguments
        if minimum_production_frequency < 1:
            raise ValueError("minimum_production_frequency must be at least 1")

        # If no tokeniser given, use default
        if word_tokenise is None:
            word_tokenise = CategoryProduction._default_word_tokenise

        # Load and prepare data

        self.data: DataFrame = read_csv(Preferences.linguistic_wordlist_csv_path, index_col=None, header=0)
        rt_data: DataFrame = read_csv(Preferences.linguistic_wordlist_rt_csv_path, index_col=0, header=0)

        # Only consider unique category–response pairs
        self.data.drop_duplicates(
            subset=[ColNames.Category, ColNames.Response],
            inplace=True)

        # Drop columns which disambiguated duplicate entries
        self.data.drop(['Item', 'Participant', 'Trial.no.', 'Rank'], axis=1, inplace=True)

        # Hide those with minimum production frequency
        self.data = self.data[self.data[ColNames.ProductionFrequency] >= minimum_production_frequency]

        # A nan in the FRF column means the first-rank frequency is zero
        # Set FRF=NAN rows to FRF=0 and convert to int
        self.data[ColNames.FirstRankFrequency] = self.data[ColNames.FirstRankFrequency].fillna(0).astype(int)

        # Trim whitespace and convert all words to lower case
        self.data[ColNames.Category] = self.data[ColNames.Category].str.strip()
        self.data[ColNames.Category] = self.data[ColNames.Category].str.lower()
        self.data[ColNames.Response] = self.data[ColNames.Response].str.strip()
        self.data[ColNames.Response] = self.data[ColNames.Response].str.lower()

        self.data[ColNames.MeanRT]  = self.data.apply(partial(_get_mean_rt, rt_data=rt_data, use_zrt=False), axis=1)
        self.data[ColNames.MeanZRT] = self.data.apply(partial(_get_mean_rt, rt_data=rt_data, use_zrt=True), axis=1)

        # Build lists

        self.category_labels              = sorted({category for category in self.data[ColNames.Category]})
        self.category_labels_sensorimotor = sorted({category for category in self.data[ColNames.CategorySensorimotor]})
        self.response_labels              = sorted({response for response in self.data[ColNames.Response]})
        self.response_labels_sensorimotor = sorted({response for response in self.data[ColNames.ResponseSensorimotor]})

        # Build vocab lists

        # All multi-word tokens in the dataset
        # TODO: this should be an unsorted set
        self.vocabulary_multi_word  = sorted(set(self.category_labels)
                                             | set(self.response_labels))
        # All single-word tokens in the dataset
        # TODO: this should be an unsorted set
        self.vocabulary_single_word = sorted(set(word
                                                 for vocab_item in self.vocabulary_multi_word
                                                 for word in word_tokenise(vocab_item)
                                                 if word not in CategoryProduction._ignored_words))

    def responses_for_category(self,
                               category: str,
                               single_word_only: bool = False,
                               sort_by: 'ColNames' = None,
                               use_sensorimotor_responses: bool = False) -> List[str]:
        """
        Responses for a provided category.
        :param category:
        :param single_word_only:
            Give only single-word responses
        :param sort_by: ColNames
            Default: ColNames.MeanRank.
        :param use_sensorimotor_responses:
            Give the sensorimotor-norms version of the response.
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
        if use_sensorimotor_responses:
            filtered_data = filtered_data[ColNames.ResponseSensorimotor]
        else:
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
