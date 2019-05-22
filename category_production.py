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
from typing import List, Set

from numpy import mean, nan
from pandas import DataFrame, read_csv

from .category_production_preferences import Preferences

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

    # Extra predictor columns

    LogWordFreq          = "LgSUBTLWF"
    Typicality           = "typicality.rating"

    # Computed columns for RT data

    # Mean reaction time for first responses
    MeanRT               = "Mean.RT"
    # Mean standardised reaction time for first responses
    MeanZRT              = "Mean.zRT"


class CategoryProduction(object):
    """
    Represents the dataset for Experiment 1.4 - Category Production.
    In particular, the BrEng version to be used for the linguistic model.
    """

    ignored_words = {
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

        if not CategoryProduction._could_load_cache():
            self.data: DataFrame = CategoryProduction._load_from_source(minimum_production_frequency)
            self._save_cache()
        else:
            CategoryProduction._load_from_cache()

        # Build lists

        self.category_labels: List[str]              = sorted({category for category in self.data[ColNames.Category]})
        self.category_labels_sensorimotor: List[str] = sorted({category for category in self.data[ColNames.CategorySensorimotor]})
        self.response_labels: List[str]              = sorted({response for response in self.data[ColNames.Response]})
        self.response_labels_sensorimotor: List[str] = sorted({response for response in self.data[ColNames.ResponseSensorimotor]})

        # Build vocab lists

        # All multi-word tokens in the dataset
        self.vocabulary_multi_word: Set[str]  = set(set(self.category_labels) | set(self.response_labels))
        # All single-word tokens in the dataset
        self.vocabulary_single_word: Set[str] = set(word
                                                    for vocab_item in self.vocabulary_multi_word
                                                    for word in word_tokenise(vocab_item)
                                                    if word not in CategoryProduction.ignored_words)

    @classmethod
    def _load_from_source(cls, minimum_production_frequency) -> DataFrame:
        """Load and rebuild file from source."""
        data: DataFrame = read_csv(Preferences.master_main_data_csv_path, index_col=0, header=0)
        # Only consider unique category–response pairs
        data.drop_duplicates(
            subset=[ColNames.Category, ColNames.Response],
            inplace=True)
        # Drop columns which disambiguated duplicate entries
        data.drop(['Item', 'Participant', 'Trial.no.', 'Rank'], axis=1, inplace=True)
        # Hide those with minimum production frequency
        data = data[data[ColNames.ProductionFrequency] >= minimum_production_frequency]
        # A nan in the FRF column means the first-rank frequency is zero
        # Set FRF=NAN rows to FRF=0 and convert to int
        data[ColNames.FirstRankFrequency] = data[ColNames.FirstRankFrequency].fillna(0).astype(int)
        # Trim whitespace and convert all words to lower case
        data[ColNames.Category] = data[ColNames.Category].str.strip()
        data[ColNames.Category] = data[ColNames.Category].str.lower()
        data[ColNames.Response] = data[ColNames.Response].str.strip()
        data[ColNames.Response] = data[ColNames.Response].str.lower()
        # Add RT and zRT columns
        rt_data: DataFrame = read_csv(Preferences.master_rt_data_csv_path, index_col=0, header=0)
        data[ColNames.MeanRT] = data.apply(partial(_get_mean_rt, rt_data=rt_data, use_zrt=False), axis=1)
        data[ColNames.MeanZRT] = data.apply(partial(_get_mean_rt, rt_data=rt_data, use_zrt=True), axis=1)
        data.reset_index(drop=True, inplace=True)
        return data

    def _save_cache(self):
        """Save current master data file."""
        logger.info(f"Saving cached data file to {Preferences.cached_data_csv_path}")
        with open(Preferences.cached_data_csv_path, mode="w", encoding="utf-8") as cache_file:
            self.data.to_csv(cache_file, header=True, index=False)

    @classmethod
    def _load_from_cache(cls) -> DataFrame:
        """Load cached master data file."""
        logger.info(f"Loading cached data file from {Preferences.cached_data_csv_path}")
        with open(Preferences.cached_data_csv_path, mode="r", encoding="utf-8") as cached_file:
            return DataFrame.from_csv(cached_file, header=0, index_col=None)

    @classmethod
    def _could_load_cache(cls) -> bool:
        """If the cached file could be loaded."""
        return path.isfile(Preferences.cached_data_csv_path)

    def responses_for_category(self,
                               category: str,
                               single_word_only: bool = False,
                               sort_by: 'ColNames' = None,
                               use_sensorimotor: bool = False) -> List[str]:
        """
        Responses for a provided category.
        :param category:
        :param single_word_only:
            Give only single-word responses
        :param sort_by: ColNames
            Default: ColNames.MeanRank.
        :param use_sensorimotor:
            Give the sensorimotor-norms version of the response to the sensorimotor-norms version of the category.
        :return:
            List of responses.
        :raises CategoryNotFoundError: When requested category is not found in the norms
        """
        # Set default values
        if sort_by is None:
            sort_by = ColNames.MeanRank

        # Check validity
        if use_sensorimotor:
            if category not in self.category_labels_sensorimotor:
                raise CategoryNotFoundError(category)
        else:
            if category not in self.category_labels:
                raise CategoryNotFoundError(category)

        # Filter data
        filtered_data: DataFrame = self.data[self.data[
            ColNames.CategorySensorimotor if use_sensorimotor else ColNames.Category
        ] == category]
        filtered_data = filtered_data.sort_values(by=sort_by, ascending=True)
        filtered_data = filtered_data[
            ColNames.ResponseSensorimotor if use_sensorimotor else ColNames.Response
        ]

        if single_word_only:
            return [r for r in filtered_data if " " not in r]
        else:
            return [r for r in filtered_data]

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
