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

from typing import List

from pandas import DataFrame

from .category_production_preferences import Preferences


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

        self.data = DataFrame.from_csv(Preferences.linguistic_wordlist_csv_path, index_col=0, header=0)

        # Only consider unique category–response pairs
        self.data.drop_duplicates(
            subset=[CategoryProduction.ColNames.Category, CategoryProduction.ColNames.Response],
            inplace=True)

        # Hide those with production frequency 1
        self.data = self.data[self.data[CategoryProduction.ColNames.ProductionFrequency] != 1]

        # Trim whitespace and convert all words to lower case
        self.data[CategoryProduction.ColNames.Category] = self.data[CategoryProduction.ColNames.Category].str.strip()
        self.data[CategoryProduction.ColNames.Category] = self.data[CategoryProduction.ColNames.Category].str.lower()
        self.data[CategoryProduction.ColNames.Response] = self.data[CategoryProduction.ColNames.Response].str.strip()
        self.data[CategoryProduction.ColNames.Response] = self.data[CategoryProduction.ColNames.Response].str.lower()

        # Apply specific substitutions.
        self.data.replace(Preferences.specific_substitutions, inplace=True)

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
