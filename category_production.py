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

from category_production_preferences import Preferences


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

        # Trim whitespace and convert all words to lower case
        self.data[CategoryProduction.ColNames.Category] = self.data[CategoryProduction.ColNames.Category].str.strip()
        self.data[CategoryProduction.ColNames.Category] = self.data[CategoryProduction.ColNames.Category].str.lower()
        self.data[CategoryProduction.ColNames.Response] = self.data[CategoryProduction.ColNames.Response].str.strip()
        self.data[CategoryProduction.ColNames.Response] = self.data[CategoryProduction.ColNames.Response].str.lower()

        # Build vocab lists

        self.single_word_vocabulary = set()
        self.multi_word_vocabulary = set()

        for row_i, data_row in self.data.iterrows():
            category = data_row[CategoryProduction.ColNames.Category]
            response = data_row[CategoryProduction.ColNames.Response]

            self.multi_word_vocabulary.add(category)
            self.multi_word_vocabulary.add(response)

            # Use the same tokenisation strategy as the corpus, but ignore some function words
            category_words = [word for word in word_tokenise(category) if word not in CategoryProduction._ignored_words]
            response_words = [word for word in word_tokenise(response) if word not in CategoryProduction._ignored_words]

            self.single_word_vocabulary |= set(category_words)
            self.single_word_vocabulary |= set(response_words)

        # Convert to alphabetical lists
        self.single_word_vocabulary = sorted(self.single_word_vocabulary)
        self.multi_word_vocabulary = sorted(self.multi_word_vocabulary)

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


# For debug
if __name__ == '__main__':
    cp = CategoryProduction()
    pass
