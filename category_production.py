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
from pandas import DataFrame

from category_production_preferences import Preferences


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

    def __init__(self,
                 tokenize: callable = None):
        """
        :param tokenize:
        (Optional.)
        If provided and not None: A function which maps strings (strings) to lists of strings (token substrings).
        Default: s ↦ s.split(" ")
        """

        if tokenize is None:
            tokenize = lambda s: s.split(" ")

        # Load and prepare data

        self.data = DataFrame.from_csv(Preferences.linguistic_wordlist_csv_path, index_col=0, header=0)
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
            category_words = [word for word in tokenize(category) if word not in CategoryProduction.ignored_words]
            response_words = [word for word in tokenize(response) if word not in CategoryProduction.ignored_words]

            self.single_word_vocabulary |= set(category_words)
            self.single_word_vocabulary |= set(response_words)

        # Convert to alphabetical lists
        self.single_word_vocabulary = sorted(self.single_word_vocabulary)
        self.multi_word_vocabulary = sorted(self.multi_word_vocabulary)

    class ColNames(object):
        """Column names used in the data files."""
        Category = "Category"
        Response = "Response"
