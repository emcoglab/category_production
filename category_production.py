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
from logging import getLogger
from os import path, remove
from typing import List, Set

from pandas import DataFrame, read_csv

from .utils import unique
from .exceptions import CategoryNotFoundError, ResponseNotFoundError
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
    # precomputed ppmi
    LinguisticPPMI       = "Linguistic.PPMI"

    # Extra predictor columns for RT data
    NSyll                = "NSyll"
    PLD                  = "PLD"
    CategoryRT           = "Category.RT.secs"

    # Computed columns for RT data

    # Mean reaction time for first responses
    MeanRT               = "Mean.RT"
    # Mean standardised reaction time for first responses
    MeanZRT              = "Mean.zRT"

    # Participant data columns

    Participant          = "Participant"


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

    # individual z-rts outside +/- this limit won't count towards the mean
    zrt_outliers_limit = 3

    @classmethod
    def _default_word_tokenise(cls, x):
        """Default tokeniser to use if none is provided."""
        return x.split(" ")

    def __init__(self,
                 minimum_production_frequency: int = 2,
                 word_tokenise: callable = None,
                 use_cache: bool = False):
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
        :param use_cache:
            (Optional.)
            Use a cached version of the data if available.  Use with caution in case the format of the underlying data
            has changed after an update.
            Default: False.
        """

        if not self._saved_cache_version_valid():
            logger.warning("Cache version invalid")
            self._clear_saved_cache()

        # Validate arguments
        if minimum_production_frequency < 1:
            raise ValueError("minimum_production_frequency must be at least 1")

        # If no tokeniser given, use default
        if word_tokenise is None:
            word_tokenise = CategoryProduction._default_word_tokenise

        # Data for individual participant responses we can always load from source as there are no computed columns
        self.participant_data: DataFrame = CategoryProduction._load_participant_data_from_source()
        self._process_participant_data()

        self.participants: List[int] = list(self.participant_data[ColNames.Participant].unique())

        # Data collapsed over category/response pairs has additional computed columns, which we may want to cache
        if (not use_cache) or (not CategoryProduction._could_load_cache()):
            # If we can't use the cache, we produce it as normal
            self.data: DataFrame = self.participant_data.copy()
            self._process_collapsed_data()
            self._save_cache()
        else:
            # If we can load from cache, we do
            self.data = CategoryProduction._load_from_cache()

        # Delete rows with minimum production frequency
        self.data = self.data[self.data[ColNames.ProductionFrequency] >= minimum_production_frequency]
        self.participant_data = self.participant_data[self.participant_data[ColNames.ProductionFrequency] >= minimum_production_frequency]

        # Build label lists
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
    def _load_participant_data_from_source(cls) -> DataFrame:
        participant_data: DataFrame = read_csv(Preferences.master_main_data_csv_path, index_col=0, header=0)
        return participant_data

    def _process_participant_data(self):
        """Mutates self.participant_data"""
        # A nan in the FRF column means the first-rank frequency is zero
        # Set FRF=NAN rows to FRF=0 and convert to int
        self.participant_data[ColNames.FirstRankFrequency] = self.participant_data[ColNames.FirstRankFrequency].fillna(0).astype(int)

        # Trim whitespace and convert all words to lower case
        self.participant_data[ColNames.Category] = self.participant_data[ColNames.Category].str.strip()
        self.participant_data[ColNames.Category] = self.participant_data[ColNames.Category].str.lower()
        self.participant_data[ColNames.Response] = self.participant_data[ColNames.Response].str.strip()
        self.participant_data[ColNames.Response] = self.participant_data[ColNames.Response].str.lower()

        # Sometimes participants give duplicate responses, and we only want the first one for each participant
        # First sort the participant data so we can drop the correct duplicates.
        # NOTE: we don't adjust any other ranks (so there will be non-sequential ranks where duplicates were dropped).
        self.participant_data.sort_values(by=[ColNames.Category, ColNames.Response, ColNames.Participant, 'Rank'],
                                          inplace=True)
        self.participant_data.drop_duplicates(subset=[ColNames.Category, ColNames.Response, ColNames.Participant],
                                              keep='first', inplace=True)

    def _process_collapsed_data(self):
        """Mutates self.data"""
        # Collapse data over category–response pairs
        self.data.drop_duplicates(
            subset=[ColNames.Category, ColNames.Response],
            inplace=True)
        self.data.drop(['Item', 'Participant', 'Trial.no.', 'Rank'], axis=1, inplace=True)

        # Get data from RT file

        rt_data: DataFrame = read_csv(Preferences.master_rt_data_csv_path, index_col=0, header=0)

        # Add RT and zRT columns
        self.data = self.data.merge(
            rt_data[[ColNames.Category, ColNames.Response, "RT"]]
                .groupby([ColNames.Category, ColNames.Response])
                .mean()
                .reset_index(), how="left"
        ).rename(columns={"RT": ColNames.MeanRT})
        # For mean zRT we want to filter outliers
        central_rt_data = rt_data[
            (rt_data["zscore_per_pt"] <= CategoryProduction.zrt_outliers_limit)
            & (rt_data["zscore_per_pt"] >= -CategoryProduction.zrt_outliers_limit)]
        self.data = self.data.merge(
            central_rt_data[[ColNames.Category, ColNames.Response, "zscore_per_pt"]]
                .groupby([ColNames.Category, ColNames.Response])
                .mean()
                .reset_index(), how="left"
        ).rename(columns={"zscore_per_pt": ColNames.MeanZRT})

        # Add NSyll, PLD and CategoryRT columns
        self.data = self.data.merge(
            rt_data[[ColNames.Category, ColNames.Response, ColNames.NSyll]]
                .groupby([ColNames.Category, ColNames.Response])
                .first()
                .reset_index(), how="left")
        self.data = self.data.merge(
            rt_data[[ColNames.Category, ColNames.Response, ColNames.PLD]]
                .groupby([ColNames.Category, ColNames.Response])
                .first()
                .reset_index(), how="left")
        self.data = self.data.merge(
            rt_data[[ColNames.Category, ColNames.Response, ColNames.CategoryRT]]
                .groupby([ColNames.Category, ColNames.Response])
                .first()
                .reset_index(), how="left")

        # Add participant hitrate data
        for participant in self.participants:
            logger.info(f"Adding participant {participant} hit info")
            this_ppt_data = self.participant_data[self.participant_data[ColNames.Participant] == participant].copy()
            this_ppt_data[f"Participant {participant} saw category"] = True
            this_ppt_data[f"Participant {participant} response hit"] = True
            # add saw-category column
            self.data = self.data.merge(this_ppt_data[[ColNames.Category, f"Participant {participant} saw category"]].groupby(ColNames.Category).first(), how="left", on=ColNames.Category)
            self.data[f"Participant {participant} saw category"] = self.data[f"Participant {participant} saw category"].fillna(False).astype(bool)
            # add response-hit column
            self.data = self.data.merge(this_ppt_data[[ColNames.Category, ColNames.Response, f"Participant {participant} response hit"]], how="left", on=[ColNames.Category, ColNames.Response])
            self.data[f"Participant {participant} response hit"] = self.data[f"Participant {participant} response hit"].fillna(False).astype(bool)
        self.data.reset_index(drop=True, inplace=True)

    # region Cache

    def _save_cache(self):
        """Save current master data file."""
        logger.info(f"Saving cached data file to {Preferences.cached_data_csv_path}")
        with open(Preferences.cached_data_csv_path, mode="w", encoding="utf-8") as cache_file:
            self.data.to_csv(cache_file, header=True, index=False)
        with open(Preferences.cache_version_path, mode="w", encoding="utf-8") as cache_version:
            cache_version.write(self._cache_version)

    @property
    def _cache_version(self) -> str:
        """The version of the current cache."""
        try:
            from git import Repo
            git_hash = Repo(search_parent_directories=True).head.object.hexsha
        except ModuleNotFoundError:
            try:
                import subprocess
                git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
            except OSError:
                git_hash = "Unknown"
        return git_hash

    @classmethod
    def _load_from_cache(cls) -> DataFrame:
        """Load cached master data file."""
        logger.warning(f"Using cached data file from {Preferences.cached_data_csv_path}")
        with open(Preferences.cached_data_csv_path, mode="r", encoding="utf-8") as cached_file:
            return read_csv(cached_file, header=0, index_col=False)

    @classmethod
    def _could_load_cache(cls) -> bool:
        """If the cached file could be loaded."""
        return path.isfile(Preferences.cached_data_csv_path)

    def _saved_cache_version_valid(self) -> bool:
        try:
            with open(Preferences.cache_version_path, mode="r", encoding="utf-8") as cache_file:
                cache_version = cache_file.read()
            return cache_version == self._cache_version
        except FileNotFoundError:
            return False

    def _clear_saved_cache(self):
        """Clear the current cached data file."""
        if self._could_load_cache():
            logger.warning(f"Clearing cached data [{Preferences.cached_data_csv_path}]")
            remove(Preferences.cached_data_csv_path)
            remove(Preferences.cache_version_path)

    # endregion

    def responses_for_category(self,
                               category: str,
                               single_word_only: bool = False,
                               sort_by: 'ColNames' = None,
                               use_sensorimotor: bool = False,
                               force_unique: bool = False,
                               ) -> List[str]:
        """
        Responses for a provided category.
        :param category:
        :param single_word_only:
            Give only single-word responses
        :param sort_by: ColNames
            Default: ColNames.MeanRank.
        :param use_sensorimotor:
            Give the sensorimotor-norms version of the response to the sensorimotor-norms version of the category.
        :param force_unique:
            If set to true, gives UNIQUE responses only.
            Only applies when `use_sensorimotor` is True:
            When returning sensorimotor equivalents, sometimes there will be collisions.
                E.g.
                winter sports -> downhill skiing [skiing]
                winter sports -> skiing [skiing]
            In this case if `force_unique` is False, there will be 2 skiing entries (to match with each of the unique
            non-sensorimotor versions), and if True, there will be one (just "skiing").
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
            responses = [r for r in filtered_data if " " not in r]
        else:
            responses = [r for r in filtered_data]

        if use_sensorimotor and force_unique:
            responses = unique(responses)

        return responses

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


# For debug
if __name__ == '__main__':
    cp = CategoryProduction()
    pass
