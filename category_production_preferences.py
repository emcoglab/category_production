"""
===========================
Preferences and paths for this analysis.
===========================

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

from os import path


class Preferences(object):

    # Root dirs
    _box_root = "/Users/caiwingfield/Box Sync/LANGBOOT Project/"
    _experiment_dir = path.join(_box_root, "Experiments/Phase 1 - Categorisation/Experiment 1.4 - Category production/")

    # Specific paths
    linguistic_wordlist_csv_path = path.join(_experiment_dir, "Data & Analysis/1.4_ALL Data & variables.csv")
    linguistic_wordlist_rt_csv_path = path.join(_experiment_dir, "Data & Analysis/1.4_RT data_ALL.csv")

    # Substitutions

    specific_substitutions = {
        "bobsledging": "bobsledding",
        "dodgeball":   "dodge-ball",
        "fifty-seven": "57",
        "gokart":      "go-kart",
        "jetskiing":   "jet-skiing",
        "micrometre":  "micrometer",
        "slalem":      "slalom",
        "stepparents": "step-parents",
        "yogurt":      "yoghurt",
    }
