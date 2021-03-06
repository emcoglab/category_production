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
    _experiment_dir = path.join(_box_root, "Experiments/Phase 1 - Categorisation/Experiment 1.4 - Category production/Data & Analysis/Errors and reanalysis March 2021/")

    master_main_data_csv_path = path.join(_experiment_dir, "1.4_FULL data & variables_FINAL_Mar 2021.csv")
    master_rt_data_csv_path = path.join(_experiment_dir, "1.4_RT Data FULL_FINAL_Mar 2021.csv")
