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
    _box_root = "/storage/hpc/57/wingfiel/"
    _experiment_dir = path.join(_box_root, "experimental_data")

    # Specific paths
    main_data_csv_path = path.join(_experiment_dir, "1.4_FULL data & variables_May 2019.csv")
    rt_data_csv_path = path.join(_experiment_dir, "1.4_RT data_ALL.csv")
