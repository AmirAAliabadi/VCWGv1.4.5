# -*- coding: utf-8 -*-
from __future__ import print_function
import sys

"""
Progress Bar
Developed by Saeran Vasanthakumar
Last update: March 2018
"""

"""
Call in a loop to create terminal progress bar
    iteration   - Required  : current iteration (int)
    total       - Required  : total iterations (int)
    prefix      - Optional  : prefix string (str)
    suffix      - Optional  : suffix string (str)
    bar_length  - Optional  : character length of bar (int)
"""

def print_progress(iteration, total, prefix='', suffix='', bar_length=50):

    str_format = "{0:." + str(1) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '|' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write("\r{} |{}| {}{} {}".format(prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')

    sys.stdout.flush()
