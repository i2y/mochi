"""Helpers for working with paths an similar.
"""

import os


class TempDir(object):
    """Switch to a temporary directory and back.

    This is a context manager.

    usage:

        # in orig_dir
        with TempDir('/path/to/temp_dir'):
            # in temp_dir
            do_something_in_temp_dir()
        # back in orig_dir


    """
    # pylint: disable=too-few-public-methods

    def __init__(self, temp_dir):
        self.temp_dir = temp_dir

    def __enter__(self):
        # pylint: disable=attribute-defined-outside-init
        self.orig_dir = os.getcwd()
        os.chdir(self.temp_dir)

    def __exit__(self, *args):
        os.chdir(self.orig_dir)
