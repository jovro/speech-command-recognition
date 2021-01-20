from os import makedirs
import logging


def create_directories(dirs: list):
    for directory in dirs:
        makedirs(directory, exist_ok=True)
    logging.getLogger("Filesystem utils").info("Checked that directories exist: {}".format(", ".join(dirs)))
