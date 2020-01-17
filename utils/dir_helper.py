import os


def check_and_make_path(to_make):
    if not os.path.exists(to_make):
        os.makedirs(to_make)
