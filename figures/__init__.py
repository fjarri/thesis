import os.path


def get_path(script_name, fname):
    return os.path.join(
        os.path.dirname(os.path.abspath(script_name)),
        fname)
