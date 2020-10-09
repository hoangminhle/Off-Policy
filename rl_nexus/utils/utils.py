# utils.py: shared misc. functions
import os

def is_windows():
    return os.name == "nt"

def get_username():
    ev_user = "username" if is_windows() else "user"
    username = os.getenv(ev_user, "")
    if not username:
        # some backends don't set username, so get it from xt info
        username = os.getenv("XT_USERNAME", "")
    return username

def ensure_dir_exists(dir=None, file=None):
    if file:
        dir = os.path.dirname(file)

    if dir and not os.path.exists(dir):
        os.makedirs(dir)

def resolve_path(path_in):
    if path_in is None:
        return None
    else:
        return os.path.abspath(path_in)

