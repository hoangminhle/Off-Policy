# storage.py: functions to read/write to azure storage (using XT)
import os
from rl_nexus.utils import utils
from xtlib.run import Run

def upload_files(share_name, share_path, local_path, username=None, overwrite=True, show_progress=True):
    '''
    uploads files specified by local_path to the datastore path formed by:
        share_name/username/share_path.

        :param share_name: the XT share name where files will be stored (usually one of: data, models, trajectories)
        :param share_path: the path where file(s) will be stored on the share (e.g., "maze" or "procgen")
        :param local_path: the path to the local files to be uploaded
        :param username: the username associated with the data on the share (if None, will use OS username)
        :param overwrite: if False, existing files will not be overwritten (not yet supported)
        :param show_progress: if True, progress messages will be printed 
    '''
    if username is None:
        username = utils.get_username()
    
    share_path = os.path.join(username, share_path)
    share_path = share_path.replace("\\", "/")

    # use XT to prevent interactive authenication (which will fail for remote runs)
    xt_run = Run() 
    results = xt_run.upload_files_to_share(share_name, share_path, local_path, show_feedback=show_progress)
    return results

def download_files(share_name, share_path, local_path, username=None, overwrite=True, show_progress=True):
    '''
    downloads files from the datastore path formed by:
        share_name/username/share_path.

        :param share_name: the XT share name where files are stored (usually one of: data, models, trajectories)
        :param share_path: the path where files are stored on the share (e.g., "maze" or "procgen")
        :param local_path: the directory or fn where the file(s) will be downloaded
        :param username: the username associated with the data on the share (if None, will use OS username)
        :param overwrite: if False, existing files will not be overwritten (not yet supported)
        :param show_progress: if True, progress messages will be printed 
    '''
    if username is None:
        username = utils.get_username()
    
    share_path = os.path.join(username, share_path)
    share_path = share_path.replace("\\", "/")

    # use XT to prevent interactive authenication (which will fail for remote runs)
    xt_run = Run()
    results = xt_run.download_files_from_share(share_name, share_path, local_path, show_feedback=show_progress)
    return results
