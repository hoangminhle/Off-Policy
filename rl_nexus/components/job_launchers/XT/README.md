# XT Component

## Setting up

Before launching XT jobs from your local machine, first follow the [Dilbert installation instructions](../../../../).
In particular, your **rl_nexus** directory must contain a copy of **rl_nexus/useful_run_specs/start/xt_config.yaml**, 
which you can modify locally if needed.

To make sure your installation is up-to-date, perform the following steps:
* Update your local dilbert repo.

        git pull

* Activate your virtual environment (depending on how it was created and what it was named).

        conda activate dilbert

* Check the XT version.

        xt --version

* If that XT version does not match the one listed in [dilbert/requirements.txt](../../../../requirements.txt), 
reinstall the dependencies to get the correct version of XT.

        cd dilbert
        pip install -r requirements.txt

* After updating to a new XT version, copy the current **useful_run_specs/start/xt_config.yaml**
into the **rl_nexus** directory. 
Or, if you have modified your personal copy in rl_nexus, merge any changes into that copy.

* All XT commands (except version) must be executed in the **rl_nexus** directory, where your copy of **xt_config.yaml** resides. 

## Tutorials

Documentation for using XT is divided into several tutorials.
Each one builds on topics covered in the earlier tutorials.

1. [Launching jobs](1__Launching_jobs.md)
1. [Monitoring jobs](2__Monitoring_jobs.md)
1. [Plotting job results](3__Plotting_job_results.md)
1. [Downloading files](4__Downloading_files.md)
1. [HP search](5__HP_search.md)
