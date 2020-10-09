# Tutorial 4: Downloading files

We recommend organizing downloaded files in separate job directories under a **dilbert/jobs** root, such as **dilbert/jobs/job#**.
The **dilbert/jobs** directory tree is ignored by git.

To direct the output of a **list runs** command to a .csv file, use the **--export** option:

    xt list runs job265 --export=../jobs/job265_list_runs.csv

XT can download **individual job files** or entire directory trees to a specified locations.
For instance, to download and view the (potentially live) console output from a single run:

	xt download mirrored/console.txt ../jobs/job265/run527.1/console.txt --run=run527.1
	notepad ../jobs/job265/run527.1/console.txt
	
To download and view the **original runspec** that launched the job:

	xt download after/rl_nexus/launched_job_spec.yaml ../jobs/job265/run527.1/launched_job_spec.yaml --run=run527.1
	notepad ../jobs/job265/run527.1/launched_job_spec.yaml
	
A run's repro_spec can also be downloaded this way:
 
	xt download mirrored/expanded_spec.yaml ../jobs/job265/run527.1/repro_spec.yaml --run=run527.1
	notepad ../jobs/job265/run527.1/repro_spec.yaml

Alternatively, a single command can download the directory tree for an **entire run**.

	xt download mirrored ../jobs/job265/run527.1 --run=run527.1

This downloads both console.txt and repro_spec.yaml, along with any other files logged by the run,
such as models, trajectories, and tensorboard logs.

It's also possible to download a zipped copy of the job's **rl_nexus** source code. 

	xt download before/code ../jobs/job265 --job=job265

If required, **all job files** can be downloaded together. This can take some time:

    xt extract job265 ../jobs/job265

## Replicating runs and jobs

* To replicate a specific run:
    * Download that run's **repro_spec.yaml** as described above.
    * Execute it like any runspec.
    * Each instance of randint is replaced by the actual random seed used in that run.
    
* To replicate an entire job:
    * If the original runspec is available, simply execute it again. Otherwise,
    * Download and unzip that job's rl_nexus source code as described above.
    * Execute **rl_nexus/launched_job_spec.yaml** like any runspec.

## [Next tutorial:  HP search](5__HP_search.md)
