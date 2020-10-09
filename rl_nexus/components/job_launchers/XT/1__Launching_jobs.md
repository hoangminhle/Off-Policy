# Tutorial 1: Launching jobs

## Preparing a runspec

For a runspec to be executed as a remote job, it needs to contain an enabled XT component section at the top.
As an example, here is the top of [useful_run_specs/xt_walk_through.yaml](../../../useful_run_specs/xt_walk_through.yaml):

    component: root/Root_Component
    log_to_tensorboard: true
    experiment_path: ../results
    job_launcher:
      component: job_launchers/XT
      enabled: 1
      hp_tuning: false
      total_runs: 20
      compute_nodes: 5
      runs_per_node: 4
      compute_target: azb-cpu
      low_priority: true
      hold: false

Each job consists of some number of runs, where each run executes the full sequence of processing stages in the runspec.
The XT section above is configured to execute the runspec 20 times, 
in 20 parallel runs spread over 5 compute nodes (VMs) in Azure Batch,
with 4 instances running at a time on each node.
If **total_runs** were raised to 40, then each finished run would be replaced by a new run until reaching 40.

See [rl_nexus/components/job_launchers/XT/XT.yaml](../../../components/job_launchers/XT/XT.yaml) for details on 
other XT component properties.  

Each run's entire **../results** directory tree gets copied (mirrored) live to Azure Storage.
But results will not be mirrored if **experiment_path** is set to some location outside of **../results**.


## Launching a job

To launch an XT job, enable the XT component then execute the runspec. 
The only hard part is making sure your runspec matches your intentions.
The following steps help avoid simple errors that waste time and compute.

#### Pre-flight checklist

* Look over your entire runspec, making the following changes where necessary:
    * To prevent your runs from all doing the same deterministic thing, 
use **randint** as the value of appropriate random seeds, such as the environment's **agent_placement_seed**, or SSL_Proc's **random_seed**.
    * Make sure the **max_iterations** property in each processing stage is reasonable for your experiment.
    * To produce smooth curves, and to reduce database hits and download sizes, 
consider setting **iters_per_report** to between 5% and 10% of **max_iterations**. 
(See the example used below: **xt_walk_through.yaml**)
    * Enable the XT component.
* Perform a local run as a sanity check.
    * Make a temporary copy of your runspec.
    * Modify the copy:
        * Disable the XT component.
        * Set **max_iterations** in each stage to something small, so the run will finish quickly.
    * Execute the runspec copy as usual. This will be a local run.
    * Review repro_spec.yaml checking for anything unexpected. The XT section will not appear in repro_spec.yaml since it was disabled.
    * Delete the runspec copy.
* Launch the job by executing the runspec as usual.
    * **python run.py \<path to runspec\>**

You may be prompted to authenticate your credentials for Azure access.
If all goes well, job launch messages will appear:

    (dilbert) D:\dilbert\rl_nexus>python run.py useful_run_specs\xt_walk_through.yaml
    ------- Runspec (useful_run_specs\xt_walk_through.yaml) copied to launched_job_spec.yaml
    job265, target: azb-cpu, creating parent runs: 5/5, uploading 212 code files (zipped), ws1/run527, ws1/run528, ws1/run529, ws1/run530, ws1/run531, job submitted
    ------- Job launched. Use XT commands to access results.

## After job launch

Record the job#, such as **job265** in the example above. This is the unique string that you or anyone else can use to access the job's results 
while the job is running or long after the job has finished.
It's good practice to keep a personal list of job numbers in a permanent place, like OneNote,
along with a brief description of each job. 

After launching a job, it is common to make some small modification to your runspec then launch another.
This allows many variations of a job to be launched in quick succession.
But once your compute target's quota is reached, new jobs will be queued until other jobs finish.

## [Next tutorial:  Monitoring jobs](2__Monitoring_jobs.md)
