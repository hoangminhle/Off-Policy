# Tutorial 2: Monitoring jobs

## Monitoring jobs from the monitor window

Immediately after launching a job, XT opens a monitor window showing the job's progress.
You can shift the monitor's focus to different compute nodes, terminate the job, 
or close the window without terminating the job.
To reopen a monitor window:

    xt --echo monitor job265

## Monitoring jobs from the command line

From the **rl_nexus** directory, type **xt list runs job#** to get a tabular summary of job progress and results. 
Right after launching a job, the summary should look something like this:

    (dilbert) D:\dilbert\rl_nexus>xt list runs job265
    
    workspace  job     target   run     status
    
    ws1        job265  azb-cpu  run527  queued
    ws1        job265  azb-cpu  run528  queued
    ws1        job265  azb-cpu  run529  queued
    ws1        job265  azb-cpu  run530  queued
    ws1        job265  azb-cpu  run531  queued
    
    total runs listed: 5

Each of the 5 lines corresponds to one compute node being provisioned.
You can watch the details of the provisioning through Azure Batch Explorer, described further down.

Executing the same XT command a bit later will show that some of the runs have started:

    (dilbert) D:\dilbert\rl_nexus>xt list runs job265
    
    workspace  job     target   run       status    steps  s1-hrs  s1-rew  s1-win
    
    ws1        job265  azb-cpu  run527    queued
    ws1        job265  azb-cpu  run528    spawning
    ws1        job265  azb-cpu  run528.1  running    5000   0.020   0.001   0.081
    ws1        job265  azb-cpu  run528.2  running    5000   0.021   0.003   0.141
    ws1        job265  azb-cpu  run528.3  running    5000   0.020   0.000   0.024
    ws1        job265  azb-cpu  run528.4  running    5000   0.021   0.002   0.112
    ws1        job265  azb-cpu  run529    queued
    ws1        job265  azb-cpu  run530    spawning
    ws1        job265  azb-cpu  run530.1  running
    ws1        job265  azb-cpu  run530.2  running
    ws1        job265  azb-cpu  run530.3  running
    ws1        job265  azb-cpu  run530.4  running
    ws1        job265  azb-cpu  run531    spawning
    ws1        job265  azb-cpu  run531.1  running    5000   0.020   0.002   0.132
    ws1        job265  azb-cpu  run531.2  running    5000   0.020   0.002   0.135
    ws1        job265  azb-cpu  run531.3  running    5000   0.020   0.000   0.012
    ws1        job265  azb-cpu  run531.4  running    5000   0.020   0.003   0.163
    
    workspace  job     target   run       status    steps  s1-hrs  s1-rew  s1-win
    
    total runs listed: 17

The run names containing decimal points are called **child runs**.
Each child run is executing the entire runspec, one stage at a time. 
The run names without decimal points are called **parent runs**.
Each one represents the controller process that manages one compute node.
The table shows the most recently logged metrics for each run,
where the **s1-** prefix refers to processing stage 1.

A few minutes later, all runs should have started. Here we see 4 runs in progress on each of the 5 nodes.

    (dilbert) D:\dilbert\rl_nexus>xt list runs job265
    
    workspace  job     target   run       status    iters  s1-hrs  s1-rew  s1-win
    
    ws1        job265  azb-cpu  run527    spawning
    ws1        job265  azb-cpu  run527.1  running   15000   0.060  0.0018   0.103
    ws1        job265  azb-cpu  run527.2  running   15000   0.060  0.0106   0.465
    ws1        job265  azb-cpu  run527.3  running   15000   0.060  0.0100   0.424
    ws1        job265  azb-cpu  run527.4  running   15000   0.060  0.0118   0.484
    ws1        job265  azb-cpu  run528    spawning
    ws1        job265  azb-cpu  run528.1  running   15000   0.060  0.0110   0.474
    ws1        job265  azb-cpu  run528.2  running   15000   0.060  0.0074   0.352
    ws1        job265  azb-cpu  run528.3  running   15000   0.059  0.0104   0.441
    ws1        job265  azb-cpu  run528.4  running   15000   0.060  0.0058   0.305
    ws1        job265  azb-cpu  run529    spawning
    ws1        job265  azb-cpu  run529.1  running   15000   0.060  0.0100   0.431
    ws1        job265  azb-cpu  run529.2  running   15000   0.060  0.0068   0.343
    ws1        job265  azb-cpu  run529.3  running   15000   0.060  0.0028   0.154
    ws1        job265  azb-cpu  run529.4  running   15000   0.060  0.0046   0.242
    ws1        job265  azb-cpu  run530    spawning
    ws1        job265  azb-cpu  run530.1  running   15000   0.060  0.0094   0.439
    ws1        job265  azb-cpu  run530.2  running   15000   0.059  0.0088   0.407
    ws1        job265  azb-cpu  run530.3  running   15000   0.059  0.0074   0.366
    ws1        job265  azb-cpu  run530.4  running   15000   0.059  0.0064   0.305
    ws1        job265  azb-cpu  run531    spawning
    ws1        job265  azb-cpu  run531.1  running   15000   0.060  0.0040   0.215
    ws1        job265  azb-cpu  run531.2  running   15000   0.060  0.0106   0.457
    ws1        job265  azb-cpu  run531.3  running   15000   0.059  0.0060   0.309
    ws1        job265  azb-cpu  run531.4  running   15000   0.059  0.0116   0.472
    
    workspace  job     target   run       status    iters  s1-hrs  s1-rew  s1-win
    
    total runs listed: 25

As new processing stages are executed, additional columns appear to the right.
The **steps** column always refers to the currently running stage for the given run.

Since **job265** was launched from **useful_run_specs\xt_walk_through.yaml**, a total of 5 stages were executed.
The **hpmax** metric represents the objective that would have been maximized
if this had been a hyperparameter tuning job.

    (dilbert) D:\dilbert\rl_nexus>xt list runs job265
    
    workspace  job     target   run       status      iters  s1-hrs  s1-rew  s1-win  s2-hrs  s2-rew  s2-win  s3-hrs  s3-loss  s3-acc  s4-hrs  s4-rew  s4-win  s5-hrs  s5-rew  s5-win  hpmax
    
    ws1        job265  azb-cpu  run527    completed
    ws1        job265  azb-cpu  run527.1  completed  100000   0.390   0.070   0.997   0.155   0.071   0.997   0.089    0.326  84.375   0.380   0.072   1.000   0.115   0.073   0.997  0.072
    ws1        job265  azb-cpu  run527.2  completed  100000   0.386   0.074   1.000   0.156   0.073   1.000   0.089    0.313  85.575   0.378   0.070   1.000   0.117   0.067   0.997  0.067
    ws1        job265  azb-cpu  run527.3  completed  100000   0.388   0.069   1.000   0.155   0.069   1.000   0.089    0.309  85.925   0.382   0.070   1.000   0.116   0.069   1.000  0.070
    ws1        job265  azb-cpu  run527.4  completed  100000   0.388   0.061   0.987   0.155   0.072   1.000   0.089    0.314  85.575   0.384   0.073   1.000   0.115   0.069   1.000  0.072
    ws1        job265  azb-cpu  run528    completed
    ws1        job265  azb-cpu  run528.1  completed  100000   0.385   0.067   1.000   0.155   0.069   1.000   0.089    0.304  86.125   0.382   0.068   1.000   0.114   0.069   1.000  0.070
    ws1        job265  azb-cpu  run528.2  completed  100000   0.387   0.067   1.000   0.155   0.070   1.000   0.089    0.330  84.800   0.379   0.069   1.000   0.114   0.071   1.000  0.071
    ws1        job265  azb-cpu  run528.3  completed  100000   0.385   0.071   1.000   0.155   0.071   1.000   0.088    0.300  86.812   0.376   0.072   1.000   0.113   0.066   1.000  0.068
    ws1        job265  azb-cpu  run528.4  completed  100000   0.387   0.071   1.000   0.155   0.069   1.000   0.089    0.309  85.987   0.384   0.070   1.000   0.112   0.071   1.000  0.072
    ws1        job265  azb-cpu  run529    completed
    ws1        job265  azb-cpu  run529.1  completed  100000   0.391   0.070   1.000   0.156   0.072   0.995   0.090    0.310  85.838   0.386   0.067   0.997   0.116   0.067   1.000  0.069
    ws1        job265  azb-cpu  run529.2  completed  100000   0.389   0.074   1.000   0.155   0.072   1.000   0.089    0.297  86.763   0.380   0.071   0.997   0.115   0.071   1.000  0.072
    ws1        job265  azb-cpu  run529.3  completed  100000   0.393   0.068   1.000   0.156   0.071   1.000   0.090    0.303  86.037   0.385   0.072   1.000   0.116   0.073   1.000  0.071
    ws1        job265  azb-cpu  run529.4  completed  100000   0.393   0.071   1.000   0.156   0.072   1.000   0.090    0.312  85.463   0.385   0.061   0.965   0.116   0.069   1.000  0.069
    ws1        job265  azb-cpu  run530    completed
    ws1        job265  azb-cpu  run530.1  completed  100000   0.387   0.074   1.000   0.157   0.072   1.000   0.089    0.290  87.200   0.386   0.066   1.000   0.114   0.073   1.000  0.071
    ws1        job265  azb-cpu  run530.2  completed  100000   0.386   0.071   1.000   0.155   0.069   1.000   0.088    0.300  86.625   0.379   0.071   1.000   0.113   0.068   1.000  0.070
    ws1        job265  azb-cpu  run530.3  completed  100000   0.383   0.065   0.997   0.154   0.068   1.000   0.089    0.304  86.513   0.374   0.074   1.000   0.113   0.073   1.000  0.072
    ws1        job265  azb-cpu  run530.4  completed  100000   0.389   0.068   0.997   0.156   0.069   1.000   0.088    0.305  85.838   0.379   0.071   1.000   0.114   0.069   1.000  0.072
    ws1        job265  azb-cpu  run531    completed
    ws1        job265  azb-cpu  run531.1  completed  100000   0.386   0.070   1.000   0.154   0.072   1.000   0.089    0.313  85.600   0.385   0.070   1.000   0.119   0.068   1.000  0.069
    ws1        job265  azb-cpu  run531.2  completed  100000   0.387   0.070   1.000   0.155   0.072   1.000   0.089    0.318  85.875   0.381   0.070   1.000   0.121   0.072   1.000  0.070
    ws1        job265  azb-cpu  run531.3  completed  100000   0.385   0.073   1.000   0.153   0.072   1.000   0.089    0.316  85.700   0.382   0.069   1.000   0.119   0.070   1.000  0.071
    ws1        job265  azb-cpu  run531.4  completed  100000   0.387   0.068   0.997   0.153   0.069   1.000   0.088    0.324  85.088   0.382   0.069   1.000   0.119   0.075   1.000  0.072
    
    workspace  job     target   run       status      iters  s1-hrs  s1-rew  s1-win  s2-hrs  s2-rew  s2-win  s3-hrs  s3-loss  s3-acc  s4-hrs  s4-rew  s4-win  s5-hrs  s5-rew  s5-win  hpmax
    
    total runs listed: 25

Unlike **job265**, which used **Reordered_Observation_Trainer** for SSL training,
**job266** used **Masked_Trainer** with **final_action_predictor: true** (behavior cloning). 
We will refer to these two jobs in the tutorial on plotting.

    (dilbert) D:\dilbert\rl_nexus>xt list runs job213
    
    workspace  job     target   run       status      steps   s1-hrs  s1-rew  s1-win  s2-hrs  s2-rew  s2-win  s3-hrs  s3-loss  s4-hrs  s4-rew  s4-win  s5-hrs  s5-rew  s5-win
    
    ws1        job213  azb-cpu  run198    completed
    ws1        job213  azb-cpu  run198.1  completed  100000    0.379   0.071   1.000   0.161   0.071   1.000   0.086    0.112   0.379   0.072   0.997   0.117   0.073   1.000
    ws1        job213  azb-cpu  run198.2  completed  100000    0.379   0.066   0.997   0.159   0.069   1.000   0.085    0.118   0.381   0.073   1.000   0.115   0.070   1.000
    ws1        job213  azb-cpu  run198.3  completed  100000    0.390   0.072   1.000   0.161   0.072   1.000   0.086    0.124   0.384   0.075   1.000   0.117   0.071   1.000
    ws1        job213  azb-cpu  run198.4  completed  100000    0.383   0.070   1.000   0.158   0.071   1.000   0.085    0.152   0.381   0.069   1.000   0.114   0.069   1.000
    ws1        job213  azb-cpu  run199    completed
    ws1        job213  azb-cpu  run199.1  completed  100000    0.391   0.070   1.000   0.160   0.069   1.000   0.087    0.175   0.389   0.072   1.000   0.120   0.068   1.000
    ws1        job213  azb-cpu  run199.2  completed  100000    0.382   0.069   1.000   0.158   0.069   1.000   0.085    0.090   0.382   0.073   1.000   0.119   0.073   1.000
    ws1        job213  azb-cpu  run199.3  completed  100000    0.387   0.073   1.000   0.159   0.073   1.000   0.086    0.082   0.385   0.069   1.000   0.120   0.073   1.000
    ws1        job213  azb-cpu  run199.4  completed  100000    0.386   0.071   1.000   0.158   0.073   1.000   0.086    0.095   0.382   0.071   1.000   0.119   0.070   1.000
    ws1        job213  azb-cpu  run200    completed
    ws1        job213  azb-cpu  run200.1  completed  100000    0.387   0.073   1.000   0.158   0.071   1.000   0.086    0.095   0.385   0.074   1.000   0.115   0.073   0.997
    ws1        job213  azb-cpu  run200.2  completed  100000    0.385   0.074   1.000   0.158   0.071   1.000   0.086    0.132   0.383   0.070   1.000   0.115   0.073   1.000
    ws1        job213  azb-cpu  run200.3  completed  100000    0.385   0.071   1.000   0.157   0.071   1.000   0.085    0.108   0.380   0.070   1.000   0.114   0.072   1.000
    ws1        job213  azb-cpu  run200.4  completed  100000    0.391   0.068   1.000   0.160   0.070   1.000   0.086    0.120   0.384   0.074   1.000   0.114   0.072   1.000
    ws1        job213  azb-cpu  run201    completed
    ws1        job213  azb-cpu  run201.1  completed  100000    0.379   0.072   1.000   0.160   0.070   1.000   0.085    0.154   0.379   0.073   1.000   0.115   0.075   1.000
    ws1        job213  azb-cpu  run201.2  completed  100000    0.388   0.074   1.000   0.161   0.073   1.000   0.086    0.109   0.383   0.074   1.000   0.117   0.071   1.000
    ws1        job213  azb-cpu  run201.3  completed  100000    0.384   0.071   1.000   0.158   0.072   1.000   0.085    0.134   0.384   0.071   1.000   0.115   0.068   1.000
    ws1        job213  azb-cpu  run201.4  completed  100000    0.387   0.071   1.000   0.159   0.071   1.000   0.086    0.097   0.383   0.071   1.000   0.116   0.065   1.000
    ws1        job213  azb-cpu  run202    completed
    ws1        job213  azb-cpu  run202.1  completed  100000    0.383   0.071   1.000   0.158   0.073   1.000   0.086    0.142   0.383   0.071   1.000   0.127   0.067   1.000
    ws1        job213  azb-cpu  run202.2  completed  100000    0.388   0.068   1.000   0.158   0.068   1.000   0.086    0.115   0.388   0.071   1.000   0.128   0.074   1.000
    ws1        job213  azb-cpu  run202.3  completed  100000    0.385   0.074   1.000   0.157   0.074   1.000   0.086    0.148   0.384   0.068   0.983   0.127   0.072   1.000
    ws1        job213  azb-cpu  run202.4  completed  100000    0.384   0.069   0.997   0.156   0.066   1.000   0.086    0.131   0.382   0.070   1.000   0.126   0.072   1.000
    
    workspace  job     target   run       status      steps   s1-hrs  s1-rew  s1-win  s2-hrs  s2-rew  s2-win  s3-hrs  s3-loss  s4-hrs  s4-rew  s4-win  s5-hrs  s5-rew  s5-win
    
    total runs listed: 25

## Monitoring jobs in Azure Batch Explorer

Because the compute target was set to **azb-cpu** for these jobs, 
they were deployed through the Azure Batch service called **dilbertbatch**.
To monitor live jobs, first download, install, and run the [Azure Batch Explorer](https://azure.github.io/BatchExplorer/).
As a member of the Dilbert group, you should have access to the **dilbertbatch** service in the **RL & Robotics** subscription.
These should appear on the **Dash** page once you sign in to Batch Explorer.

![](../../../../images/xt1.png)

Immediately after launching a job like those above, the **Pools** page should show a pool expanding from 0 to 5 nodes.

![](../../../../images/xt2.png)

A few minutes later, node boxes should appear and start changing color according to their status. 
Here, the green node is already running.

![](../../../../images/xt3.png)

The **Jobs** page shows a list of specific jobs, either running or completed.
Use this page to terminate a job, or to remove old jobs that are no longer running.
Deleting a job will also delete its pool, if the pool is still running.
But deleting a job will not remove that job's results from Azure Storage.

After clicking on your job, you should see one task on the right for each allocated node.

![](../../../../images/xt4.png)

Clicking on a particular task gives you access to files on that node, such as the **stdout.txt** file shown below.
This file receives console output from all runs, which can be helpful in debugging a job.
It's also possible to ssh directly to a vm from the Configuration tab.

![](../../../../images/xt5.png)

## [Next tutorial:  Plotting job results](3__Plotting_job_results.md)
