# Tutorial 5: Hyperparameter search

## Preparing the runspec

To configure a runspec to perform hyperparameter tuning, the **hp_tuning** property in the XT section
must be set to a supported tuning method such as **random**. 
As an example, here is the top of [useful_run_specs/random_hp_search.yaml](../../../useful_run_specs/random_hp_search.yaml):

    component: root/Root_Component
    log_to_tensorboard: true
    experiment_path: ../results
    job_launcher:
      component: job_launchers/XT
      enabled: 1
      hp_tuning: random
      total_runs: 500
      compute_nodes: 25
      runs_per_node: 4
      compute_target: azb-cpu
      low_priority: true
      hold: false

The runspec must also contain a **hyperparameters** section such as this:

    hyperparameters:
      - name: &seqlen
          ordered_tuning_values: [4, 6, 8, 12, 16, 24, 32]
          tuned_value: 24
      - name: &disc
          ordered_tuning_values: [0.95, 0.90, 0.85, 0.80]
          tuned_value: 0.90
      - name: &clip
          ordered_tuning_values: [64, 128, 256, 512, 1024, 2048]
          tuned_value: 2048
      - name: &lr
          ordered_tuning_values: [1.0e-4, 1.6e-4, 2.5e-4, 4.0e-4, 6.3e-4, 1.0e-3, 1.6e-3, 2.5e-3, 4.0e-3, 6.3e-3]
          tuned_value: 0.0004
      - name: &rscale
          ordered_tuning_values: [2, 4, 8, 16, 32]
          tuned_value: 32
      - name: &embed
          ordered_tuning_values: [32, 64, 128, 256, 512, 1024]
          tuned_value: 1024
      - name: &units
          ordered_tuning_values: [128, 192, 256, 384, 512]
          tuned_value: 384
      - name: &out
          ordered_tuning_values: [32, 64, 128, 256, 512, 1024]
          tuned_value: 1024

This section defines 8 HPs to be tuned, each one described by 3 lines:

1. The HP name, which is an arbitrary yaml &anchor. This string must be referenced by at least one *alias in the runspec.
Try to use reasonably short HP names that are meaningful in the context of your experiment.
1. The list of discrete values which the HP is allowed to take. (Continuous ranges are not currently supported.)
These values can be integers, floats, bools, and strings (including component names).
Most HP values carry some notion of ordering, in the sense that neighboring values are expected to produce more
similar effects than more distant values.
Random search disregards ordering, but DGD and Bayesian Optimization leverage it.
The values should be listed in their natural order (either high-to-low or low-to-high).
If the HP values possess no inherent order, their key should be **unordered_tuning_values**.
Refer to each component's default yaml file for reasonable ranges of its most commonly tuned properties.
1. The single value which will be used by default in any run that is not part of an HP tuning job. 
This makes it easy to test a single HP configuration without manually replacing all HP references with literal values.

#### Avoid wasteful HP searches

The efficiency of the search can be drastically impaired if any hyperparameter listed in a runspec has no effect on the experiment.
This can happen in several ways:
* If the HP anchor is not referenced in the rest of the runspec.
* If the HP anchor is referenced only by part of the runspec that does not actually get executed.
* If the Python code does not use the variable controlled by the HP.
* If one HP (like layer_size) depends on another HP (like num_layers) such that some value of the latter (num_layers=0)
causes the former HP to be ignored.

## Launching the hyperparameter search

Before launching the HP tuning job, it's especially important to follow the recommended [pre-flight checklist](1__Launching_jobs.md).
In our example here:

* Look over [useful_run_specs/random_hp_search.yaml](../../../useful_run_specs/random_hp_search.yaml), noting how:
    * **randint** is used to prevent deterministic behavior.
    * **iters_per_report** is set to 10% of **max_iterations**. 
    * the XT component is enabled.
* Perform a local run as a sanity check.
    * Copy **rl_nexus/useful_run_specs/random_hp_search.yaml** to **rl_nexus/spec.yaml**.
    * Modify **spec.yaml**:
        * Disable the XT component.
        * Set **max_iterations** to 5000 and **iters_per_report** to 1000.
    * Execute **spec.yaml**.

```
(dilbert) D:\dilbert\rl_nexus>python run.py spec.yaml
Chosen hyperparameter values:
  seqlen: 24
  disc: 0.9
  clip: 2048
  lr: 0.0004
  rscale: 32
  embed: 1024
  units: 384
  out: 1024
    
Processing stage 1 of 1
RL_Proc:  Started
  0.002 hrs         1,000 iters      0.00200 rew      0.11765 win
  0.004 hrs         2,000 iters      0.00200 rew      0.11111 win
  0.007 hrs         3,000 iters      0.00100 rew      0.05882 win
  0.009 hrs         4,000 iters      0.00100 rew      0.05882 win
  0.011 hrs         5,000 iters      0.00200 rew      0.11111 win
Stage summary (mean reward per step):  0.00160
RL_Proc:  Completed

Objective that would be maximized by hyperparameter tuning (hpmax):  0.00160

All processing stages completed.
```

There are several things to note in the console output of this local run:

* The listed HP values match the **tuned_value** settings in the hyperparameter section.
* As in any run, the **hpmax** value reported at the end is derived from the summary metric above it,
which in this case is the mean value of the **reward per step** metrics returned by the environment.
Always make sure that the first metric of the last stage of your runspec is the objective that you intend to optimize in the HP search.

Once you are certain that your original runspec (not the pre-flight copy) is ready to go, launch the job as normal.

    (dilbert) D:\dilbert\rl_nexus>python run.py useful_run_specs/random_hp_search.yaml
    ------- Runspec (spec.yaml) copied to launched_job_spec.yaml
    job278, target: azb-cpu, creating parent runs: 25/25, uploading 219 code files (zipped), ws1/run660, ws1/run661, ws1/run662, ws1/run663, ws1/run664, ws1/run665, ws1/run666, ws1/run667, ws1/run668, ws1/run669, ws1/run670, ws1/run671, ws1/run672, ws1/run673, ws1/run674, ws1/run675, ws1/run676, ws1/run677, ws1/run678, ws1/run679, ws1/run680, ws1/run681, ws1/run682, ws1/run683, ws1/run684, job submitted
    ------- Job launched. Use XT commands to access results.

## Hyperparameter search results

You can let the HP search run to completion or terminate it at any time.
The quickest way to find the best hyperparameter configuration is to sort the runs by **hpmax**:


    (dilbert) D:\dilbert\rl_nexus>xt list runs job278 --sort=metrics.hpmax --last=10
    
    workspace  job     target   run        status      iters  s1-hrs  s1-rew  s1-win  hpmax  seqlen   disc  clip       lr  rscale  embed  units   out
    
    ws1        job278  azb-cpu  run684.16  completed  100000   0.116   0.075   1.000  0.038      12  0.900   256  0.00063       8    512    192    64
    ws1        job278  azb-cpu  run680.5   completed  100000   0.124   0.073   1.000  0.039      24  0.850   256  0.00025      16    256    256   256
    ws1        job278  azb-cpu  run681.17  completed  100000   0.168   0.072   1.000  0.039      12  0.900   512  0.00040      32     64    384  1024
    ws1        job278  azb-cpu  run660.6   completed  100000   0.210   0.072   1.000  0.040      16  0.950  1024  0.00040       8    512    384   512
    ws1        job278  azb-cpu  run667.17  completed  100000   0.253   0.072   0.999  0.040       8  0.950   256  0.00025       8   1024    384  1024
    ws1        job278  azb-cpu  run678.8   completed  100000   0.147   0.073   0.999  0.040      24  0.900  1024  0.00025      16    256    256  1024
    ws1        job278  azb-cpu  run665.8   completed  100000   0.243   0.067   1.000  0.041       4  0.850  1024  0.00025      32    512    256   512
    ws1        job278  azb-cpu  run679.7   completed  100000   0.279   0.071   1.000  0.042       6  0.800  2048  0.00016      16   1024    384  1024
    ws1        job278  azb-cpu  run672.12  completed  100000   0.150   0.075   0.999  0.042      32  0.900   512  0.00100      16    256    384   512
    ws1        job278  azb-cpu  run662.7   completed  100000   0.146   0.075   1.000  0.045       8  0.900   256  0.00040       8    256    128  1024
    
    workspace  job     target   run        status      iters  s1-hrs  s1-rew  s1-win  hpmax  seqlen   disc  clip       lr  rscale  embed  units   out
    
    total runs listed: 10 (defaulted to --last=10)


The final row in this list shows that out of all 500 runs, **run662.7** achieved the highest value of **hpmax**.
The HP settings are shown to the right of each run.
You may wonder why **hpmax** is lower than **s1-rew** for each run in this table. 
Recall that the summary metric for each stage is the *mean* of all reports,
so **hpmax** is the mean of all reported **s1-rew** values, only the last of which is shown in the table.

You can also export all the run results to a file, then open that file in Excel to sort and analyze.

    xt list runs job278 --export=../jobs/job278_list_runs.csv

Because of selection bias, the metrics shown in the sorted list for **run662.7** are overly optimistic.
It is easy to obtain results that are free from selection bias:
* Copy **rl_nexus/useful_run_specs/random_hp_search.yaml** to **rl_nexus/spec.yaml** to modify.
* Copy the 8 winning HP settings shown above for **run662.7** into the **tuned_value** settings of **spec.yaml**.
* Enable the XT component.
* Set **hp_tuning: false**
* Lower **total_runs** to 20.
* Lower **compute_nodes** to 5.
* Execute the runspec.

```
(dilbert) D:\dilbert\rl_nexus>python run.py spec.yaml
------- Runspec (spec.yaml) copied to launched_job_spec.yaml
job289, target: azb-cpu, creating parent runs: 5/5, uploading 221 code files (zipped), ws1/run887, ws1/run888, ws1/run889, ws1/run890, ws1/run891, job submitted
------- Job launched. Use XT commands to access results.
```

After this job has finished, list the runs:

    (dilbert) D:\dilbert\rl_nexus>xt list runs job289
    
    workspace  job     target   run       status      iters  s1-hrs  s1-rew  s1-win   hpmax  seqlen   disc  clip       lr  rscale  embed  units   out
    
    ws1        job289  azb-cpu  run887    completed
    ws1        job289  azb-cpu  run887.1  completed  100000   0.111   0.075   0.999  0.0294       8  0.900   256  0.00040       8    256    128  1024
    ws1        job289  azb-cpu  run887.2  completed  100000   0.110   0.070   1.000  0.0203       8  0.900   256  0.00040       8    256    128  1024
    ws1        job289  azb-cpu  run887.3  completed  100000   0.112   0.072   1.000  0.0382       8  0.900   256  0.00040       8    256    128  1024
    ws1        job289  azb-cpu  run887.4  completed  100000   0.110   0.070   1.000  0.0329       8  0.900   256  0.00040       8    256    128  1024
    ws1        job289  azb-cpu  run888    completed
    ws1        job289  azb-cpu  run888.1  completed  100000   0.109   0.068   0.983  0.0472       8  0.900   256  0.00040       8    256    128  1024
    ws1        job289  azb-cpu  run888.2  completed  100000   0.110   0.069   0.997  0.0350       8  0.900   256  0.00040       8    256    128  1024
    ws1        job289  azb-cpu  run888.3  completed  100000   0.110   0.071   1.000  0.0442       8  0.900   256  0.00040       8    256    128  1024
    ws1        job289  azb-cpu  run888.4  completed  100000   0.108   0.070   1.000  0.0231       8  0.900   256  0.00040       8    256    128  1024
    ws1        job289  azb-cpu  run889    completed
    ws1        job289  azb-cpu  run889.1  completed  100000   0.109   0.027   0.809  0.0067       8  0.900   256  0.00040       8    256    128  1024
    ws1        job289  azb-cpu  run889.2  completed  100000   0.110   0.069   0.996  0.0270       8  0.900   256  0.00040       8    256    128  1024
    ws1        job289  azb-cpu  run889.3  completed  100000   0.110   0.071   0.999  0.0261       8  0.900   256  0.00040       8    256    128  1024
    ws1        job289  azb-cpu  run889.4  completed  100000   0.110   0.071   0.996  0.0426       8  0.900   256  0.00040       8    256    128  1024
    ws1        job289  azb-cpu  run890    completed
    ws1        job289  azb-cpu  run890.1  completed  100000   0.111   0.069   1.000  0.0228       8  0.900   256  0.00040       8    256    128  1024
    ws1        job289  azb-cpu  run890.2  completed  100000   0.112   0.072   0.994  0.0348       8  0.900   256  0.00040       8    256    128  1024
    ws1        job289  azb-cpu  run890.3  completed  100000   0.111   0.073   1.000  0.0357       8  0.900   256  0.00040       8    256    128  1024
    ws1        job289  azb-cpu  run890.4  completed  100000   0.113   0.072   1.000  0.0321       8  0.900   256  0.00040       8    256    128  1024
    ws1        job289  azb-cpu  run891    completed
    ws1        job289  azb-cpu  run891.1  completed  100000   0.111   0.069   0.999  0.0253       8  0.900   256  0.00040       8    256    128  1024
    ws1        job289  azb-cpu  run891.2  completed  100000   0.111   0.072   1.000  0.0301       8  0.900   256  0.00040       8    256    128  1024
    ws1        job289  azb-cpu  run891.3  completed  100000   0.111   0.068   1.000  0.0293       8  0.900   256  0.00040       8    256    128  1024
    ws1        job289  azb-cpu  run891.4  completed  100000   0.110   0.061   0.998  0.0165       8  0.900   256  0.00040       8    256    128  1024
    
    workspace  job     target   run       status      iters  s1-hrs  s1-rew  s1-win   hpmax  seqlen   disc  clip       lr  rscale  embed  units   out
    
    total runs listed: 25

As expected, the **mean hpmax** for these 20 final runs is **0.030**, much worse than the **0.045** of the HP search.
This type of selection bias is present in studies that publish the performance of *"the best"* or *"the best 5"* 
out of some number of random runs with different HP configurations.

Finally, we plot the 20 unbiased runs for this tuned hyperparameter configuration:

    xt plot job289 s1-rew --aggre=mean --group=job --shadow-type=std

![](../../../../images/xt-plot9.png)

## Distributed Grid Descent (DGD)

Specify **hp_tuning: dgd** to perform a DGD hyperparameter search.
Details of DGD can be found in [Working Memory Graphs](https://arxiv.org/abs/1911.07141). 

Use [eval_hp_search.py](../../../scripts/readme.md) to evaluate the results of a DGD job or any other HP search,
in order to decide when the results have converged sufficiently to terminate the search.
As shown in the example below, this tool reports the best HP combinations found, 
and estimates the performance they would obtain on new training runs.
The best HP combination so far is the last in the list. 

    (dilbert) D:\dilbert\rl_nexus>python scripts\eval_hp_search.py job411
    seqlen    disc      clip      lr        rscale    embed     units     out      BEST FOR   EST HPMAX  RUNS  BEST
    32        0.95      128       0.0063    32        64        128       32         1 runs    0.020409    1   run3757.3
    24        0.85      128       0.0025    8         32        256       64         3 runs    0.020524    1   run3768.2
    32        0.9       512       0.0016    32        512       128       64         6 runs    0.033908    3   run3762.5
    ...
    24        0.9       2048      0.00063   16        512       128       512        8 runs    0.049182   16   run3755.152
    24        0.9       1024      0.001     16        512       256       512        6 runs    0.047129    9   run3765.165
    24        0.9       2048      0.00063   16        512       128       512      155 runs    0.049182   16   run3755.152
    seqlen    disc      clip      lr        rscale    embed     units     out      BEST FOR   EST HPMAX  RUNS  BEST
    History of the best hyperparameter combination after each of the 4781 completed runs.

![](../../../../images/eval_hp_search.png)

The same plot is provided by the following XT command:

    (dilbert) D:\dilbert\rl_nexus>xt plot summary job411
