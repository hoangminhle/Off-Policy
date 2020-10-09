# Tutorial 3: Plotting job results

Job results can be plotted from the command line. This XT command will plot a separate metric curve for each run in the job.

    xt plot job265 s1-rew

![](../../../../images/xt-plot1.png)

To aggregate the runs into a single averaged curve, include --aggregate and --group-by arguments. 
(Five characters suffice to specify many arguments, as shown here.)

    xt plot job265 s1-rew --aggre=mean --group=job

![](../../../../images/xt-plot2.png)

The --shadow-type argument can display standard deviation (std), standard error (sem), or other measures of variance.

    xt plot job265 s1-rew --aggre=mean --group=job --shadow-type=std

![](../../../../images/xt-plot3.png)

To plot multiple metrics from the same job, provide the metric names in a comma-separated list:

    xt plot job265 s1-rew,s4-rew --aggre=mean --group=job

![](../../../../images/xt-plot4.png)

To plot metrics from different jobs, provide the job names in a comma-separated list. 

    xt plot job265,job266 s1-rew --aggre=mean --group=job

Here, we see that the first-stage reward was very similar for these two jobs, as expected:
![](../../../../images/xt-plot5.png)

But the s4-rew metrics (after pre-training) show a large difference between job265 and job266.
Using behavior cloning (instead of observation reordering) clearly improves downstream training performance. 

    xt plot job265,job266 s4-rew --aggre=mean --group=job

![](../../../../images/xt-plot6.png)

Multiple metrics can be plotted from multiple jobs:

    xt plot job265,job266 s1-rew,s4-rew --aggre=mean --group=job

![](../../../../images/xt-plot7.png)

The legend titles and the x axis title can be overridden with more meaningful strings. 

    xt plot job265,job266 s1-rew,s4-rew --aggre=mean --group=job --legend-titles='No pretraining','Pretrained by observation reordering','No pretraining','Pretrained by next action prediction' --x-label='Training steps'

![](../../../../images/xt-plot8.png)

Pass Matplotlib arguments for more control over plotting. 

    xt plot job265 s1-rew --legend-args={loc=right}

    xt plot job265 s1-rew --plot-args={linestyle=--}

    xt plot job265 s1-rew --plot-args={marker=o}

These and other XT commands are documented in detail online:

    xt --browse

## [Next tutorial:  Downloading files](4__Downloading_files.md)
