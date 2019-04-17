# Incremental Learning

*Also called lifelong learning, or continual learning.*

This repository will store all my implementations of Incremental Learning's papers.

## Structures

Every model must inherit `inclearn.models.base.IncrementalLearner`.


## iCaRL

I have not yet been able to reproduce the results of iCaRL as you can see on the
following figure:

![icarl](figures/icarl.png)

My experiments are in green, with their means & standard deviations plotted.
They were runned 40 times, with seed going from 1 to 40, each producing a
different classes ordering.

It seems that the mean is consistently lower by 10pts compared to iCaRL's paper
results. I guess it means something is wrong during the training & not simply
due to the distillation loss or the examplars selection because we can see this
loss from the first task.
