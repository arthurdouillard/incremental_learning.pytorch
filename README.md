# Incremental Learning

*Also called lifelong learning, or continual learning.*

This repository will store all my implementations of Incremental Learning's papers.

## Structures

Every model must inherit `inclearn.models.base.IncrementalLearner`.

## Papers implemented:

:white_check_mark: --> Paper implemented & reached expected results.\
:construction: --> Runnable but not yet reached expected results.\
:x: --> Not yet implemented or barely working.\

[1]: :white_check_mark: iCaRL\
[2]: :construction: LwF\
[3]: :construction: End-to-End Incremental Learning\


## iCaRL

![icarl](figures/icarl.png)

My experiments are in green, with their means & standard deviations plotted.
They were runned 40 times, with seed going from 1 to 40, each producing a
different classes ordering.

The metric used is the `average incremental accuracy`:

> The result of the evaluation are curves of the classification accuracies after
> each batch of classes. If a single number is preferable, we report the average of
> these accuracies, called average incremental accuracy.

If I understood well, the accuracy at task i (computed on all seen tasks) is averaged
with all previous accuracy. A bit weird, but doing so get me a curve very similar
to what the papier displayed.

---

# References

[1] iCaRL:

```
@InProceedings{icarl,
author = {Rebuffi, Sylvestre-Alvise and Kolesnikov, Alexander and Sperl, Georg and Lampert, Christoph H.},
title = {iCaRL: Incremental Classifier and Representation Learning},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {July},
year = {2017}
}
```

[2]: LwF:

```
@ARTICLE{lwf,
author={Z. {Li} and D. {Hoiem}},
journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
title={Learning without Forgetting},
year={2018},
volume={40},
number={12},
pages={2935-2947},
keywords={convolution;feature extraction;feedforward neural nets;learning (artificial intelligence);fine-tuning adaption techniques;CNN;forgetting method;convolutional neural network;vision system;feature extraction;Feature extraction;Deep learning;Training data;Neural networks;Convolutional neural networks;Knowledge engineering;Learning systems;Visual perception;Convolutional neural networks;transfer learning;multi-task learning;deep learning;visual recognition},
doi={10.1109/TPAMI.2017.2773081},
ISSN={0162-8828},
month={Dec},}
```

[3]: End-to-End Incremental Learning:

```
@inproceedings{end_to_end_inc_learn,
  TITLE = {{End-to-End Incremental Learning}},
  AUTHOR = {Castro, Francisco M. and Mar{\'i}n-Jim{\'e}nez, Manuel J and Guil, Nicol{\'a}s and Schmid, Cordelia and Alahari, Karteek},
  URL = {https://hal.inria.fr/hal-01849366},
  BOOKTITLE = {{ECCV 2018 - European Conference on Computer Vision}},
  ADDRESS = {Munich, Germany},
  EDITOR = {Vittorio Ferrari and Martial Hebert and Cristian Sminchisescu and Yair Weiss},
  PUBLISHER = {{Springer}},
  SERIES = {Lecture Notes in Computer Science},
  VOLUME = {11216},
  PAGES = {241-257},
  YEAR = {2018},
  MONTH = Sep,
  DOI = {10.1007/978-3-030-01258-8\_15},
  KEYWORDS = {Incremental learning ; CNN ; Distillation loss ; Image classification},
  PDF = {https://hal.inria.fr/hal-01849366/file/IncrementalLearning_ECCV2018.pdf},
  HAL_ID = {hal-01849366},
  HAL_VERSION = {v1},
}
```
