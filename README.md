# CommentG
This repository stores the source code of inline comment generation.
You can find more details, analyses, and baseline results in our paper "ICG: A Benchmark Dataset for Inline Comments Generation Task".

## Link of Data set
* The data set could be found in here, click the [download](https://zenodo.org/record/8022713) button.

## Folders Introduction
* [/step1 dataset/](https://github.com/Sherww/CommentG/tree/main/dataset). This file (click [here](https://github.com/Sherww/CommentG/tree/main/dataset/example_data.jsonl)) contains several examples of inline comments and corresponding code, and its context. The complete dataset can be found in [here](https://zenodo.org/record/8022713). 
* [/step2 dataset evaluation/](https://github.com/Sherww/CommentG/tree/main/dataset%20evaluation). This folder contains the code and results for the automatic classification of the inline comments and the results. The [./statistics/](https://github.com/Sherww/CommentG/tree/main/dataset%20evaluation/statistic) folder contains the statistical results and calculation methods of different categories, and the [./manual review/](https://github.com/Sherww/CommentG/tree/main/dataset%20evaluation/manual%20review) folder contains the dataset used in manual review and statistical analysis code. The [./classify](https://github.com/Sherww/CommentG/tree/main/dataset%20evaluation/classify) folder contains the models used in automatic classification.
* [/step3 generation/](https://github.com/Sherww/CommentG/tree/main/generation). This folder contains the baseline models used for inline comments generation.

## Train
* The model training and prediction was conducted on a machine with Nvidia GTX 1080 GPU, Intel(R) Core(TM) i7-6700 CPU and 16 GB RAM. The operating system is Ubuntu.



