# CommentG
This repository stores the source code of inline comment generation.
You can find more details, analyses, and baseline results in our paper "ICG: A Benchmark Dataset for Inline Comments Generation Task".

## Link of Data set
* The data set could be found in here, click the [download](https://zenodo.org/record/8022713) button.

## Project Selection

* Stars number could measure the popularity of a project on GitHub, which is widely used in research. Popular projects are generally actively developed and likely contain more inline comments. Therefore, following \[1-3], we rank Java projects based on Star numbers and select the top-ranked 8000 projects with the most stars as the experimental projects.


* Selection criteria. We use the same selection criteria as Zhang et al.\[4], which can be seen as follows: 

1) Projects with great popularity. Popular projects are generally actively developed and are likely to contain more inline comments. Following \[1-3], we use the Stars number to measure the popularity of a project on GitHub. We rank Java projects based on Star numbers, and select the top-ranked 1,000 projects with most stars as experimental projects.

2) English commented projects. We only consider projects with comments written in English. We first check if the comment can be encoded by ASCII. Then we calculate the percentage of comments that are all ASCII encoded in a project. If the percentage exceeds 90%, it will be considered as an English-commented project. Otherwise it will be considered as a non-English project and hence is removed.

3) Non-toy typical software development projects. These projects are mainly software development projects rather than for example documentation or experimental/test projects. We take a two-step method to filter out toy projects. First, we use heuristic patterns to identify potential toy projects, by checking whether their readme files contain keywords such as "toy", "test", "experiment", "learn",  and "exercises". Then, for each obtained project, we manually check its readme file and code base, and determine whether it is a toy project or not. The first three authors are involved in the checking process.


* \[1] X. Hu, G. Li, X. Xia, D. Lo, and Z. Jin, “Deep code comment generation with hybrid lexical and syntactical information,” Empirical Software Engineering, vol. 25, no. 3, pp. 2179–221
* \[2] T. D. Nguyen, A. T. Nguyen, and T. N. Nguyen, “Mapping api elements for code migration with vector representations,” in 2016 IEEE/ACM 38th International Conference on Software Engineering Companion (ICSE-C). IEEE, 2016, pp. 756–758.
* \[3] P. Leitner and C.-P. Bezemer, “An exploratory study of the state of practice of performance testing in java-based open source projects,” in Proceedings of the 8th ACM/SPEC on International Conference on Performance Engineering, 2017, pp. 373–3
* \[4] X. Zhang, W. Zou, L. Chen, Y. Li, and Y. Zhou, “Towards the analysis and completion of syntactic structure ellipsis for inline comments,” IEEE Transactions on Software Engineering, 2022

## Folders Introduction
* __/step1 dataset/__. This file (click [here](https://github.com/Sherww/CommentG/tree/main/dataset/example_data.jsonl)) contains several examples of inline comments and corresponding code, and its context. The complete dataset can be found in [here](https://zenodo.org/record/8022713). 
* __/step2 dataset evaluation/__. This folder contains the code and results for the automatic classification of the inline comments and the results. Specifically, the __./statistics/__ folder contains the statistical results and calculation methods of different categories. The __./manual review/__ folder contains the dataset used in manual review and statistical analysis code. The __./classify__ folder contains the models used in automatic classification.
* __/step3 generation/__. This folder contains the baseline models used for inline comments generation.

## Train
* The model training and prediction was conducted on a machine with Nvidia GTX 1080 GPU, Intel(R) Core(TM) i7-6700 CPU and 16 GB RAM. The operating system is Ubuntu.



