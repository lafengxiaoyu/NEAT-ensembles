# NEAT-ensembles
This repository is a support for my master's thesis "NEAT-based Multiclass Classificationwith Class Binarization"

* [Zhenyu Gao](https://scholar.google.com/citations?user=F8bdTwUAAAAJ&hl=en)
* [Gongjin Lan](https://scholar.google.com/citations?user=WBfADs4AAAAJ&hl=en&oi=ao)(supervisor)

## Introduction
Multiclass classification is a fundamental and challenging task in machine learning. 
Class binarization is a popular method to achieve multiclass classification by converting multiclass classification to multiple binary classifications. 
NeuroEvolution, such as NeuroEvolution of Augmenting Topologies (NEAT), is broadly used to generate Artificial Neural Networks by applying evolutionary algorithms. 

In this paper, we propose a new method, ECOC-NEAT, which applies Error-Correcting Output Codes (ECOC) to improve the multiclass classification of NEAT. 
The experimental results illustrate that ECOC-NEAT with a considerable number of binary classifiers is highly likely to perform well. 
ECOC-NEAT also shows significant benefits in a flexible number of binary classifiers and strong robustness against errors.

The structure of ECOC-NEAT is like the following picture shows.


![Structures of ECOC-NEAT](https://github.com/lafengxiaoyu/NEAT-ensembles/blob/main/ECOCNEAT.png)

## Datasets
There are three datasets, which are Digit from [Skilearn package](https://scikit-learn.org/stable/), Ecoli and Satellite datasets from the [UCI ML repos](https://archive.ics.uci.edu/ml/index.php).  

## Results
![Results of different methods](https://github.com/lafengxiaoyu/NEAT-ensembles/blob/main/Results.png)


## Good to know
- the slurm files are used for running in clusters
- please cite this paper if it is helpful to you.
