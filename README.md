# GradCAM-based declustrisation for cervical smear images

This directory contains code suited to CRIC or Bialystok dataset. Before executing download the chosen dataset.

Model weights and cluster images used in this code are stored [here](https://cloud.ibib.waw.pl/index.php/s/7aVb0DNS1l07mj5).

## Environment

Python     3.9

Tensorflow 2.5

Keras      2.4

Cuda   	  11.8




## Contents


### Directory Bialystok 
contains step by step code to perform declusterisation of cluster images contained in clusters directory.

To find the nuclei use segmenthsilempatchesloop.py

To segment cell based on previously detected nuclei use test_waterSeeded_split_tos.m

To classify the cells run cut_out.py, then colors2classes.py and finally drawresults.py.



### Directory Cric 
contains the segmenting-classifiny pipeline with declustering step fitted to the CRIC dataset.



_________________
Please contact me if you'll encounter any problems.
