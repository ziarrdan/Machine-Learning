# CS7641 Machine Learning
This repository contains all the necessary files to reproduce all the figures, models and evaluation metrics that were submitted as part of the analysis reports for the assignments of OMSCS CS7641 Machine Learning course.\
This repository consists of four distinct projects referred to as assignments. 

### Assignment 1: Supervised Learning
The first assignment uses KNN, Decision Tree, Support Vector Machines (SVM), Boosting and Artificial Neural Networks (ANN) learners to classify two distinct datasets. The final result is printed in a text file and the learning curves and complexity curves are saved as .png files. Any other classification problem, balanced or imbalanced, binary or multiclass classification, could also be tackled by the current implementation. When adding a new dataset, the hyper-parameters search span are potentially required to be changed.

## Required IDE and Libraries
The code in this repository was written and tested in PyCharm 2018.2.4 using the following libraries (the dependencies of each library is not mentioned):

library | Version
--------------|------------
matplotlib | 3.1.2
numpy | 1.18.1
pandas | 0.25.3
scikit-learn | 0.22.1
scipy | 1.4.1

## How to Run
To run the code, simply run the runscript.py file under each assignment directory. The figures are generated under its corresponding  "Figures" folder and all the logs are generated in the root assignment directory.
