# CS7641 Machine Learning
This repository contains all the necessary files to reproduce all the figures, models and evaluation metrics that were submitted as part of the analysis reports for the assignments of OMSCS CS7641 Machine Learning course.\
This repository consists of four distinct projects referred to as assignments. 

### Assignment 1: Supervised Learning
The first assignment uses KNN, Decision Tree, Support Vector Machines (SVM), Boosting and Artificial Neural Networks (ANN) learners to classify two distinct datasets. The final result is printed in a text file and the learning curves and complexity curves are saved as .png files. Any other classification problem, balanced or imbalanced, binary or multiclass classification, could also be tackled by the current implementation. When adding a new dataset, the hyper-parameters search span are potentially required to be changed.

### Assignment 2: Randomized Optimization
The second assignment uses Randomized Hill Climbing, Simulated Annealing, Genetic Algorithm and MIMIC optimizers to maximize three different fitness functions. Additionally, it uses Randomized Hill Climbing, Simulated Annealing, Genetic Algorithm and Gradient Descent to tune a Neural Network set of weights and generates plots for comparison. The final result is generated as a set of plots and curves and is saved as .png files.

### Assignment 3: Unsupervised Learning and Dimensionality Reduction
The third assignment uses unsupervised learning algorithms, namely K-Means and Expectation Maximization (EM) in addition to dimensionality reduction algorithms, namely PCA, ICA, Random Projections (RP) and Feature Agglomeration (FA) to dimensionally reduce and cluster different datasets and extract meaningful data and plots, such as, t-SNE, parallel coordinates to make sense out of the process. Additionally, it investigates the effects of dimensionality reduction on ANN training time and performance. The final result is generated as a set of plots and curves and is saved as .png files.

### Assignment 4: Markov Decision Processes
The fourth assignment uses value iteration, policy iteration and Q-Learning to study two distinct Markov Decision Processes. The two problems are first, a grid world problem called Frozen Lake and second, mdptoolbox's Forest Management. Each of the algorithms find the optimal policy for both problems and different plots are generated to facilitate the comparison process. The final result is generated as a set of plots and curves and is saved as .png files.

## Required IDE and Libraries
The code in this repository was written and tested in PyCharm 2018.2.4 using the following libraries (the dependencies of each library is not mentioned). Note that for assignment 2, Conda Package Manager could be used to add a specific version of mlrose library (mlrose-hiive) to the environment. 

libraries for A1 | Version
--------------|------------
matplotlib | 3.1.2
numpy | 1.18.1
pandas | 0.25.3
scikit-learn | 0.22.1
scipy | 1.4.1

libraries for A2 | Version
--------------|------------
mlrose-hiive* | 1.2.0
matplotlib | 3.1.2
numpy | 1.18.1
pandas | 0.25.3
scikit-learn | 0.22.1
scipy | 1.4.1

libraries for A3 | Version
--------------|------------
matplotlib | 3.2.0
numpy | 1.18.1
pandas | 1.0.1
scikit-learn | 0.22.2
scipy | 1.4.1
seaborn | 0.10.0

libraries for A4 | Version
--------------|------------
mdptoolbox-hiive** | 4.0.3.1
gridworld-viz | 0.0.0
matplotlib | 3.1.2
numpy | 1.18.1
pandas | 0.25.3
scikit-learn | 0.22.1
scipy | 1.4.1

\* https://github.com/hiive/mlrose
\** https://github.com/hiive/hiivemdptoolbox

## How to Run
To run the code, simply run the runscript.py file under each assignment directory. The figures are generated under its corresponding  "Figures" folder and all the logs are generated in the root assignment directory.
