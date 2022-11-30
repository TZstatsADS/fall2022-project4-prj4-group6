# Project 4: Machine Learning Fairness

### [Project Description](doc/project4_desc.md)

Term: Fall 2022

+ Team #6
+ Project title: Machine Learning Fairness 
+ Team members
	+ Xilin Huang
	+ Kieu-Giang Nguyen
	+ Gabriel Spade
	+ Yayuan Wang
	+ Jiapeng Xu
	
+ Project summary: In this project, we are comparing two methodologies introduced by [A2-Maximizing accuracy under fairness constraints (C-SVM and C-LR)](https://arxiv.org/abs/1507.05259) and [A7-Information Theoretic Measures for Fairness-aware Feature selection (FFS)](https://arxiv.org/abs/2106.00772) to obtain a better understanding of the trade-off between accuracy and fairness. The criterion we use to quantify fairness is calibration, meaning that the prediction accuracy of the protected group ought to be equal to the accuracy of the unprotected group.
  A2 proposes an in-processing method which modifies the cost functions/constraints, adding constraints to the loss function so that the algorithm can take the fairness into account. A7 introduces a way to select features in a more fair way by considering attributes’ marginal accuracy and discrimination coefficients. 
  Baed on the results, we conclude that between two methods, the feature selection approach does a better job of increasing accuracy and lowering the calibration. In addition, With similar accuracy scores, S-SVM is more promising regarding trade-off accuracy and fairness considerations. 

	
**Contribution statement**: 

Following [suggestions](http://nicercode.github.io/blog/2013-04-05-projects/) by [RICH FITZJOHN](http://nicercode.github.io/about/#Team) (@richfitz). This folder is orgarnized as follows.

```
proj/
├── lib/
├── data/
├── doc/
├── figs/
└── output/
```

Please see each subfolder for a README file.
