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

  A2 proposes an in-processing method that modifies the cost functions/constraints, adding constraints to the loss function so that the algorithm can take fairness into account. A7 introduces a way to select features in a more fair way by considering attributes’ marginal accuracy and discrimination coefficients.  
 
  Based on the results, we conclude that between the two methods, the feature selection approach does a better job of increasing accuracy and lowering the calibration. In addition, with similar accuracy scores, S-SVM is more promising regarding trade-off accuracy and fairness considerations. 

	
**Contribution statement**: 

+ Xilin Huang: Responsible for A2 C-SVM part. Did research for A2 C-SVM part. Implemented C-SVM model. And wrote the comment for C-SVM part.
+ Kieu-Giang Nguyen: did research on A2 methods, implemented A2 C-LR model based on the paper and source code, and validated the algorithm using accuracy and calibration.
+ Gabriel Spade:
+ Yayuan Wang (presenter): learned both A2 & A7 papers, outlined the overall structure of the python notebook, did data cleansing, constructed the baseline models (logistic regression and SVM without fairness considerations), integrated each part from the teammates, and completed the model evaluation/comparison part. 
+ Jiapeng Xu: researched paper A7, implemented the algorithm in the paper, validated the algorithm using accuracy and calibration on our baseline models, and finally, contributed the explanation of algorithm A7 on the presentation slides

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
