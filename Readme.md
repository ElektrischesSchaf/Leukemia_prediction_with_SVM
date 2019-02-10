## Using two kinds of feature selection methods, recursive feature elimination (RFE) and 
sequential feature selector (SFS) for support vector machine (SVM) to predict leukemia  
The leukemia dataset contains 7218 genes from 72 samples. These data are classified into two type of leukemia, acute 
lymphocytic leukemia (ALL) and acute myelocytic leukemia (AML).
We run 50 iterations with different feature numbers, feature selection methods, and different kernels. 
The feature numbers vary from 10 to 70. In each iteration, we randomize the order of the samples before selecting 
38 training data and 34 testing data.  That means the training data and testing data consist of different samples 
everytime. Then we calculate the Matthews correlation coefficient to evaluate the result of each feature selection method.
## The result visualized in violin plot:
![01](https://github.com/ElektrischesSchaf/Leukemia_prediction_with_SVM/blob/master/violin_plot/Training_RFE_Linear_Classification_Kernel.PNG)  
Training data with RFE linear classification kernel
---
![02](https://github.com/ElektrischesSchaf/Leukemia_prediction_with_SVM/blob/master/violin_plot/Testing_RFE_Linear_Classification_Kernel.PNG)
Testing data with RFE linear classification kernel
---
![03](https://github.com/ElektrischesSchaf/Leukemia_prediction_with_SVM/blob/master/violin_plot/Training_RFE_RBF_Classification_Kernel.PNG)
Training data with RFE RBF classification kernel
---
![04](https://github.com/ElektrischesSchaf/Leukemia_prediction_with_SVM/blob/master/violin_plot/Testing_RFE_RBF_Classification_Kernel.PNG)
Testing data with RFE RBF classification kernel
---
![05](https://github.com/ElektrischesSchaf/Leukemia_prediction_with_SVM/blob/master/violin_plot/Training_SFS_KNN_Regression_Kernel.PNG)
Training data with SFS KNN regression kernel
---
![06](https://github.com/ElektrischesSchaf/Leukemia_prediction_with_SVM/blob/master/violin_plot/Testing_SFS_KNN_Regression_Kernel.PNG)
Testing data with SFS KNN regression kernel
---
![07](https://github.com/ElektrischesSchaf/Leukemia_prediction_with_SVM/blob/master/violin_plot/Training_SFS_Logistic_Regression_Kernel.PNG)
Training data with SFS KNN logistic kernel
---
![08](https://github.com/ElektrischesSchaf/Leukemia_prediction_with_SVM/blob/master/violin_plot/Testing_SFS_Logistic_Regression_Kernel.PNG)
Testing data with SFS KNN logistic kernel
---