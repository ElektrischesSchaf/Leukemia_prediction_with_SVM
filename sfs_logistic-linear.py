import os
import csv
import numpy as np
import random
import csv
import time
from sklearn import  svm, metrics
from  sklearn.svm import SVC
from sklearn.svm import SVR

# Standard scientific Python imports
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from mlxtend.feature_selection import SequentialFeatureSelector as SFS


feature_number=140
iteration_number=10

root_name='SFS_logistic-linear_'
feature_number_tag='features_number_'
iteration_tag='_iteration_number_'

# feature number loop start
for i_feature in range(0, feature_number, 10):

    date = time.strftime('%Y-%m-%d_T_%H_%M_%S')
    f1=open(root_name+feature_number_tag+str(i_feature+1)+iteration_tag+str(iteration_number)+'_'+date+'.csv','w')

    #iteration loop start
    for i_number in range(iteration_number):
        
        #estimator = KNeighborsClassifier(n_neighbors=3)
        #estimator=SVR(kernel="rbf")
        estimator=LogisticRegression()
        #estimator=Ridge(alpha=.5)

        sfs1 = SFS(estimator, 
                k_features=i_feature+1, 
                forward=True, 
                floating=False, 
                verbose=2,
                scoring='accuracy',
                cv=0)

        head=[]
        data=[]
        selected_index=[]
        selector_support=[]

        with open('leukemia.csv', mode='r') as infile:    
            rows=csv.reader(infile) 
            for row in rows:
                head.append(row[0])
                data.append(row[1:])

        for i in range(72):
            if head[i]=='ALL':
                head[i]=1
            if head[i]=='AML':
                head[i]=2


        data_array=np.array(data)
        head_array=np.array(head)

        for i in range(72):
            ran_num=random.randint(1,70)

            temp1=head_array[i].copy()
            head_array[i]=head_array[ran_num].copy()
            head_array[ran_num]=temp1.copy()

            temp2=data_array[i].copy()
            data_array[i]=data_array[ran_num].copy()
            data_array[ran_num]=temp2.copy()

        #data_array.reshape(1,-1)

        X=data_array.astype(np.float64)
        y=head_array.astype(np.float64)

        print('\n---------------------------------\n')
        print("X.shape: ",end='')
        print(X.shape)
        print('\n---------------------------------\n')
        print("y.shape: ",end='')

        sfs1 = sfs1.fit(X[:37], y[:37])

        print('\n---------------------------------\n')
        print('sfs1.subsets_:  ',end='')
        print(sfs1.subsets_)
        print('\n---------------------------------\n')
        print('sfs1.subsets_[i_feature+1][feature_names]:  ',end='')
        print(sfs1.subsets_[i_feature+1]['feature_names'])
        print('\n---------------------------------\n')
        selected_index=list(sfs1.subsets_[i_feature+1]['feature_names']).copy()
        print(type(selected_index))
        print('\n---------------------------------\n')

        print('X column number: ',end='')
        print(X.shape[1])
        print('\n---------------------------------\n')

        for i in range(X.shape[1]):
            #print(i)
            selector_support.append(False)

        print('length of selected_index:  ',end='')
        print(len(selected_index))
        print('\n---------------------------------\n')

        for i in range(len(selected_index)):
            selector_support[  int(     selected_index[i]   )   ]=True

        #print('selector_support=',end='')
        #print(selector_support)
        print('\n---------------------------------\n')
        print('length of selector_support:  ',end='')
        print(len(selector_support))
        print('\n---------------------------------\n')

        new_X=X[:,selector_support]
        classifier = svm.SVC(gamma=0.001,kernel='linear')
        classifier.fit(new_X[:37],y[:37])
        print('new_X[:37] shape:  ',end='')
        print(new_X[:37].shape)
        print('\n---------------------------------\n')
        print('y[:37] shape:  ',end='')
        print(y[:37].shape)
        print('\n---------------------------------\n')

        expected_training_data=y[:37]
        predicted_training_data=classifier.predict(new_X[:37])

        expected=y[37:]    
        predicted=classifier.predict(new_X[37:])

        print('expected: ',end='')
        print(expected)
        print('\n---------------------------------\n')
        print('predicted: ',end='')
        print(predicted)
        print('\n---------------------------------\n')
        
        print("Classification report for classifier %s:\n%s\n" % (classifier, metrics.classification_report(expected, predicted)))

        print('MCC for training data: ',end='')
        training_matthews_correlation_coefficient=metrics.matthews_corrcoef(expected_training_data,predicted_training_data)
        training_matthews_correlation_coefficient=float("{0:.3f}".format(training_matthews_correlation_coefficient))
        print(str(training_matthews_correlation_coefficient))

        print('MCC for testing data: ',end='')
        testing_matthews_correlation_coefficient=metrics.matthews_corrcoef(expected,predicted)
        testing_matthews_correlation_coefficient=float("{0:.3f}".format(testing_matthews_correlation_coefficient))
        print(str(testing_matthews_correlation_coefficient))

        print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
        print('\n---------------------------------\n')
        f1.write('Iteration   '+str(i_number+1)+'   ')
        f1.write('MCC_1     '+str(training_matthews_correlation_coefficient)+'     ')
        f1.write('MCC_2     '+str(testing_matthews_correlation_coefficient)+'\n')
        f1.write(metrics.classification_report(expected, predicted) )
        f1.write('\n')
    #iteration loop end

# feature number loop end
