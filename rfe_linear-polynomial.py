import os
import csv
import numpy as np
import random
import csv
import time
# Standard scientific Python imports
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.feature_selection import RFE
from  sklearn.svm import SVC
from sklearn.svm import SVR


feature_number=140
iteration_number=50

classifier = svm.SVC(gamma=0.001)
root_name='RFE_linear-polynomial_'
feature_number_tag='features_number_'
iteration_tag='_iteration_number_'

# feature number loop start
for i_feature in range( 0, feature_number, 10):

    date = time.strftime('%Y-%m-%d_T_%H_%M_%S')
    f1=open(root_name+feature_number_tag+str(i_feature+1)+iteration_tag+str(iteration_number)+'_'+date+'.csv','w')

    #iteration loop start
    for i_number in range(iteration_number):
        
        head=[]
        data=[]

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

        X=data_array.astype(np.float64)
        y=head_array.astype(np.float64)
        print('\n---------------------------------\n')
        print("X.shape: ",end='')
        print(X.shape)
        print('\n---------------------------------\n')
        print("y.shape: ",end='')
        print(y.shape)
        #svc = SVC(kernel="linear", C=1)
        #rfe = RFE(estimator=LogisticRegression(), n_features_to_select=10, step=1)
        estimator=SVR(kernel="linear")
        selector = RFE(estimator, i_feature+1, step=200)
        selector = selector.fit(X[:37],y[:37])
        print('\n---------------------------------\n')
        print('selector.support: ',end='')
        print(selector.support_)
        print('\n---------------------------------\n')
        print('length of selector.support: ',end='')
        print(len(selector.support_))
        print('\n---------------------------------\n')
        new_X=X[:,selector.support_]
        print('new_X.shape:  ',end='')
        print(new_X.shape)
        print('\n---------------------------------\n')
        classifier = svm.SVC(gamma=0.001,kernel='poly')
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
        print('='*50)
        print('type of metrics.classification_report(expected, predicted):',end='')
        print(type(metrics.classification_report(expected, predicted) )  )
        print('\n')

        print('MCC for training data: ',end='')
        training_matthews_correlation_coefficient=metrics.matthews_corrcoef(expected_training_data,predicted_training_data)
        training_matthews_correlation_coefficient=float("{0:.3f}".format(training_matthews_correlation_coefficient))
        print(str(training_matthews_correlation_coefficient))

        print('MCC for testing data: ',end='')
        testing_matthews_correlation_coefficient=metrics.matthews_corrcoef(expected,predicted)
        testing_matthews_correlation_coefficient=float("{0:.3f}".format(testing_matthews_correlation_coefficient))
        print(str(testing_matthews_correlation_coefficient))

        f1.write('Iteration   '+str(i_number+1)+'   ')
        f1.write('MCC_1     '+str(training_matthews_correlation_coefficient)+'     ')
        f1.write('MCC_2     '+str(testing_matthews_correlation_coefficient)+'\n')
        f1.write(metrics.classification_report(expected, predicted) )
        f1.write('\n')
        
        print('='*50)
        print("Classification report for classifier %s:\n%s\n" % (classifier, metrics.classification_report(expected, predicted)))
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
        #print('expected=',end='')
        #print(expected)
        #print('predicted',end='')
        #print(predicted)       

    f1.close()
    #iteration loop end

#feature number loop end