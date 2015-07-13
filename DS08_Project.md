# | Classifying The Quality Of Activity
| With Machine Learning

July 20, 2015  



## Executive Summary:

The goal of this project is to develop a model capable 
of predicting the quality of weight lifting activity 
from data collected from accelerometers on the belt, 
forearm, arm, and dumbell of six participants. The 
participants were asked to perform barbell lifts 
correctly and incorrectly in five different ways, 
with "A" being the correct way and "B" - "E" being 
the incorrect ways.  

For more information about the original study, 
please see: http://groupware.les.inf.puc-rio.br/har  

## Data Cleaning & Exploratory Analysis:





The master dataset is comprised of 19622 observations across
160 variables. The distribution of the classe variable
is shown in Figure 1 in the Appendix. Note that the data is 
not uniformly distributed, especially with regards to the 
"A" results. This skewness indicates that cross-validation
is a good strategy given the possible dataset sizes for 
training and testing.

The first step is to divide the master dataset into a training
dataset and a testing dataset. The division is such that the
training set contains 70% of the observations, and the testing 
set contains the remaining 30% of the observations. (The testing 
set is then set aside for the final check of the model.)


```r
# Set The Seed For Reproducibility:
set.seed(12345)

# Split The Master Dataset Into Training & Testing:
inMaster <- createDataPartition(y = masterset$classe, 
                                p = 0.70, list = FALSE)
training <- masterset[ inMaster,]
testing  <- masterset[-inMaster,]
```



Examining the summary output (not shown) for the training set 
reveals a number of variables that are either used for accounting 
(e.g., user name, timestamps, windows, etc.) or else contain high
percentages of missing or empty values. Removing these variables
results in the following:


```r
cleanCols <- c("roll_belt","pitch_belt",
               "yaw_belt","total_accel_belt",
               "gyros_belt_x","gyros_belt_y","gyros_belt_z",
               "accel_belt_x","accel_belt_y","accel_belt_z",
               "magnet_belt_x","magnet_belt_y","magnet_belt_z",
               "roll_arm","pitch_arm",
               "yaw_arm","total_accel_arm",
               "gyros_arm_x","gyros_arm_y","gyros_arm_z",
               "accel_arm_x","accel_arm_y","accel_arm_z",
               "magnet_arm_x","magnet_arm_y","magnet_arm_z",
               "roll_dumbbell","pitch_dumbbell",
               "yaw_dumbbell","total_accel_dumbbell",
               "gyros_dumbbell_x","gyros_dumbbell_y","gyros_dumbbell_z",
               "accel_dumbbell_x","accel_dumbbell_y","accel_dumbbell_z",
               "magnet_dumbbell_x","magnet_dumbbell_y","magnet_dumbbell_z",
               "roll_forearm","pitch_forearm",
               "yaw_forearm","total_accel_forearm",
               "gyros_forearm_x","gyros_forearm_y","gyros_forearm_z",
               "accel_forearm_x","accel_forearm_y","accel_forearm_z",
               "magnet_forearm_x","magnet_forearm_y","magnet_forearm_z",
               "classe")
```

Note that this list includes the classe variable, which is the one
the model will attempt to predict. 

The training and testing sets can now be subsetted to just these 
53 variables.


```r
cln_train <- training[cleanCols]
cln_test  <- testing[cleanCols]
```

## Model Selection:

The goal of the model is to predict the correct categorization 
for the quality of the weight lifting activity (classe variable). 
Tree-based models in general and random forest models in particular 
are especially appropriate for this type of prediction. After much
experimentation, the final model that was chosen for this project is
a random forest with cross validation resampling (three iterations). 
This combination in conjunction with a training set size of 70% of 
the observations provides a good balance between a reasonable run 
time to train the model and the accuracy of the predictions 
(as will be seen below).  


```r
rfMod <- train(classe ~ ., method = "rf",
               data = cln_train,
               trControl = trainControl(method = "cv"), number = 3)
rfMod
```

```
## Random Forest 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## 
## Summary of sample sizes: 12362, 12364, 12363, 12364, 12365, 12363, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9925745  0.9906069  0.002373141  0.003001921
##   27    0.9911923  0.9888581  0.001981222  0.002506984
##   52    0.9868249  0.9833314  0.004582261  0.005800369
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```



Notice that the cross-validated resampling was done with ten folds, 
and that the accuracy of the optimal model against the training set
was 0.9926.

The results of running this model against the testing set are as
follows:


```r
# Predict Against The Testing Set:
rfPrdTest <- predict(rfMod, cln_test)
# Check The Results, Especially The Accuracy:
confusionMatrix(cln_test$classe, rfPrdTest)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    1    0    0    0
##          B   11 1124    4    0    0
##          C    0   17 1006    3    0
##          D    0    0   23  941    0
##          E    0    0    0    3 1079
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9895          
##                  95% CI : (0.9865, 0.9919)
##     No Information Rate : 0.2862          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9867          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9935   0.9842   0.9739   0.9937   1.0000
## Specificity            0.9998   0.9968   0.9959   0.9953   0.9994
## Pos Pred Value         0.9994   0.9868   0.9805   0.9761   0.9972
## Neg Pred Value         0.9974   0.9962   0.9944   0.9988   1.0000
## Prevalence             0.2862   0.1941   0.1755   0.1609   0.1833
## Detection Rate         0.2843   0.1910   0.1709   0.1599   0.1833
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9966   0.9905   0.9849   0.9945   0.9997
```



The expected out of sample error rate is 0.9895, and 
this translates into a misclassification rate of 0.0105.

Notice also that the accuracy is lower on the testing set 
than it is on the training set, which is to be expected.

These results are shown graphically in Figure 2 in the Appendix.

## Conclusion:

The random forest model used in this project was able to predict
all twenty test cases correctly for the second part of the project. 
The prediction is not however perfect. For the twenty test cases,
the model is expected to get 19.79 correct on average, and 
repeated runs of the model against the validation set for the 
second part of the project did indeed prove this out. It is also 
interesting to note in Figure 2 that when the model did miss the 
classification, it was only off by one. For example, if the actual 
value was "C", when the model missed, it missed by "B" or "D" 
rather than "A" or "E".

----------
## Appendix:

### Figure 1: Distribution Of classe Variable In Master Dataset

![](DS08_Project_files/figure-html/unnamed-chunk-12-1.png) 

### Figure 2: Scatterplot Of Aggregated Results Of Model Run

![](DS08_Project_files/figure-html/unnamed-chunk-13-1.png) 
