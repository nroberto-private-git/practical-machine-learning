---
title: "Practical Machine Learning Week 4 Project"
author: "Nuno R"
date: "March 16, 2020"
output:
  pdf_document: default
  html_document: default
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE, warning=FALSE, message=FALSE)
```

## Executive Summary
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. 
They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. 

More information is available from the website here: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har 

## Data
The training data for this project are available here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har

## Load required packages for Initialization and Analysis
```{r}
# this is only needed for tests in the local env
setwd("C:/Users/nroberto/practical_machine_learning_wk4_proj")

# not needed when running from R STudio
library(knitr)

# Clean up env, recommended by:
# https://community.rstudio.com/t/how-to-clear-the-r-environment/14303
rm(list=ls())

# More information on caret can be found here: 
# http://topepo.github.io/caret/index.htmllibrary("caret")
library(caret)
# library(e1071) # Error: package e1071 is required

# More information on rpart can be found here: 
# https://www.rdocumentation.org/packages/rpart/versions/4.1-15/topics/rpart 
library(rpart)
library(rpart.plot)

# GUI for Data Science
# https://www.rdocumentation.org/packages/rattle/versions/5.3.0
library(rattle)

# Classification and Regression with Random Forest
# https://www.rdocumentation.org/packages/randomForest/versions/4.6-14/topics/randomForest
library(randomForest)

# Package for really cool correlation matrixes
# http://www.sthda.com/english/wiki/visualize-correlation-matrix-using-correlogram
library(corrplot)

# Generalized Boosted Regression Modeling (GBM)
# https://www.rdocumentation.org/packages/gbm/versions/2.1.5/topics/gbm
library(gbm)
```

## Load Data for Analysis
```{r}
seedValue = 202019
set.seed(seedValue)

precisionPoints = 5

# set the URL for the download
pmlTrainingData <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
pmlTestingData  <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

# download the datasets
trainingData <- read.csv(url(pmlTrainingData))
testingData  <- read.csv(url(pmlTestingData))
```

## Prepare Data for Analysis
Create a partition with the training dataset 
```{r}
# More information can be found here:
# https://www.rdocumentation.org/packages/caret/versions/6.0-85/topics/createDataPartition
percentageForTraining = 0.60
trainingPartition  <- createDataPartition(trainingData$classe, 
                                          p=percentageForTraining, 
                                          list=FALSE)

trainingSet <- trainingData[trainingPartition, ] # eg, 60% for training
testingSet  <- trainingData[-trainingPartition, ] # eg, 40% for testing
```

Let's examine the Training Set
```{r}
dim(trainingSet)
```

Let's examine the Testing Set
```{r}
dim(testingSet)
```

NOTE: There are a lot of columns with no variance or close to 0 variance. They are not meaningful for the analysis and we need to remove them.
This will further trim the Training and Testing sets for a more streamlined analysis
```{r}
# Note: additional information can be found here borrowed from the caret package:
# https://www.rdocumentation.org/packages/mixOmics/versions/6.3.2/topics/nearZeroVar
columnsZeroVar <- nearZeroVar(trainingSet)
trainingSet <- trainingSet[, -columnsZeroVar]
testingSet  <- testingSet[, -columnsZeroVar]
```

Let's examine the Training set after trimming columns with zero variance
```{r}
dim(trainingSet)
```

Let's examine the Testing set after trimming columns with zero variance
```{r}
dim(testingSet)
```

NOTE: As can be observed there are a lot of missing data points, including NA, DIV/0 and empty cells. We need to clear this data before further analysis.
```{r}
# find columns with a lot of NAs
percentageOfNAacceptable = 0.97
foundNAs <- sapply(trainingSet, 
                   function(x) 
                     mean(is.na(x))) > 
                      percentageOfNAacceptable

#... and now remove them since they are not relevant for the analysis
trainingSet <- trainingSet[, foundNAs==FALSE]
testingSet  <- testingSet[, foundNAs==FALSE]
```

The Training and Testing sets are now cleaned up.
Let's take a look at the Training set again
```{r}
dim(trainingSet)
```

Let's take a look at the Training set again
```{r}
dim(testingSet)
```

The data sets are are still showing columns that don't have to be used for analysis, so we can remove them before further analyzing the data
```{r}
removeColumns = 5
trainingSet <- trainingSet[, -(1:removeColumns)]
testingSet  <- testingSet[, -(1:removeColumns)]
```

```{r}
dim(trainingSet)
```

```{r}
dim(testingSet)
```

Let's take a look at the correlation matrix for the relevant features
```{r message=TRUE, warning=TRUE, paged.print=TRUE}
pickColumnsToRemove = 54
correlationMatrix <- cor(trainingSet[, -pickColumnsToRemove])

# now plot it
# More info: https://www.rdocumentation.org/packages/corrplot/versions/0.2-0/topics/corrplot
corrplot(correlationMatrix, 
         order  = "FPC",           
         method = "shade", 
         shade.method = "all",
         lwd.shade = 1,
         type   = "upper", 
         tl.cex = 0.4, 
         tl.col = rgb(0,0,1))
```


```{r}
# ########################################################################
```

## ML Algorithm 1: Random Forest
```{r}
set.seed(seedValue)

# Really good information and exmaples for TrainControl
# This function prepares and controls the parameters for the function train
# https://topepo.github.io/caret/model-training-and-tuning.html
# 
# official docs: https://www.rdocumentation.org/packages/caret/versions/4.47/topics/trainControl

randomForestTrainControl <- trainControl(method="cv", 
                                         number=5, 
                                         verboseIter=TRUE,
                                         returnData = TRUE
                                        )
```

ML Algorithm 1: Now let's train the model using the Training data set
```{r}
# More info here: https://www.rdocumentation.org/packages/caret/versions/6.0-85/topics/train
trainedRandomForestModel <- train(classe ~ ., 
                                  data=trainingSet, 
                                  method="rf",
                                  metric = "Accuracy",
                                  trControl=randomForestTrainControl)
```

ML Algorithm 1: The final model fits the data after training as below:
```{r}
trainedRandomForestModel$finalModel
```

ML Algorithm 1: Let's predict using the data set we split earlier for Testing
```{r}
# More info here: 
# https://www.rdocumentation.org/packages/raster/versions/3.0-12/topics/predict
randomForestPrediction <- predict(trainedRandomForestModel, 
                                  newdata=testingSet)
```

ML Algorithm 1: Let's create a confusion matrix for the Random Forest prediction done above and using the Testing data set
```{r}
# More info here: https://www.rdocumentation.org/packages/caret/versions/3.45/topics/confusionMatrix
randomForestConfusionMatrix <- confusionMatrix(randomForestPrediction, 
                                                testingSet$classe)
```

ML Algorithm 1: The confusion matrix based for the Random Forest algorithm using the Testing data set looks like:
```{r}
randomForestConfusionMatrix
```

### ML Algorithm 1: The Confusion Matrix accuracy plot
```{r}
plot(randomForestConfusionMatrix$table, 
     col = randomForestConfusionMatrix$byClass, 
     main = paste("ML Algorithm 1: Random Forest (Accuracy) = ",
                  round(randomForestConfusionMatrix$overall['Accuracy'], 
                        precisionPoints)))
```


```{r}
# ########################################################################
```

## ML Algorithm 2: Decision Trees
```{r}
set.seed(seedValue)

# Recursive partitioning for decision tress
# More info here: https://www.rdocumentation.org/packages/rpart/versions/4.1-15/topics/rpart
decisionTreeTrained <- rpart(classe ~ ., 
                              data=trainingSet, 
                              method="class")
```

ML Algorithm 2: Let's plot the decision tree as partitioned above
```{r}
# More info here: https://www.rdocumentation.org/packages/rattle/versions/5.3.0/topics/fancyRpartPlot
# site: https://rattle.togaware.com/
fancyRpartPlot(decisionTreeTrained)
```

ML Algorithm 2: Let's predict using the data set we split earlier for Testing
```{r}
# More info here: 
# https://www.rdocumentation.org/packages/raster/versions/3.0-12/topics/predict
decisionTreePrediction <- predict(decisionTreeTrained, 
                                  newdata=testingSet, 
                                  type="class")
```

ML Algorithm 2: Let's create a confusion matrix for the Decision Tree prediction done above and using the Testing data set
```{r}
decisionTreeConfusionMatrix <- confusionMatrix(decisionTreePrediction, 
                                                testingSet$classe)
```

ML Algorithm 2: The confusion matrix for the Decision Tree algorithm using the Testing data set looks like:
```{r}
decisionTreeConfusionMatrix
```

### ML Algorithm 2: The Confusion Matrix accuracy plot
```{r}
plot(decisionTreeConfusionMatrix$table, 
     col = decisionTreeConfusionMatrix$byClass, 
     main = paste("ML Algorithm 2: Decision Trees (Accuracy) = ",
                  round(decisionTreeConfusionMatrix$overall['Accuracy'], 
                        precisionPoints)))
```

```{r}
# ########################################################################
```

## ML Algorithm 3: Generalized Boosted Model
```{r}
set.seed(seedValue)
# Now let's train the model using the Training data set
# More info here: https://www.rdocumentation.org/packages/caret/versions/6.0-85/topics/train
gbmControl <- trainControl(method = "repeatedcv", 
                           number = 5, 
                           repeats = 1,
                           verboseIter = TRUE,
                           returnData = TRUE)
```

ML Algorithm 3: The final model fits the data after training as below:
```{r}
gbmTrained  <- train(classe ~ ., 
                      data=trainingSet, 
                      method = "gbm",
                      trControl = gbmControl, 
                      verbose = TRUE)
```

ML Algorithm 3: The final model fits the data after training as below:
```{r}
gbmTrained$finalModel
```


ML Algorithm 3: Let's predict using the data set we split earlier for Testing
```{r}
# More info here: 
# https://www.rdocumentation.org/packages/raster/versions/3.0-12/topics/predict
gbmPrediction <- predict(gbmTrained, 
                         newdata=testingSet)
```

ML Algorithm 3: Let's create a confusion matrix for the Generalized Boosted Model prediction done above and using the Testing data set
```{r}
# More info here: https://www.rdocumentation.org/packages/caret/versions/3.45/topics/confusionMatrix
gbmConfusionMatrix <- confusionMatrix(gbmPrediction, 
                                      testingSet$classe)
```

ML Algorithm 3: The confusion matrix based for the Generalized Boosted Model algorithm using the Testing data set looks like:
```{r}
gbmConfusionMatrix
```

### ML Algorithm 3: The Confusion Matrix accuracy plot
```{r}
plot(gbmConfusionMatrix$table, 
     col = gbmConfusionMatrix$byClass, 
     main = paste("ML Algorithm 3: Generalized Boosted Model (Accuracy) = ", 
                  round(gbmConfusionMatrix$overall['Accuracy'], 
                        precisionPoints)))
```

```{r}
# ########################################################################
```

## Final Prediction using the chosen algorithm 
```{r}
qzPrediction <- predict(trainedRandomForestModel, 
                        newdata=testingData)

qzPrediction
```

## Final Thoughts and Conclusion

The results show that the Random Forest algorithm outperforms the Decision Tree in terms of accuracy. 
We are getting 99% in sample accuracy, followed by the GBM at 98%, whereas the Decision Tree algorithm is only 75% accurate.
