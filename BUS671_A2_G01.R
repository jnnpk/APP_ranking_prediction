rm(list=ls())

###import packages

#install.packages("rpart") 
#install.packages("caret") 
#install.packages("pROC") 
#install.packages("partykit") 
#install.packages("C50")  
#install.packages("e1071") 
#install.packages("pillar")
#install.packages("rlang")
#install.packages("ggplot2")
#install.packages("rpart.plot")
#install.packages("adabag")

library(pillar)
library(dplyr)
library(rpart) 
library(pROC)
library(caret)
library(partykit)
library(C50)
library(e1071)
library(randomForest)
library(ipred)
library(adabag)
library(rpart.plot)
library(tidyverse)
library(lubridate)

### import and merge datasets ###
setwd("/Users/parkh/Downloads/Data") # datasets should be placed in the same directory  
filenames <- list.files(path=getwd())  
numfiles <- length(filenames)
MergedData <- do.call(rbind, lapply(filenames, read.csv))

# Basic EDA 
head(MergedData)
nrow(MergedData)
str(MergedData)
names(MergedData)
summary(MergedData[,c("AppId", "Rank","Name","ReleaseDate", "Category", "Price", "Seller", "Developer", "Screenshot","Size","StarsAllVersions", "RatingsAllVersions","StarsCurrentVersion","RatingsCurrentVersion","Version","UpdatedDate","TopInAppPurchases")])


## Feature selection & define target variable ##
TopHundred <- ifelse(MergedData$Rank <= 100, "Yes", "No")
MergedData <- cbind(MergedData,data.frame(TopHundred)); 
MergedData<- MergedData[, c("Price", "Screenshot","Size","StarsCurrentVersion","RatingsCurrentVersion","UpdatedDate","TopInAppPurchases", "TopHundred")]
head(MergedData)


### Preprocessing ###


# type conversion
MergedData$TopHundred <- as.factor(MergedData$TopHundred)
MergedData$Price <- as.integer(MergedData$Price)
MergedData$Size <- as.integer(MergedData$Size)
MergedData$Screenshot <- as.integer(MergedData$Screenshot)

## handling missing value ##
# detect total missing values
colSums(is.na(MergedData))

# TopInAppPurchases, delete data
MergedData[is.na(MergedData$TopInAppPurchases),]
MergedData <- MergedData[!is.na(MergedData$TopInAppPurchases),]
# Price, mean imputation
MergedData[is.na(MergedData$Price),]
MergedData$Price[is.na(MergedData$Price)== 'TRUE'] <- mean(MergedData$Price, na.rm = TRUE)
# Size, delete data
MergedData[is.na(MergedData$Size),]
MergedData <- MergedData[!is.na(MergedData$Size),]
# Screenshot, mean imputation
MergedData[is.na(MergedData$Screenshot),]
MergedData$Screenshot[is.na(MergedData$Screenshot)== 'TRUE'] <- mean(MergedData$Screenshot, na.rm = TRUE)
# starsCurrentVersion, delete data
MergedData[is.na(MergedData$StarsCurrentVersion),]
MergedData <- MergedData[!is.na(MergedData$StarsCurrentVersion),]
# RatingsCurrentVersion, delete data
MergedData[is.na(MergedData$RatingsCurrentVersion),]
MergedData <- MergedData[!is.na(MergedData$RatingsCurrentVersion),]

colSums(is.na(MergedData))

# reset cols
rownames(MergedData)=NULL
MergedData <- na.omit(MergedData)

## Date data preprocessing ##
# sort date data for formmating
MergedData$UpdatedDate <- sort(MergedData$UpdatedDate)

# format to datetime data by lubridate package
date1 <- MergedData[1:351, ]
date1$UpdatedDate <- mdy(date1$UpdatedDate)
date2 <- MergedData[352:834, ]
date2$UpdatedDate <- dmy(date2$UpdatedDate)
date3 <- MergedData[835:2705, ]
date3$UpdatedDate <- ymd(date3$UpdatedDate)
date4 <- MergedData[2706:2783, ]
date4$UpdatedDate <- dmy(date4$UpdatedDate)

# merging datasets
MergedData <- do.call("rbind", list(date1,date2,date3,date4))

# create derived variable - updated days from the collection day
NewlyUpdated <- as.Date("2013-01-01") - MergedData$UpdatedDate
MergedData <- cbind(MergedData,data.frame(NewlyUpdated))
MergedData$NewlyUpdated <- as.numeric(MergedData$NewlyUpdated)
MergedData <- subset(MergedData, select=-UpdatedDate)

## save
write_csv(MergedData,"BUS671_A2_G01.csv")

## descriptive stat ##
str(MergedData)
nrow(MergedData)
summary(MergedData)
colSums(is.na(MergedData))

round(sd(MergedData$Price),3)
round(sd(MergedData$Screenshot),3)
round(sd(MergedData$Size),3)
round(sd(MergedData$StarsCurrentVersion),3)
round(sd(MergedData$RatingsCurrentVersion),3)
round(sd(MergedData$TopInAppPurchases),3)
round(sd(MergedData$NewlyUpdated),3)

## correlation - pearson, encode dichotomous variable into dummy variable ##
TH_corr <- ifelse(MergedData$TopHundred == "Yes",1,0)
MergedData2 <- cbind(MergedData,TH_corr)
head(MergedData2)

# correlation table
cor(MergedData2[-7])


### splitting training and test sets ###
table(MergedData$TopHundred)

num_obs <- nrow(MergedData)
train_size <- num_obs * 0.8 
set.seed(2022)
train_sample <- sample(num_obs, train_size)

App_Train <- MergedData[train_sample, ]
App_Test  <- MergedData[-train_sample, ]

nrow(App_Train); nrow(App_Test)

table(App_Train$TopHundred)
table(App_Test$TopHundred)

# check the proportion of class variable
prop.table(table(App_Train$TopHundred))
prop.table(table(App_Test$TopHundred))


### Classificaion: Decision Trees ###
# Train the model with Training Set
App_model <- rpart(TopHundred ~ ., data = App_Train, method="class")


# Plot Tree 
plot(as.party(App_model))

# Display simple facts about the tree
App_model

# Display detailed information about the tree
summary(App_model)

# Create a vector of predictions on test data
App_pred <- predict(App_model, App_Test, type="class")
mean(App_pred == App_Test$TopHundred)

# Confusion Matrix 
confusionMatrix(App_pred, App_Test$TopHundred, positive = "Yes")

# ROC Curve
pred_value <- ifelse(App_pred == "Yes",1,0)
actual_value <- ifelse(App_Test$TopHundred == "Yes",1,0)

App_roc <- roc(pred_value, actual_value)
App_roc
plot(App_roc)

## Pruning ##
# 10-fold validation repeated Five times
cv_control <- trainControl(method='repeatedcv', number=10, repeats=5, 
                           summaryFunction = twoClassSummary, classProbs = TRUE)


Caret_Tree <- train(TopHundred  ~ ., data = App_Train, method ="rpart", 
                    trControl=cv_control, metric = "ROC")

Caret_Tree
plot(Caret_Tree)
plot(as.party(Caret_Tree$finalModel))

# Prediction 
CaretTree_predict <- predict(Caret_Tree, App_Test)
confusionMatrix(CaretTree_predict, App_Test$TopHundred, positive = "Yes")


### Boosting ###
Tree_boost10 <- C5.0(App_Train[-7],App_Train$TopHundred, trials = 10 )

Tree_boost10
summary(Tree_boost10)

# Prediction 
Tree_boost10_pred <- predict(Tree_boost10,App_Test)
confusionMatrix(Tree_boost10_pred, App_Test$TopHundred, positive = "Yes")


### Bagging ###
Tree_bagging <- bagging(TopHundred ~.,data=App_Train,mfinal=5, trControl = cv_control, control=rpart.control(maxdepth=5, minsplit=5))
Tree_bagging$trees
rpart.plot(Tree_bagging$trees[[1]]) #plotting first tree
Tree_bagging$importance #Feature importance

# Prediction
Tree_bagging_pred <- predict(Tree_bagging, App_Test)$class
confusionMatrix(as.factor(Tree_bagging_pred), App_Test$TopHundred, positive = "Yes")


### RandomForest ###
rf<- randomForest(TopHundred ~ ., data=App_Train, ntree = 800)
rf$importance#Feature importance

# Prediction
rf_pred <- predict(rf, App_Test)
rf_pred
confusionMatrix(rf_pred, App_Test$TopHundred, positive = "Yes")
plot(rf) #plot the error rate


### Cost Matrix ###
# Specify the dimensions #
# Since the predicted and actual values can both take two values, yes or no,#
# we have to describe a 2 x2 matrix, using a list of two vector, each with two values.#
matrix_dimensions <- list(c("No", "Yes"), c("No", "Yes"))
names(matrix_dimensions) <- c("Predicted", "Actual")
matrix_dimensions

## Construct a error cost matrix ##
# weight 4:1
error_cost <- matrix(c(0,1,2,0), nrow=2, ncol=2, dimnames = matrix_dimensions)
error_cost


## Construct a tree with the cost matrix ##
ClassData_cost <- C5.0(App_Train[-7],App_Train$TopHundred, costs = error_cost)
summary(App_Test$TopHundred)

# Prediction
ClassData_cost_pred <- predict(ClassData_cost, App_Test)
confusionMatrix(ClassData_cost_pred, App_Test$TopHundred, positive = "Yes")

