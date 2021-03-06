
rm(list=ls())



#install.packages("tidyverse")
#install.packages("broom")
#install.packages("glmnet")
library(tidyverse)
library(broom)
library(glmnet)

#install.packages("ISLR")
library(ISLR)

attach(College)

head(College)

######################################################################

# Compute RMSE and R Square from true and predicted values
eval_results <- function(true, predicted, df) {
  SSE <- sum((predicted - true)^2)
  SST <- sum((true - mean(true))^2)
  R_square <- 1 - (SSE / SST)
  RMSE = sqrt(SSE/nrow(df))
  
  
  # Model performance metrics
  data.frame(
    RMSE = RMSE,
    Rsquare = R_square
  )
  
}


########################################################

############## Partitioning training and test data ###############

College$Private = factor(ifelse(College$Private == "Yes",1,0))
College$Private

#install.packages("caret")
library(caret)
set.seed(3456)
trainIndex <- createDataPartition(College$Private, p = .8,
                                  list = FALSE,
                                  times = 1)
Train <- College[ trainIndex,]
Test <- College[-trainIndex,]


x_train = data.matrix(Train[,-1])
y_train = Train$Private

x_test= data.matrix(Test[,-1])
y_test = Test$Private





###########  Ridge Regression   ####################


lambdas <- 10^seq(3, -2, by = -.1)

fit <- cv.glmnet(x_train,y_train, alpha = 0, lambda = lambdas,family = "binomial",type.measure = "mse")

summary(fit)
plot(fit)
plot(fit,xvar="lambda")
fit$lambda.min
fit$lambda.1se
coef(fit)



  train_model <- predict(fit, newx = x_train, type = "response")
head(train_model)
summary(train_model)
predtrain<-factor(ifelse(train_model>0.5,1,0))
predtrain


test_model <- predict(fit, newx = x_test, type = "response")
head(test_model)
summary(test_model)
predtest <- factor(ifelse(test_model>0.5,1,0))
predtest


# evaluation on train data
eval_results(as.numeric(y_train), as.numeric(train_model), Train)
confusionMatrix(reference= y_train, data= predtrain)


# evaluation on test data
eval_results(as.numeric(y_test),as.numeric(test_model), Test)

confusionMatrix(reference= y_test, data=predtest )



################   Lasso Regression    ###########################


lambdas <- 10^seq(3, -2, by = -.1)

fit <- cv.glmnet(x_train,y_train, alpha = 1, lambda = lambdas, family = "binomial",type.measure="mse")
summary(fit)
plot(fit)
fit$lambda.min
fit$lambda.1se
coef(fit)


train_model <- predict(fit, newx = x_train, type = "response")
head(train_model)
summary(train_model)
predtrain<-factor(ifelse(train_model>0.5,1,0))
predtrain



test_model <- predict(fit, newx = x_test, type = "response")
head(test_model)
summary(test_model)
predtest <- factor(ifelse(test_model>0.5,1,0))
predtest


# evaluation on train data
eval_results(as.numeric(y_train), as.numeric(train_model), Train)
confusionMatrix(reference= y_train, data=predtrain )

# evaluation on test data
eval_results(as.numeric(y_test),as.numeric(test_model), Test)
confusionMatrix(reference= y_test, data=predtest )







