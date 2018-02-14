###
## imputation

#setwd("K:/Dropbox/Machine Learning Paper")
#setwd("B:/Dropbox/Machine Learning Paper")
#setwd("/Backup//Dropbox/Machine Learning Paper")

#("sandwich", "xtable", "lme4", "effects", "Matching", "rgenoud", "car", "cem", "arm", "lattice", "plm", "stargazer", "aod", "ggplot2")

library(foreign)
library(sandwich)
library(xtable)
library(lme4)
library(effects)
library(Matching)
library(rgenoud)
library(car)
library(cem)
library(arm)
library(lattice)
library(plm)
library(stargazer)
library(aod)
library(ggplot2)
library(compactr)
#library(MASS)
library(stats)
library(xtable)
library(dplyr)
library(tidyr)
library(data.table)
library(corrplot)
library(foreach)
library(ggplot2)
library(grid)
library(gridExtra)
library(DataCombine)
library(caret)
library(doParallel)
library(beepr)
#library(pls)
#library(elasticnet)
#library(lars)
#library(glmnet)
#library(rpart)
#library(randomForest)
#library(mlr)
options(scipen=7)
options(digits=3)

# for phil's computer
setwd("F:/Dropbox/Machine Learning Paper/")
setwd("/Backup/Dropbox/Machine Learning Paper")

# Load in data
load("2017-10-12 final.RData")
merged.dat<-data.table(final.dat.2)

# remove lagged and differenced data #
# create new levels-only data set    #
levels.dat <- merged.dat[,c(1:64)] # lags and diff 64-107 #

# Select which data set
dat<-dplyr::select(levels.dat, -memberid, -X,-caseid, -district, -birth)


### Glance at the outcome variable ###
# Look at the outcome variable by year
dat.year<-group_by(dat, year)

# number of observations per year
tally(dat.year)

# mean income per year
mean.year<-dplyr::summarize(dat.year, 
                            mean=mean(hisinc),
                            sd=sd(hisinc))
mean.year

# plot all income
ggplot(dat.year, aes(x=year, y=hisinc))+
        geom_point(alpha=0.5)+
        geom_line(data=mean.year, aes(x=year, y=mean), color="blue")+
        labs(x="Year")+
        labs(y="Income (IHS)")+
        theme_minimal()


### Summary statistics for predictors
tmp <- do.call(data.frame, 
               list(mean = apply(dat, 2, mean),
                    sd = apply(dat, 2, sd),
                    median = apply(dat, 2, median),
                    min = apply(dat, 2, min),
                    max = apply(dat, 2, max),
                    n = apply(dat, 2, length)))

### KEVIN LOOK AT THIS AND IDENTIFY PROBLEMATIC VARIABLES
tmp


# RMSE function
error<-function(pred.Y, Y) {
        sqrt(sum((pred.Y-Y)^2)/nrow(Y))
}


# Test Data Set
# We are predicting the years 2013-2014 across all of our modeling approches
testDat<-as.data.table(subset(dat, year>=2012))
# Do not touch until the very end!


### Average across imputations ###
##### Naive Training ######

#dat<-dat[[h]]

# We split the training data into five separate folds irrespective of time
# Conduct nested cross validation to identify the model which performs best out of sample
set.seed(1999)
naiveDat<-as.data.table(subset(dat, year<2012))
naiveIter<-createFolds(naiveDat$hisinc, k=5)
naiveFolds<-list(c(naiveIter$Fold1),  c(naiveIter$Fold2), c(naiveIter$Fold3), c(naiveIter$Fold4), c(naiveIter$Fold5))

# iterate over these when training
# We will select tuning parameters via cross validation on the training set (ie, using part of the training set as a validation set)
# We will select the tuning parameters which minimize error on the training set (though pay attention to the 1se rule), and then, having selected 
# the tuning parameters, we will estimate the model performance on a validation set, which will be one of the folds that we leave out

# Split training data into five folds for nested cross validation
set.seed(1999)
folds<-createFolds(naiveDat$hisinc, k=5)
iter<-list(c(folds$Fold1),  c(folds$Fold2), c(folds$Fold3), c(folds$Fold4), c(folds$Fold5))

### Set up tuning control
ctrl <- trainControl(method = "cv",
                     n=5,
                     # selectionFunction="oneSE", 
                     verboseIter = TRUE,
                     allowParallel = FALSE,
                     savePredictions="final")

ctrlJustRun <- ctrl
ctrlJustRun$method <- "none"

ctrlParallel <- ctrl
ctrlParallel$verboseIter <- FALSE
ctrlParallel$allowParallel <- TRUE

#### Walk Forward Training #####

# Split the data into five specific intervals of time
# Train model on interval 1, predict 2; train model on interval 1 and 2, predict 3, ...
# We're using intervals coinciding with election years
# This gives us a window of roughly four years for each training

time<-sort(unique(naiveDat$year))

# train starting with 1995
train_breaks<-seq(1995, 2012, 2)
# test on a two year window
test_breaks<-seq(1997, 2012, 2)

sliceTrain<-foreach(i=1:(length(train_breaks))) %do%{
        years<-train_breaks[1]:(train_breaks[i]+1)
}

sliceTest<-foreach(i=1:(length(test_breaks))) %do%{
        years<-(test_breaks[i]):(test_breaks[i]+1)
}


# to select dataset that corresponds to these years, use subset
# steadily increasing window of train data

walkOut<-foreach(i=1:length(sliceTest)) %do% {
        
        # expanding window of train data
        walkTrain<-subset(naiveDat, 
                          year>=min(sliceTrain[[i]]) & year<=max(sliceTrain[[i]]))
        
        # two year window of test data
        walkTest<-subset(naiveDat,
                         year>=min(sliceTest[[i]]) & year<=max(sliceTest[[i]]))
        
        # train models
        set.seed(1999)
        walk_null<-suppressWarnings(train(hisinc ~ 1, 
                                          data=walkTrain,
                                          method="lm",
                                          trControl=ctrlParallel))
        walk_null$method<-"Null"
        
        # ols
        set.seed(1999)
        walk_ols<-suppressWarnings(train(hisinc ~ ., 
                                         data=walkTrain,
                                         method="lm",
                                         trControl=ctrlParallel))
        
        # pls
        set.seed(1999)
        walk_pls<-suppressWarnings(train(hisinc ~ ., 
                                         data=walkTrain,
                                         method="pls",
                                         tuneLength=5,
                                         trControl=ctrlParallel,
                                         preProcess=c("center", "scale")))
        
        # cart
        set.seed(1999)
        walk_cart<-suppressWarnings(train(hisinc ~ .,
                                          data=walkTrain,
                                          tuneLength=5,
                                          method="rpart2",
                                          trControl=ctrlParallel))
        
        # KNN
        set.seed(1999)
        walk_knn<-suppressWarnings(train(hisinc ~ .,
                                         data=walkTrain,
                                         tuneLength=5,
                                         method="knn",
                                         trControl=ctrlParallel,
                                         preProcess=c("center", "scale")))
        # MARS
        set.seed(1999)
        walk_mars<-train(hisinc ~ ., 
                        data=walkTrain,
                        method="earth",
                        tuneLength=5,
                        trControl=ctrlParallel,
                        preProcess=c("center", "scale"))
        
        # Cubist
        set.seed(1999)
        walk_cub<-train(hisinc ~ ., 
                        data=walkTrain,
                        method="cubist",
                        tuneLength=5,
                        trControl=ctrlParallel)
        
        # Boosted Trees
        set.seed(1999)
        walk_gbm<-train(hisinc ~ ., 
                        data=walkTrain,
                        method="gbm",
                        tuneLength=5,
                        trControl=ctrlParallel,
                        verbose=F)
        
        # random forest
        set.seed(1999)
        walk_rf<-train(hisinc ~ ., 
                       data=walkTrain,
                       method="ranger",
                       tuneLength=5,
                       trControl=ctrlParallel)
        
        # conditional inference trees
        set.seed(1999)
        walk_rf<-train(hisinc ~ ., 
                       data=walkTrain,
                       method="ctree",
                       tuneLength=5,
                       trControl=ctrlParallel)
        
        # svm
        set.seed(1999)
        walk_svm<-train(hisinc ~ ., 
                       data=walkTrain,
                       method="svmRadial",
                       tuneLength=5,
                       trControl=ctrlParallel,
                       preProcess=c("center", "scale"))
        
        models_walk<-lapply(ls(pattern="walk_"), get)
        
        # training performance
        trainPred<-as.tbl(foreach(i=1:length(models_walk), .combine=cbind.data.frame) %do% {
                
                bar <- as.tbl(models_walk[[i]]$pred) %>% 
                        arrange(rowIndex) %>%
                        dplyr::select(pred)
                bar
                
                names(bar)<-models_walk[[i]]$method
                bar
                
        })
        
        trainObs<- as.matrix(models_walk[[1]]$pred %>% 
                        arrange(rowIndex) %>%
                        dplyr::select(obs))
        
        # rmse on the test set
        trainRMSE<-as.tbl(data.frame(Model=names(trainPred),
                                    RMSE=apply(as.matrix(trainPred), 2, error, trainObs),
                                    Time=min(walkTest$year)))
        
        # grab predictions for test
        evalPred<-as.tbl(
                foreach(j=1:length(models_walk), .combine = cbind.data.frame) %do% {
                        pred<-data.frame(predict.train(models_walk[[j]], newdata=walkTest))
                        colnames(pred)<-models_walk[[j]]$method
                        pred
                })
        
        # rmse on the test set
        evalRMSE<-as.tbl(data.frame(Model=names(evalPred),
                                    RMSE=apply(as.matrix(evalPred), 2, error, as.matrix(walkTest$hisinc)),
                                    Time=min(walkTest$year)))
        
        out<-list("trainPred"=trainPred,
                  "trainRMSE"=trainRMSE,
                  "evalPred"=evalPred,
                  "evalRMSE"=evalRMSE)
        
        out
        
}


# extract
trainPred<-do.call(rbind, lapply(walkOut, '[[', 'trainPred'))
evalPred<-do.call(rbind, lapply(walkOut, '[[', 'evalPred'))

trainRMSE<-do.call(rbind, lapply(walkOut, '[[', 'trainRMSE'))
evalRMSE<-do.call(rbind, lapply(walkOut, '[[', 'evalRMSE'))

# function cleaning up names
cleanMod <- function(x){
        x <- gsub(pattern = "glm", replacement = "Logit", x)
        x <- gsub(pattern = "null", replacement = "Null", x)
        x <- gsub(pattern = "lm", replacement = "OLS", x)
        x <- gsub(pattern = "pls", replacement = "PLS", x)
        x <- gsub(pattern = "enet", replacement = "Elastic Net", x)
        x <- gsub(pattern = "C5.0", replacement = "C5.0", x)
        x <- gsub(pattern = "ranger", replacement = "Random Forest", x)
        x <- gsub(pattern = "rpart2", replacement = "CART", x)
        x <- gsub(pattern = "LogitBoost", replacement = "Boosted Logit", x)
        x <- gsub(pattern = "svmRadial", replacement = "SVM Radial", x)
        x <- gsub(pattern = "knn", replacement = "KNN", x)
        x <- gsub(pattern = "avNNet", replacement = "Neural Nets", x)
        x <- gsub(pattern = "ensemble", replacement = "Ensemble", x)
        x <- gsub(pattern = "cubist", replacement = "Cubist", x)
        x <- gsub(pattern = "ensemble", replacement = "Ensemble", x)
        x <- gsub(pattern = "gbm", replacement = "Boosted Trees", x)
        x <- gsub(pattern = "earth", replacement = "MARS", x)
        
        x
}

# average performance
walkTab<-evalRMSE %>%
        dplyr::group_by(Model=cleanMod(Model)) %>%
        dplyr::summarize(rmse=mean(RMSE),
                  sd=sd(RMSE)) %>%
        arrange(rmse) %>%
        as.data.frame() %>%
        tibble::column_to_rownames(var="Model")

walkTab

# clean up names
evalRMSE <- evalRMSE %>%
        mutate(Model=cleanMod(Model))

# plot evalRMSE by model over time
ggplot(evalRMSE, aes(x=Time, y=RMSE))+
        geom_point(data=evalRMSE,aes(x=Time, y=RMSE, color=Model))+
        geom_line(data=evalRMSE, aes(x=Time, y=RMSE, color=Model, group=Model))+
        theme_minimal()

# trim this, plot only null, ols, ranger, and cubist
trimRMSE <- evalRMSE%>%
        dplyr::filter(Model %in% c("Null", "OLS", "Cubist", "Random Forest"))

ggplot(trimRMSE, aes(x=Time, y=RMSE))+
        geom_point(data=trimRMSE,aes(x=Time, y=RMSE, color=Model), size=3)+
        geom_line(data=trimRMSE, aes(x=Time, y=RMSE, color=Model, group=Model), linetype="dashed")+
        theme_minimal()


### iterate over moving windows

# clean
rm(list=ls(pattern="walk_"))


### Re run with a moving window
# test on a two year window
test_breaks<-seq(1997, 2012, 2)

sliceTest<-foreach(i=1:(length(test_breaks))) %do%{
        years<-(test_breaks[i]):(test_breaks[i]+1)
}

# windows to test with
q<-c(2, 4, 6, 8)


# to select dataset that corresponds to these years, use subset
# steadily increasing window of train data
# change to 4 year moving

walkOut_move<-foreach(h=1:(length(q))) %do%{
        
        sliceTrain<-foreach(j=1:length(test_breaks)) %do% {
                years<-(test_breaks[j]-q[h]):(test_breaks[j]-1)
                }
        
        hold<-foreach(i=1:length(sliceTest)) %do% {
        
                # expanding window of train data
                walkTrain<-subset(naiveDat, 
                                  year>=min(sliceTrain[[i]]) & year<=max(sliceTrain[[i]]))
                
                # two year window of test data
                walkTest<-subset(naiveDat,
                                 year>=min(sliceTest[[i]]) & year<=max(sliceTest[[i]]))
                
                # train models
                set.seed(1999)
                walk_null<-suppressWarnings(train(hisinc ~ 1, 
                                                  data=walkTrain,
                                                  method="lm",
                                                  trControl=ctrlParallel))
                walk_null$method<-"Null"
                
                # ols
                set.seed(1999)
                walk_ols<-suppressWarnings(train(hisinc ~ ., 
                                                 data=walkTrain,
                                                 method="lm",
                                                 trControl=ctrlParallel))
                
                # pls
                set.seed(1999)
                walk_pls<-suppressWarnings(train(hisinc ~ ., 
                                                 data=walkTrain,
                                                 method="pls",
                                                 tuneLength=5,
                                                 trControl=ctrlParallel,
                                                 preProcess=c("center", "scale")))
                
                # cart
                set.seed(1999)
                walk_cart<-suppressWarnings(train(hisinc ~ .,
                                                  data=walkTrain,
                                                  tuneLength=5,
                                                  method="rpart2",
                                                  trControl=ctrlParallel))
                
                # KNN
                set.seed(1999)
                walk_knn<-suppressWarnings(train(hisinc ~ .,
                                                 data=walkTrain,
                                                 tuneLength=5,
                                                 method="knn",
                                                 trControl=ctrlParallel,
                                                 preProcess=c("center", "scale")))
                # MARS
                set.seed(1999)
                walk_mars<-train(hisinc ~ ., 
                                 data=walkTrain,
                                 method="earth",
                                 tuneLength=5,
                                 trControl=ctrlParallel,
                                 preProcess=c("center", "scale"))
                
                # Cubist
                set.seed(1999)
                walk_cub<-train(hisinc ~ ., 
                                data=walkTrain,
                                method="cubist",
                                tuneLength=5,
                                trControl=ctrlParallel)
                
                # Boosted Trees
                set.seed(1999)
                walk_gbm<-train(hisinc ~ ., 
                                data=walkTrain,
                                method="gbm",
                                tuneLength=5,
                                trControl=ctrlParallel,
                                verbose=F)
                
                # random forest
                set.seed(1999)
                walk_rf<-train(hisinc ~ ., 
                               data=walkTrain,
                               method="ranger",
                               tuneLength=5,
                               trControl=ctrlParallel)
                
                # svm
                set.seed(1999)
                walk_svm<-train(hisinc ~ ., 
                                data=walkTrain,
                                method="svmRadial",
                                tuneLength=5,
                                trControl=ctrlParallel,
                                preProcess=c("center", "scale"))
                
                models_walk<-lapply(ls(pattern="walk_"), get)
                
                # training performance
                trainPred<-as.tbl(foreach(i=1:length(models_walk), .combine=cbind.data.frame) %do% {
                        
                        bar <- as.tbl(models_walk[[i]]$pred) %>% 
                                arrange(rowIndex) %>%
                                dplyr::select(pred)
                        bar
                        
                        names(bar)<-models_walk[[i]]$method
                        bar
                        
                })
                
                trainObs<- as.matrix(models_walk[[1]]$pred %>% 
                                             arrange(rowIndex) %>%
                                             dplyr::select(obs))
                
                # rmse on the test set
                trainRMSE<-as.tbl(data.frame(Model=names(trainPred),
                                             RMSE=apply(as.matrix(trainPred), 2, error, trainObs),
                                             Time=min(walkTest$year)))
                
                # grab predictions for test
                evalPred<-as.tbl(
                        foreach(j=1:length(models_walk), .combine = cbind.data.frame) %do% {
                                pred<-data.frame(predict.train(models_walk[[j]], newdata=walkTest))
                                colnames(pred)<-models_walk[[j]]$method
                                pred
                        })
                
                # rmse on the test set
                evalRMSE<-as.tbl(data.frame(Model=names(evalPred),
                                            RMSE=apply(as.matrix(evalPred), 2, error, as.matrix(walkTest$hisinc)),
                                            Time=min(walkTest$year)))
                
                out<-list("trainPred"=trainPred,
                          "trainRMSE"=trainRMSE,
                          "evalPred"=evalPred,
                          "evalRMSE"=evalRMSE)
                
                out
                
        }
        
       # trainPred_move<-do.call(rbind, lapply(hold, '[[', 'trainPred'))
        #evalPred_move<-do.call(rbind, lapply(hold, '[[', 'evalPred'))
        
        #trainRMSE_move<-do.call(rbind, lapply(hold, '[[', 'trainRMSE'))
        evalRMSE_move<-do.call(rbind, lapply(hold, '[[', 'evalRMSE'))
        
        
        walkTab_move<-evalRMSE_move %>%
                dplyr::group_by(Model=cleanMod(Model)) %>%
                dplyr::summarize(rmse=mean(RMSE),
                                 sd=sd(RMSE)) %>%
                arrange(rmse) %>%
                dplyr::group_by(Window=q[h])
        
        walkTab_move
        
}

# 
moveOut<-do.call(rbind, walkOut_move)

moveTab<-t(moveOut %>% group_by(Window, Model) %>%
        dplyr::summarize(mean=rmse) %>%
        spread(Model, mean)) %>%
        as.data.frame()

colnames(moveTab)<-c("2", "4", "6", "8")
moveTab<-moveTab[-1,]


# Combine into one table
walkTab$name<-rownames(walkTab)
moveTab$name<-rownames(moveTab)

tabOut<-left_join(walkTab, moveTab, by="name") %>%
        tibble::column_to_rownames(var="name")
colnames(tabOut)<-c("RMSE", "SD", "2", "4", "6", "8")
        
tabOut

}



rmse_imputations<-apply(abind(lapply(tune_imputations, '[[', 'rmse'), along=3), 1:2, mean)





