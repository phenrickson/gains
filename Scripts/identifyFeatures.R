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
library(pls)
library(elasticnet)
library(lars)
library(glmnet)
library(rpart)
library(randomForest)
library(party)
library(plotmo)
library(ranger)
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

# Clean up varnames
cleanMod <- function(x){
        x <- gsub(pattern = "hisinc", replacement = "Income", x)
        x <- gsub(pattern = "year", replacement = "Year", x)
        x <- gsub(pattern = "college", replacement = "College", x)
        x <- gsub(pattern = "postgrad", replacement = "PostGrad", x)
        x <- gsub(pattern = "legal", replacement = "Legal", x)
        x <- gsub(pattern = "business", replacement = "Business", x)
        x <- gsub(pattern = "management", replacement = "Management", x)
        x <- gsub(pattern = "protestant", replacement = "Protestant", x)
        x <- gsub(pattern = "catholic", replacement = "Catholic", x)
        x <- gsub(pattern = "white", replacement = "White", x)
        x <- gsub(pattern = "black", replacement = "Black", x)
        x <- gsub(pattern = "hispanic", replacement = "Hispanic", x)
        x <- gsub(pattern = "marryyes", replacement = "Married", x)
        x <- gsub(pattern = "privatepost", replacement = "PrivatePost", x)
        x <- gsub(pattern = "publicpost", replacement = "PublicPost", x)
        x <- gsub(pattern = "highpost", replacement = "HighPost", x)
        x <- gsub(pattern = "lowpost", replacement = "LowPost", x)
        x <- gsub(pattern = "taxreturn", replacement = "TaxReturn", x)
        x <- gsub(pattern = "leadership", replacement = "Leadership", x)
        x <- gsub(pattern = "chair", replacement = "Chair", x)
        x <- gsub(pattern = "rules", replacement = "Rules", x)
        x <- gsub(pattern = "approp", replacement = "AppropriationsCmte", x)
        x <- gsub(pattern = "fintax", replacement = "FinancesTaxCmte", x)
        x <- gsub(pattern = "agriculture", replacement = "AgricultureCmte", x)
        x <- gsub(pattern = "judiciary", replacement = "JudiciaryCmte", x)
        x <- gsub(pattern = "education", replacement = "EducationCmte", x)
        x <- gsub(pattern = "health", replacement = "HealthCmte", x)
        x <- gsub(pattern = "speaker", replacement = "Speaker", x)
        x <- gsub(pattern = "majld", replacement = "MajorLeader", x)
        x <- gsub(pattern = "majpty", replacement = "MajorParty", x)
        x <- gsub(pattern = "ethicsviol", replacement = "EthicsViolations", x)
        x <- gsub(pattern = "ethicscmte", replacement = "EthicsCmte", x)
        x <- gsub(pattern = "board", replacement = "Board", x)
        x <- gsub(pattern = "binboard", replacement = "BinBoard", x)
        x <- gsub(pattern = "female", replacement = "Female", x)
        x <- gsub(pattern = "aide", replacement = "Aide", x)
        x <- gsub(pattern = "council", replacement = "Council", x)
        x <- gsub(pattern = "mayor", replacement = "Mayor", x)
        x <- gsub(pattern = "commish", replacement = "Commisioner", x)
        x <- gsub(pattern = "applocal", replacement = "LocalAppointed", x)
        x <- gsub(pattern = "appstate", replacement = "StateAppointed", x)
        x <- gsub(pattern = "locparty", replacement = "LocalParty", x)
        x <- gsub(pattern = "statparty", replacement = "StateParty", x)
        x <- gsub(pattern = "prelect", replacement = "PrevElected", x)
        x <- gsub(pattern = "prapp", replacement = "PrevAppointed", x)
        x <- gsub(pattern = "prparty", replacement = "PrevParty", x)
        x <- gsub(pattern = "prexp", replacement = "PrevExp", x)
        x <- gsub(pattern = "counter", replacement = "Counter", x)
        x <- gsub(pattern = "age", replacement = "Age", x)
        x <- gsub(pattern = "yrsmilitary", replacement = "YrsMilitary", x)
        x <- gsub(pattern = "gdp01", replacement = "GDP_lag", x)
        x <- gsub(pattern = "yrselect", replacement = "YrsElected", x)
        x <- gsub(pattern = "yrsparty", replacement = "YrsParty", x)
        x <- gsub(pattern = "numberofjobs", replacement = "NumberJobs", x)
        x <- gsub(pattern = "effectivescore", replacement = "EffectiveScore", x)
        x <- gsub(pattern = "deltajobs", replacement = "DeltaJobs", x)
        x <- gsub(pattern = "tenure", replacement = "Tenure", x)
        x <- gsub(pattern = "votemarg", replacement = "VoteMargin", x)
        x <- gsub(pattern = "networth01", replacement = "NetWorth_lag", x)
        x
}

# Select which data set
dat<-dplyr::select(levels.dat, -memberid, -X,-caseid, -district, -birth)
colnames(dat)<-cleanMod(colnames(dat))


# RMSE function
error<-function(pred.Y, Y) {
        sqrt(sum((pred.Y-Y)^2)/nrow(Y))
}

# Test Data Set
# We are predicting the years 2013-2014 across all of our modeling approches
testDat<-as.data.table(subset(dat, Year>=2012))
# Do not touch until the very end!


### Average across imputations ###

#dat<-dat[[h]]

##### Naive Training ######
# We split the training data into five separate folds irrespective of time
# Conduct nested cross validation to identify the model which performs best out of sample
set.seed(1999)
naiveDat<-as.data.table(subset(dat, Year<2012))
naiveIter<-createFolds(naiveDat$Income, k=5)
naiveFolds<-list(c(naiveIter$Fold1),  c(naiveIter$Fold2), c(naiveIter$Fold3), c(naiveIter$Fold4), c(naiveIter$Fold5))

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

### Feature selection/regularization: Lasso and ridge 
# Set data set up for this

newdat<-data.table(naiveDat)

y<-dplyr::select(newdat, Income)
x_c<-dplyr::select(newdat, Age, YrsMilitary, GDP_lag, YrsElect, Yrsparty, numberofjobs, effectivescore, deltajobs, tenure, votemarg, networth01)
x_d<-dplyr::select(newdat, -hisinc, -age, -yrsmilitary, -gdp01, -yrselect, -gdp01, -yrsparty, -numberofjobs, -effectivescore, -deltajobs, -tenure, -votemarg, -networth01)

scaled<-scale(x_c, center=T, scale=T)

x<-data.table(x_d, scaled)

library(glmnet)

x<-as.matrix(x)
y<-as.matrix(y)

### Lasso/Ridge
colnames(x)<-cleanMod(colnames(x))
colnames(y)<-cleanMod(colnames(y))

# lasso
lasso<-glmnet(x, y, alpha=1, family="gaussian")
set.seed(1999)
cv.lasso<-cv.glmnet(x, y, alpha=1, family="gaussian", nfolds=nrow(x))
coef.lasso<-coef(cv.lasso, s="lambda.1se")

# ridge
ridge<-glmnet(x, y, alpha=0, family="gaussian")
set.seed(1999)
cv.ridge<-cv.glmnet(x, y, alpha=0, family="gaussian", nfolds=nrow(x))
coef.ridge<-coef(cv.ridge, s="lambda.1se")

# plot
plot(cv.lasso)
plot_glmnet(lasso,'lambda', label=10)
abline(v=log(cv.lasso$lambda.1se), lty=2, col="grey60")
abline(v=log(cv.lasso$lambda.min), lty=2)

plot(cv.ridge)
plot_glmnet(ridge,'rlambda', label=10)
abline(v=log(cv.ridge$lambda.1se), lty=2, col="grey")
abline(v=log(cv.ridge$lambda.min), lty=2)


# now run an OLS
set.seed(1999)
ols <-lm(y~x)

coef.ols<-coef(ols)

# coefficients from OLS, ridge, lasso
coef.mat<-cbind(coef.ols, coef.ridge, coef.lasso)
colnames(coef.mat)<-c("OLS", "Ridge", "Lasso")
coefTab<-round(coef.mat, 3)


##### Decision Tree
# fit carts to these datasets
tune_control<-trainControl(method="repeatedcv",
                           n=5, repeats=5,
                           predictionBounds = c(0, NA),
                           savePredictions="final")

# run CART
set.seed(1999)
cart<-train(Income~.,
            data=dat,
            method="rpart2",
            tuneLength=20,
            trControl=tune_control)

# 
library(rpart.plot)
title<-paste("CART: Maxdepth = ", paste(cart$bestTune[1]), sep="")
prp(cart$finalModel, main=title)

library(rattle)
fancyRpartPlot(cart$finalModel, sub="", tweak=0.65, cex=.85)


### Partial dependence plots
# quickly tune rf to set tuning parameter
set.seed(1999)
rf_tune<-train(Income~.,
            data=as.data.frame(dat),
            method="ranger",
            tuneLength=5,
            importance="permutation",
            trControl=ctrlParallel)

# partial dependence for each observation
pred.ice<- function(object, newdata) predict(object, newdata)

### VoteMargin
# compute
rm.ice<-partial(rf_tune, pred.var="VoteMargin", pred.fun=pred.ice)

# mutate 
rm.ice <- rm.ice %>%
        group_by(yhat.id) %>% # perform next operation within each yhat.id
        mutate(yhat.centered = yhat - first(yhat)) # so each curve starts at yhat = 0

# plot
ggplot(rm.ice, aes(VoteMargin, yhat))+
        geom_line(aes(group=yhat.id), alpha=0.2) +
        stat_summary(fun.y=mean, geom="line", col="red", size=1) +
        labs(y="Income (IHS)")+
        theme_minimal()

### Number of Jobs
# compute
rm.ice<-partial(rf_tune, pred.var="NumberJobs", pred.fun=pred.ice)

# mutate 
rm.ice <- rm.ice %>%
        group_by(yhat.id) %>% # perform next operation within each yhat.id
        mutate(yhat.centered = yhat - first(yhat)) # so each curve starts at yhat = 0

# plot
ggplot(rm.ice, aes(NumberJobs, yhat))+
        geom_line(aes(group=yhat.id), alpha=0.2) +
        stat_summary(fun.y=mean, geom="line", col="red", size=1) +
        labs(y="Income (IHS)")+
        theme_minimal()


### Female
rm.ice<-partial(rf_tune, pred.var="Female", pred.fun=pred.ice)

# mutate 
rm.ice <- rm.ice %>%
        group_by(yhat.id) %>% # perform next operation within each yhat.id
        mutate(yhat.centered = yhat - first(yhat)) # so each curve starts at yhat = 0

# plot
ggplot(rm.ice, aes(Female, yhat))+
        geom_point(aes(group=yhat.id), alpha=0.2) +
        stat_summary(fun.y=mean, geom="line", col="red", size=1) +
        labs(y="Income (IHS)")+
        theme_minimal()



### YrsMilitary
# compute
rm.ice<-partial(rf_tune, pred.var="YrsMilitary", pred.fun=pred.ice)

# mutate 
rm.ice <- rm.ice %>%
        group_by(yhat.id) %>% # perform next operation within each yhat.id
        mutate(yhat.centered = yhat - first(yhat)) # so each curve starts at yhat = 0

# plot
ggplot(rm.ice, aes(YrsMilitary, yhat))+
        geom_line(aes(group=yhat.id), alpha=0.2) +
        stat_summary(fun.y=mean, geom="line", col="red", size=1) +
        labs(y="Income (IHS)")+
        theme_minimal()


### VoteMargin and Net Worth
partial(rf_tune, pred.var=c("VoteMargin", "NetWorth_lag"), plot=T, chull=T)

### VoteMargin and Number of Jobs
partial(rf_tune, pred.var=c("VoteMargin", "NumberJobs"), plot=T, chull=T)







######## Variable Importance
### OLS
# grab the variables

vars<-colnames(dat)[-1]

# select a standard set of variables
# GDP_lag, Networth_lag
stand.vars<-which(vars %in% c("GDP_lag", "NetWorth_lag", "Year"))

# then, add in each other variable one by one
perm.vars<-seq(1, length(vars))[-stand.vars]

# set the tuning
tune_control<-trainControl(method="cv", 
                           n=5, 
                           predictionBounds = c(0, NA), 
                           savePredictions="final", 
                           allowParallel=T)

# iterate over variables, bootstrap for CIs
# register parallel backend
cl <- makeCluster(7)
registerDoParallel(cl)

n=1000

permVars_ols<-
        foreach(i=1:length(perm.vars),.combine=rbind, .packages=c('dplyr', 'foreach', 'caret')) %:% 
        foreach(j=1:n, .combine=rbind, .packages=c('dplyr', 'foreach', 'caret')) %dopar% {
                
                form<-as.formula(paste("Income~", paste(c(paste(vars[stand.vars]), paste(vars[perm.vars[i]])), collapse="+")))
                
                boot<-sample(1:nrow(dat), replace=T)
                dat.boot<-dat[boot,]
                
                olsvar<-train(form,
                              data=dat.boot,
                              method="lm",
                              trControl=tune_control)
                
                # make data.frame here
                tabVars<-data.frame(Variable=vars[perm.vars[i]],
                                    RMSE=olsvar$results[2])
                
                tabVars
        }

# extract
permTab<-permVars_ols %>%
        dplyr::group_by(Variable = as.character(cleanMod(Variable))) %>%
        dplyr::summarise(mean = mean(RMSE),
                         se = sd(RMSE),
                         ci_low=sort(RMSE)[round(0.05*n,1)-1],
                         ci_high=sort(RMSE)[round(0.95*n,1)+1]) %>%
        ungroup()
permTab

# order
olsTab<-permTab[order(-permTab$mean), , drop=F]
#save(olsTab, file="Output/olsTab.Rdata")
#load("Output/olsTab.Rdata")


# plot
y.plot<-seq(1, nrow(olsTab))
bar<-data.frame(olsTab, y.plot)
rm(y.plot)

# Null
form<-as.formula(paste("a_BatDeath_IHS~", paste(vars[stand.vars], collapse="+")))

null<-train(Income~1,
            data=dat,
            method="lm",
            trControl=tune_control)

base<-train(Income~NetWorth_lag+GDP_lag+Year,
            data=dat,
            method="lm",
            trControl=tune_control)

rmse.null<-null$results[2]
rmse.base<-base$results[2]


# plot in ggplot
olsPlot<-ggplot(bar, aes(x=mean, y=y.plot))+
        geom_point(size=1)+
        geom_errorbarh(aes(xmin=ci_low, xmax=ci_high), alpha=0.5, height=0, lwd=0.75) +
        coord_cartesian(xlim=c(0.6, 0.8),ylim=c(1,length(perm.vars)))+
        scale_y_continuous(breaks = pretty(bar$y.plot, n = nrow(bar)), labels=bar$Variable)+        
        labs(x="RMSE")+
        labs(y="")+
        ggtitle("OLS")+
       # geom_vline(xintercept = rmse.null$RMSE, "x")+
        geom_vline(xintercept = rmse.base$RMSE, "x")+
        theme_minimal()
olsPlot

### Repeat for random forests
# register parallel backend
cl <- makeCluster(7)
registerDoParallel(cl)

# set tune control
tune_control<-trainControl(method="cv", n=5,
                           predictionBounds = c(0, NA),
                           savePredictions="final",
                           allowParallel=T)

# register parallel backend
cl <- makeCluster(7)
registerDoParallel(cl)

n=1000

permVars_rf<-
        foreach(j=1:n, .combine=rbind, .packages=c('dplyr', 'foreach', 'caret', 'randomForestSRC', 'ggRandomForests')) %dopar% {
                
                boot<-sample(1:nrow(dat), replace=T)
                dat.boot<-dat[boot,]
                
                rf_tune<-train(Income~.,
                               data=dat.boot,
                               method="ranger",
                               tuneGrid=expand.grid(.mtry=16),
                               importance="permutation",
                               trControl=tune_control)
                
                hold<-rf_tune$finalModel$variable.importance
                
                out<-data.frame(variable=names(hold),
                                 importance=hold)
                out
                
        }


# extract
permTab<-permVars_rf %>%
        dplyr::group_by(Variable = as.character(cleanMod(variable))) %>%
        dplyr::summarise(mean = mean(importance),
                         se = sd(importance),
                         ci_low=sort(importance)[round(0.05*n,1)-1],
                         ci_high=sort(importance)[round(0.95*n,1)+1]) %>%
        ungroup() %>%
        arrange(mean)

rfTab<-permTab

# plot
y.plot<-seq(1, nrow(rfTab))
bar<-data.frame(rfTab, y.plot)
rm(y.plot)

# plot in ggplot
rfPlot<-ggplot(bar, aes(x=mean, y=y.plot))+
        geom_point(size=1)+
        geom_errorbarh(aes(xmin=ci_low, xmax=ci_high), alpha=0.5, height=0, lwd=0.75) +
        coord_cartesian(xlim=c(0, 1),ylim=c(1,length(perm.vars)))+
        scale_y_continuous(breaks = pretty(bar$y.plot, n = nrow(bar)), labels=bar$Variable)+        
        labs(x="Importance")+
        labs(y="")+
        ggtitle("Random Forest")+
        # geom_vline(xintercept = rmse.null$RMSE, "x")+
       geom_vline(xintercept = 0, "x", linetype="dotted")+
        theme_minimal()
rfPlot




### Conditional inference trees
# quickly tune to set tuning parameter
set.seed(1999)
cforest_tune<-train(Income~.,
                data=as.data.frame(dat),
                method="cforest",
                tuneLength=5,
                trControl=ctrlParallel)

cforest_par<-ctree_tune$bestTune

# register parallel backend
cl <- makeCluster(7)
registerDoParallel(cl)

n=1000

permVars_cforest<-
        foreach(j=1:n, .combine=rbind, .packages=c('dplyr', 'foreach', 'caret', 'randomForestSRC', 'ggRandomForests')) %dopar% {
                
                boot<-sample(1:nrow(dat), replace=T)
                dat.boot<-dat[boot,]
                
                cforest_tune<-train(Income~.,
                               data=dat.boot,
                               method="cforest",
                               tuneGrid=cforest_par,
                               trControl=tune_control)
                
                hold<-varimp(cofrest_tune$finalModel)
                
                out<-data.frame(variable=names(hold),
                                importance=hold)
                out
                
        }


# extract
permTab<-permVars_cforest %>%
        dplyr::group_by(Variable = as.character(cleanMod(variable))) %>%
        dplyr::summarise(mean = mean(importance),
                         se = sd(importance),
                         ci_low=sort(importance)[round(0.05*n,1)-1],
                         ci_high=sort(importance)[round(0.95*n,1)+1]) %>%
        ungroup() %>%
        arrange(mean)

cforestTab<-permTab

# plot
y.plot<-seq(1, nrow(cforestTab))
bar<-data.frame(cforestTab, y.plot)
rm(y.plot)

# plot in ggplot
cforestPlot<-ggplot(bar, aes(x=mean, y=y.plot))+
        geom_point(size=1)+
        geom_errorbarh(aes(xmin=ci_low, xmax=ci_high), alpha=0.5, height=0, lwd=0.75) +
        coord_cartesian(xlim=c(0, 1),ylim=c(1,length(perm.vars)))+
        scale_y_continuous(breaks = pretty(bar$y.plot, n = nrow(bar)), labels=bar$Variable)+        
        labs(x="Importance")+
        labs(y="")+
        ggtitle("Conditional Inference Forest")+
        # geom_vline(xintercept = rmse.null$RMSE, "x")+
        geom_vline(xintercept = 0, "x", linetype="dotted")+
        theme_minimal()
cforestPlot





### Cubist
# rerun cubist to find tuning parameter
# quickly tune to set tuning parameter
set.seed(1999)
cub_tune<-train(Income~.,
               data=as.data.frame(dat),
               method="cubist",
               tuneLength=5,
               trControl=ctrlParallel)

cub_par<-cub_tune$bestTune

n<-1000

permVars_cub<-
        foreach(j=1:n, .combine=rbind, .packages=c('dplyr', 'foreach', 'caret')) %dopar% {
                
                boot<-sample(1:nrow(dat), replace=T)
                dat.boot<-dat[boot,]
                
                cubModel_boots<-train(Income~., 
                                      data=dat.boot,  
                                      method="cubist", 
                                      trControl=tune_control, 
                                      tuneGrid = cub_par,
                                      parallel=T)
                
                hold<-data.frame(variable=cubModel_boots$finalModel$usage$Variable,
                                 conditions=cubModel_boots$finalModel$usage$Conditions,
                                 model=cubModel_boots$finalModel$usage$Model)
                hold
        }

# grab conditions
permTab_conditions<-permVars_cub %>%
        dplyr::group_by(Variable = as.character(cleanMod(variable))) %>%
        dplyr::summarise(mean = mean(conditions),
                         se = sd(conditions),
                         ci_low=sort(conditions)[round(0.05*n,1)-1],
                         ci_high=sort(conditions)[round(0.95*n,1)+1]) %>%
        ungroup() %>%
        arrange(mean)

cubTab_conditions<-permTab_conditions

# plot
y.plot<-seq(1, nrow(cubTab_conditions))
bar<-data.frame(cubTab_conditions, y.plot)
rm(y.plot)

# plot in ggplot
# conditions
cubPlot_conditions<-ggplot(bar, aes(x=mean, y=y.plot))+
        geom_point(size=1)+
        geom_errorbarh(aes(xmin=ci_low, xmax=ci_high), alpha=0.5, height=0, lwd=0.75) +
        coord_cartesian(xlim=c(0, 100),ylim=c(1,length(perm.vars)))+
        scale_y_continuous(breaks = pretty(bar$y.plot, n = nrow(bar)), labels=bar$Variable)+        
        labs(x="Conditions")+
        labs(y="")+
        ggtitle("Cubist")+
        # geom_vline(xintercept = rmse.null$RMSE, "x")+
        geom_vline(xintercept = 0, "x", linetype="dotted")+
        theme_minimal()
cubPlot_conditions


# grab model
permTab_model<-permVars_cub %>%
        dplyr::group_by(Variable = as.character(cleanMod(variable))) %>%
        dplyr::summarise(mean = mean(model),
                         se = sd(model),
                         ci_low=sort(model)[round(0.05*n,1)-1],
                         ci_high=sort(model)[round(0.95*n,1)+1]) %>%
        ungroup() %>%
        arrange(mean)

cubTab_model<-permTab_model

load("Output/cubTab.Rdata")
# plot
y.plot<-seq(1, nrow(cubTab_model))
bar<-data.frame(cubTab_model, y.plot)
rm(y.plot)

# plot in ggplot
# model
cubPlot_model<-ggplot(bar, aes(x=mean, y=y.plot))+
        geom_point(size=1)+
        geom_errorbarh(aes(xmin=ci_low, xmax=ci_high), alpha=0.5, height=0, lwd=0.75) +
        coord_cartesian(xlim=c(0, 100),ylim=c(1,length(perm.vars)))+
        scale_y_continuous(breaks = pretty(bar$y.plot, n = nrow(bar)), labels=bar$Variable)+        
        labs(x="Model")+
        labs(y="")+
        ggtitle("Cubist")+
        # geom_vline(xintercept = rmse.null$RMSE, "x")+
        geom_vline(xintercept = 0, "x", linetype="dotted")+
        theme_minimal()
cubPlot_model


grid.arrange(rfPlot, cubPlot_conditions, cubPlot_model, ncol=1)




#### Let's fiddle around with Altman et al
# quickly tune to set tuning parameter
set.seed(1999)
rf<-train(Income~.,
                    data=as.data.frame(naiveDat),
                    method="ranger",
                    importance="permutation",
                    tuneLength=5,
                    trControl=ctrlParallel)

# compute altman p values
set.seed(1999)
rf_object<-importance_pvalues(rf$finalModel, 
                              method="altmann",
                              num.permutations=100,
                              formula=Income~., 
                              data=naiveDat)

rf_frame<-as.data.frame(rf_object)

rfpTab<-rf_frame[order(-rf_frame$importance), , drop=F]
rfpTab

# filter out varaibles with p values greater than 0.05
omit<-rownames(rfpTab[which(rfpTab$pvalue>0.05),])

pimp_dat<-dplyr::select(naiveDat, -one_of(omit))

set.seed(1999)
rf_pimp<-train(Income~.,
          data=as.data.frame(pimp_dat),
          method="ranger",
          importance="permutation",
          tuneLength=5,
          trControl=ctrlParallel)

# performance?
error<-function(pred.Y, Y) {
        sqrt(sum((pred.Y-Y)^2)/nrow(Y))
}

# test on hold out
error(predict(rf, newdat=testDat), as.matrix(testDat$Income))
error(predict(rf_pimp, newdat=testDat), as.matrix(testDat$Income))

plot(predict(rf, newdat=testDat), testDat$Income, xlim=c(0,7), ylim=c(0,7))
abline(0,1)

plot(predict(rf_pimp, newdat=testDat), testDat$Income, xlim=c(0,7), ylim=c(0,7))
abline(0,1)

ols<-train(Income~.,
           data=as.data.frame(pimp_dat),
           method="lm",
           trControl=ctrlParallel)



