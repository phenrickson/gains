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

# RMSE function
error<-function(pred.Y, Y) {
        sqrt(sum((pred.Y-Y)^2)/nrow(Y))
}

# Test Data Set
# We are predicting the years 2013-2014 across all of our modeling approches
testDat<-as.data.table(subset(dat, year>=2012))
# Do not touch until the very end!


### Average across imputations ###

#dat<-dat[[h]]

##### Naive Training ######
# We split the training data into five separate folds irrespective of time
# Conduct nested cross validation to identify the model which performs best out of sample
set.seed(1999)
naiveDat<-as.data.table(subset(dat, year<2012))
naiveIter<-createFolds(naiveDat$hisinc, k=5)
naiveFolds<-list(c(naiveIter$Fold1),  c(naiveIter$Fold2), c(naiveIter$Fold3), c(naiveIter$Fold4), c(naiveIter$Fold5))

#
# Split training data into five folds for nested cross validation
set.seed(1999)
folds<-createFolds(trainTime$hisinc, k=5)
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

### Feature selection/regularization: Lasso and ridge 
# Set data set up for this

newdat<-data.table(dat)

y<-dplyr::select(newdat, hisinc)
x_c<-dplyr::select(newdat, age, yrsmilitary, gdp01, yrselect, gdp01, yrsparty, numberofjobs, effectivescore, deltajobs, tenure, votemarg, networth01)
x_d<-dplyr::select(newdat, -hisinc, -age, -yrsmilitary, -gdp01, -yrselect, -gdp01, -yrsparty, -numberofjobs, -effectivescore, -deltajobs, -tenure, -votemarg, -networth01)

scaled<-scale(x_c, center=T, scale=T)

x<-data.table(x_d, scaled)

library(glmnet)

x<-as.matrix(x)
y<-as.matrix(y)

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
        x <- gsub(pattern = "gdp01", replacement = "GDP_t-1", x)
        x <- gsub(pattern = "yrselect", replacement = "YrsElected", x)
        x <- gsub(pattern = "yrsparty", replacement = "YrsParty", x)
        x <- gsub(pattern = "numberofjobs", replacement = "NumberJobs", x)
        x <- gsub(pattern = "effectivescore", replacement = "EffectiveScore", x)
        x <- gsub(pattern = "deltajobs", replacement = "DeltaJobs", x)
        x <- gsub(pattern = "tenure", replacement = "Tenure", x)
        x <- gsub(pattern = "votemarg", replacement = "VoteMargin", x)
        x <- gsub(pattern = "networth01", replacement = "NetWorth_t-1", x)
        x
}

library(plotmo)

colnames(x)<-cleanMod(colnames(x))

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
coef.mat<-round(coef.mat, 3)
coef.mat





# Variable importance scores
# OLS


# Random Forest


# Cubist






# # Grab coefficients and standard errors with subset, shrinkage, and feature selection
form<-colnames(trainTime)
form
