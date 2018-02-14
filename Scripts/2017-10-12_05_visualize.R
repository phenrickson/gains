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
# 
X<-dplyr::select(trainTime, -hisinc)

depvar<-"hisinc"
features<-colnames(X)

form<-as.formula(paste(depvar, paste(features, collapse="+"), sep="~"))

# stepAIC
initial<-lm(form, data=dat)

outStep<-stepAIC(initial, direction="both")

# lasso/ridge this mofo
newdat<-data.table(dat)

y<-dplyr::select(newdat, hisinc)
x_c<-dplyr::select(newdat, age, yrsmilitary, gdp01, yrselect, gdp01, yrsparty, numberofjobs, effectivescore, deltajobs, tenure, votemarg, networth01)
x_d<-dplyr::select(newdat, -hisinc, -age, -yrsmilitary, -gdp01, -yrselect, -gdp01, -yrsparty, -numberofjobs, -effectivescore, -deltajobs, -tenure, -votemarg, -networth01)

scaled<-scale(x_c, center=T, scale=T)

x<-data.table(x_d, scaled)

library(glmnet)

x<-as.matrix(x)
y<-as.matrix(y)


library(plotmo)
# lasso

lasso<-glmnet(x, y, alpha=1, family="gaussian")

set.seed(1999)
cv.lasso<-cv.glmnet(x, y, alpha=1, family="gaussian", nfolds=nrow(x))
coef.lasso<-coef(cv.lasso, s="lambda.min")

plot(cv.lasso)
plot_glmnet(lasso,'lambda', label=10)
abline(v=log(cv.lasso$lambda.1se), lty=2)

# ridge
ridge<-glmnet(x, y, alpha=0, family="gaussian")

set.seed(1999)
cv.ridge<-cv.glmnet(x, y, alpha=0, family="gaussian", nfolds=nrow(x))
coef.ridge<-coef(cv.ridge, s="lambda.min")

plot(cv.ridge)
plot_glmnet(ridge,'rlambda', label=10)
abline(v=log(cv.ridge$lambda.1se), lty=2)

set.seed(1999)
ols <-lm(y~x)

coef.ols<-coef(ols)


# coefficients from OLS, ridge, lasso
coef.mat<-cbind(coef.ols, coef.ridge, coef.lasso)
colnames(coef.mat)<-c("OLS", "Ridge", "Lasso")
coef.mat<-round(coef.mat, 3)
coef.mat


# # Grab coefficients and standard errors with subset, shrinkage, and feature selection
form<-colnames(trainTime)
form



# Fit cubist, random forests,  cart, and boosted trees to the entirety of the dataset

# Show decision trees

# Extract variable importance measures

# Partial dependence plots


# compute OLS with subset selection 
