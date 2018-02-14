#########################################
## The Perks of Being a Lawmaker       ##
## Kevin Fahey - Dissertation May 2017 ##
## Prepared 2017-5-24 For Monkey Cage  ##
#########################################

######################
## Clear Everything ##
######################

rm(list=ls())

setwd("K:/Dropbox/Machine Learning Paper")
setwd("B:/Dropbox/Machine Learning Paper")
setwd("/Backup//Dropbox/Machine Learning Paper")

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
library(MASS)
library(stats)
library(xtable)
library(dplyr)
library(tidyr)
library(data.table)
options(scipen=7)
options(digits=3)

## ClusterMod function ##

clusterMod<-function(model, cluster)
{
  require(multiwayvcov)
  require(lmtest)
  vcovCL<-cluster.vcov(model, cluster)
  
  coef<-coeftest(model, vcovCL)
  #w<-waldtest(model, vcov = vcovCL, test = "F")
  get_confint<-function(model, vcovCL){
    t<-qt(.975, model$df.residual)
    ct<-coeftest(model, vcovCL)
    cse<-sqrt(diag(vcovCL))
    est<-cbind(ct[,1], cse,ct[,1]-t*ct[,2], ct[,1]+t*ct[,2])
    colnames(est)<-c("Estimate", "Clustered SE","LowerCI","UpperCI")
    return(est)
  }
  ci<-round(get_confint(model, vcovCL),4)
  return(list(coef, ci))
}

######################
## Read in Datasets ##
######################

dat <- read.dta("K:/Dropbox/Dissertation/Chapter 1/Data/2017-1-22 perks of being a lawmaker.dta")
dat <- read.dta("B:/Dropbox/Dissertation/Chapter 1/Data/2017-1-22 perks of being a lawmaker.dta")
dat <- read.dta("/Backup/Dropbox/Dissertation/Chapter 1/Data/2017-1-22 perks of being a lawmaker.dta")


wc.dat <- read.csv("K:/Google Drive/Dissertation/Chapter 2/2017-6-14 wordcount.out.csv")
wc.dat <- read.csv("B:/Google Drive/Dissertation/Chapter 2/2017-6-14 wordcount.out.csv")
wc.dat <- read.csv("/Backup/Dropbox/Dissertation/Chapter 2/2017-6-14 wordcount.out.csv")


ef.dat <- read.csv("K:/Google Drive/Dissertation/Chapter 2/2017-6-14 effectiveness.out.csv")
ef.dat <- read.csv("B:/Google Drive/Dissertation/Chapter 2/2017-6-14 effectiveness.out.csv")
ef.dat <- read.csv("/Backup/Dropbox/Dissertation/Chapter 2/2017-6-14 effectiveness.out.csv")

ef.dat$caseid <- ef.dat$uniqueid

####################
## Main Variables ##
####################

merge1 <- left_join(wc.dat, ef.dat, by = "caseid")

newmerge1 <- merge1[,c("caseid", "avgwords", "logwords", "effectivescore")]

full <- left_join(dat, newmerge1, by = "caseid")


################################################
## Create a Loop to Identify NAs per variable ##
################################################


NA.size <- matrix(NA, ncol = 3, nrow = 133)
NA.size[,1] <- colnames(full)

for(i in 1:dim(full)[2]){
  
  NA.size[i,2] <- length(which(is.na(full[,i])==T))
  NA.size[i,3] <- length(which(is.na(full[,i])==T)) / 2905
  
}

save(trim.dat, file="trim.dat.Rdata")


NA.size.2 <- NA.size[order(NA.size[,3]),]


###################################
## Remove and Trim Bad Variables ##
###################################

trim.dat <- dplyr::select(full, caseid, year, district, firstname, lastname, college,
                    postgrad, birth, age, legal, business, management,
                    houseterms, otherrel, yrsmilitary, protestant, catholic,
                    jewish, white, black, hispanic, marryyes, yrselect,
                    privatepost, publicpost, highpost, lowpost, gdp, gdp01,
                    gdpdif, networth, networth01, difnet, income, income01, difinc,
                    yrsparty, taxreturn, numberofjobs, leadership, chair, 
                    rules, approp, fintax, agriculture, judiciary, education, health,
                    effectivescore, deltajobs, speaker, majld, minld,
                    majpty, ethicsviol, ethicscmte, board, binboard, dem,
                    female, aide, council, mayor, commish, applocal, appstate,
                    locparty, statparty, prelect, prapp, prparty, tenure, prexp, memberid,
                    votemarg)

## New NAs ##


NA.size.trim <- as.data.frame(matrix(NA, ncol = 3, nrow = ncol(trim.dat)))
NA.size.trim[,1] <- colnames(trim.dat)

for(i in 1:dim(trim.dat)[2]){
  
  NA.size.trim[i,2] <- length(which(is.na(trim.dat[,i])==T))
  NA.size.trim[i,3] <- length(which(is.na(trim.dat[,i])==T)) / 2905
  
}

NA.size.trim2 <- NA.size.trim[order(NA.size.trim[,3]),]


save(NA.size.trim2, file="NA.size.trim2.Rdata")
save(trim.dat, file="trim.dat.Rdata")

