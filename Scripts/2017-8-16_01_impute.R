## imputation


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
library(corrplot)
options(scipen=7)
options(digits=3)


load("trim.dat.Rdata")

dat<-dplyr::select(trim.dat, -caseid, -firstname, -lastname, -otherrel, -minld, -dem, -jewish, -gdp, -gdpdif, -networth, -networth01, difnet, -income, -income01, -difinc, -houseterms, -difnet)

# omit ordinal 

noms = c("taxreturn", "lowpost", "highpost", "privatepost", "publicpost",
         "marryyes", "black", "hispanic", "white", "catholic", "protestant",
         "management", "business", "legal", "postgrad",
         "college", "prexp", "prparty", "prelect", "prapp",
         "statparty", "locparty", "appstate", "applocal", "commish",
         "mayor", "council", "aide", "female", "fintax", "approp",
         "rules", "binboard", "ethicscmte", "majpty", "majld",
         "speaker", "health", "education", "judiciary", "agriculture",
         "chair", "leadership")

# impute

library(Amelia)
set.seed(1999)

impute.out<-amelia(dat, 
                   m = 25, 
                   ts = "year", 
                   cs = "memberid",
                   noms = noms,
                  # ords = ords,
                  # idvars = idvars,
                   polytime=3,
                   intercs=F,
                   max.resample=1000,
                   empri = 0.001 * nrow(dat))

impute.out1<-impute.out$imputations$imp1
impute.out2<-impute.out$imputations$imp2
impute.out3<-impute.out$imputations$imp3
impute.out4<-impute.out$imputations$imp4
impute.out5<-impute.out$imputations$imp5
impute.out6<-impute.out$imputations$imp6
impute.out7<-impute.out$imputations$imp7
impute.out8<-impute.out$imputations$imp8
impute.out9<-impute.out$imputations$imp9
impute.out10<-impute.out$imputations$imp10


write.csv(impute.out1, "impute1.csv")
write.csv(impute.out2, "impute2.csv")
write.csv(impute.out3, "impute3.csv")
write.csv(impute.out4, "impute4.csv")
write.csv(impute.out5, "impute5.csv")
write.csv(impute.out6, "impute6.csv")
write.csv(impute.out7, "impute7.csv")
write.csv(impute.out8, "impute8.csv")
write.csv(impute.out9, "impute9.csv")
write.csv(impute.out10, "impute10.csv")



