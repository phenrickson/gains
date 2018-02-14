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
library(foreach)
library(ggplot2)
library(grid)
library(gridExtra)
library(readr)
options(scipen=7)
options(digits=3)

##########################
## Read in Imputed Data ##
##########################


imputations <- foreach(i=1:10) %do% {
  
  a <- read_csv(paste("/Backup/Dropbox/Machine Learning Paper/impute", i, ".csv", sep = ""))
  
  a
  
}


#################################
## Iterate Through Imputations ##
## And Merge Data For Each Imp ##
#################################

for(i in 1:10){
  
  impute.dat <- as.data.frame(imputations[i])
  
  # read in dataset #
  
  dv.dat <- read.dta("./Data/2017-5-24 perks of being a lawmaker.dta") 
  
  
  #######################################################
  ## First, Run Inverse Hyperbolic Sine Transformation ##
  #######################################################
  
  # IHS Transformation Function
  
  IHS <- function(x, theta){
    log(theta * x + sqrt((theta * x)^2 + 1))
  }
  
  # KS Function
  ks.test.stat <- function(x, theta){ 
    newvar <- IHS(x, theta) 
    
    ks_stat <- function(x) {
      x <- x[!is.na(x)]
      ans <- ecdf(x)(x) - pnorm(x, mean = mean(x), sd = sd(x))
      max(abs(ans))
    }
    
    ks_stat(newvar)
  }
  
  set.seed(1999)
  
  theta<-optimize(ks.test.stat, lower=0, upper=2^10, x=dv.dat$income01, maximum=F)$minimum
  
  dv.dat$hisinc<-IHS(dv.dat$income01, theta)
  
  # hist(dv.dat$hisinc, breaks = 80, col = "blue") # test hisinc distribution
  
  
  ########################
  ## Restrict dv.dat    ##
  ## Only caseid and DV ##
  ########################
  
  dv.dat2 <- dv.dat[,c("caseid", "hisinc", "networth01")]
  
  
  ###########################
  ## Create Merged Dataset ##
  ###########################
  
  impute.dat$caseid <- (impute.dat$memberid * 10000) + impute.dat$year
  
  
  merged.dat <- merge(impute.dat, dv.dat2, by.x = "caseid", by.y = "caseid", all.x = T)
  
  merged.dat <- na.omit(merged.dat)
  
  
  #########################
  ## Export Datasets     ##
  ## For Each Imputation ##
  #########################
  
  save(merged.dat, file = paste("merged.dat", i, ".Rdata", sep = ""))
  
}















#########################
## END MERGING PROCESS ##
#########################



##



##

par(mfrow=c(4,4))

for(i in 1:ncol(merged.dat)){
  d<-density(na.omit(merged.dat[,i]))
  plot(d, main=paste(colnames(merged.dat[i])))
}

par(mfrow=c(1,1))

C1 <- cor(merged.dat)
corrplot(C1)


# scatter plots for all IVs

vis.dat<-dplyr::select(merged.dat, -caseid, -memberid, -X, -district, -hisinc)
vis.names<-print(colnames(vis.dat), quote=F)


par(mfrow=c(4,4))

for (i in 1:ncol(vis.dat)) {
  
  plot(merged.dat$hisinc~vis.dat[,i], xlab=paste(vis.names[i]))

}



#######################################
## Examine Scatterplots of Variables ##
## To look for Correlations          ##
#######################################

merged.dat.2 <- dplyr::select(merged.dat, -X, -caseid, -memberid)
merged.dat.2 <- merged.dat.2[, c(59, 1:58, 60)]

cor.table <- cor(merged.dat.2, use = "pairwise.complete.obs")


par(mfrow = c(1,1))

corrplot(cor.table, tl.cex = 0.6)

 
p.table <- matrix(NA, nrow = 60, ncol = 60)

for(i in 1:60){
  
  for(j in 1:60){
    
    p.table[i,j] <- cor.test(merged.dat.2[,i], merged.dat.2[,j])$p.value
    
  }
  
}


