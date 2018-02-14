## Post-Imputation Lag Creation ##


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
library(DataCombine)
options(scipen=7)
options(digits=3)

# load in data

#load("merged.dat.Rdata")


dat <- read.csv("/Backup/Dropbox/Machine Learning Paper/Data/impute1.csv")


#############
# make lags #
#############

####################
## create counter ##
####################

dat$counter <- with(dat, ave(memberid, memberid, FUN = seq_along))

###################################
## Use Function to Generate Lags ##
###################################

do_lag <- function(the_data, variables, num_periods) {
  num_vars <- length(variables)
  num_rows <- nrow(the_data)
  
  for (j in 1:num_vars) {
    for (i in 1:num_periods) {
      the_data[[paste0(variables[j], i)]] <- c(rep(NA, i), head(the_data[[variables[j]]], num_rows - i))
    }
  }
  
  return(the_data)
}


########################
## Lag Some Variables ##
########################

lag.dat <- do_lag(dat, 
                  variables = c("gdp01", "taxreturn", 
                                "numberofjobs", "leadership",
                                "chair", "rules", 
                                "approp", "fintax",
                                "agriculture", "judiciary",
                                "education", "health", 
                                "effectivescore", "speaker",
                                "majld", "majpty", "ethicsviol",
                                "ethicscmte", "votemarg")
                  , num_periods = 1)


####################################
## Now remove the counter = 1 obs ##
####################################

for(i in 63:82){
  
  lag.dat[,i] <- ifelse(lag.dat$counter == 1, lag.dat[,i] == NA, lag.dat[,i])
  
}

# save datasets with lag structures here


save(lag.dat, file="lagged dataset.RData")


##################################
## Create Differenced Variables ##
##################################

lag.dat$gdpdif <- lag.dat$gdp01 - lag.dat$gdp011

lag.dat$taxreturndif <- lag.dat$taxreturn - lag.dat$taxreturn1
lag.dat$numberofjobsdif <- lag.dat$numberofjobs - lag.dat$numberofjobs1
lag.dat$leadershipdif <- lag.dat$leadership - lag.dat$leadership1
lag.dat$chairdif <- lag.dat$chair - lag.dat$chair1
lag.dat$rulesdif <- lag.dat$rules - lag.dat$rules1
lag.dat$appropdif <- lag.dat$approp - lag.dat$approp1
lag.dat$fintaxdif <- lag.dat$fintax - lag.dat$fintax1
lag.dat$agriculturedif <- lag.dat$agriculture - lag.dat$agriculture1
lag.dat$judiciarydif <- lag.dat$judiciary - lag.dat$judiciary1
lag.dat$educationdif <- lag.dat$education - lag.dat$education1
lag.dat$healthdif <- lag.dat$health - lag.dat$health1
lag.dat$effectivescoredif <- lag.dat$effectivescore - lag.dat$effectivescore1
lag.dat$speakerdif <- lag.dat$speaker - lag.dat$speaker1
lag.dat$majlddif <- lag.dat$majld - lag.dat$majld1
lag.dat$majptydif <- lag.dat$majpty - lag.dat$majpty1
lag.dat$ethicsvioldif <- lag.dat$ethicsviol - lag.dat$ethicsviol1
lag.dat$ethicscmtedif <- lag.dat$ethicscmte - lag.dat$ethicscmte1
lag.dat$votemargdif <- lag.dat$votemarg - lag.dat$votemarg1
 


###################################
# save datasets with lag          #
# and differenced structures here #
###################################


save(lag.dat, file="lagged and differenced dataset.RData")


###############################
## Merge with Income Dataset ##
###############################

###########################
## Read in Original Data ##
###########################

dv.dat <- read.dta("B:/Dropbox/Machine Learning Paper/Data/2017-5-24 perks of being a lawmaker.dta")
dv.dat <- read.dta("/Backup/Dropbox/Machine Learning Paper/Data/2017-5-24 perks of being a lawmaker.dta")


#######################################################
## First, Run Inverse Hyperbolic Sine Transformation ##
#######################################################

# IHS Transformation
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

hist(dv.dat$hisinc, breaks = 80, col = "blue")


########################
## Only caseid and DV ##
########################

dv.dat2 <- dv.dat[,c("caseid", "hisinc", "networth01")]


###########################
## Create Merged Dataset ##
###########################

lag.dat$caseid <- (lag.dat$memberid *10000) + lag.dat$year

final.dat <- merge(dv.dat2, lag.dat, by.x = "caseid", by.y = "caseid", all.x = T)


########################################################
## Remove all obs where hisinc and networth01 are NAs ##
########################################################

final.dat <- final.dat[!(is.na(final.dat$hisinc) | is.na(final.dat$networth01)), ] # from 2905 to 2280 obs #

#################################
## Merge hisinc and networth01 ##
#################################

########################
## Lag Some Variables ##
########################

final.dat.2 <- do_lag(final.dat, 
                  variables = c("hisinc", "networth01")
                  , num_periods = 1)


####################################
## Now remove the counter = 1 obs ##
####################################

for(i in 103:104){
  
  final.dat.2[,i] <- ifelse(final.dat.2$counter == 1, final.dat.2[,i] == NA, final.dat.2[,i])
  
}


##################################
## Create Differenced Variables ##
##################################

final.dat.2$networthdif <- final.dat.2$networth01 - final.dat.2$networth011 
final.dat.2$hisincdif <- final.dat.2$hisinc - final.dat.2$hisinc1

save(final.dat.2, file = "2017-10-12 final.RData")
