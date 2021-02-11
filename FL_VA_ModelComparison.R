install.packages("R2admb")
install.packages("glmmADMB", 
                 repos=c("http://glmmadmb.r-forge.r-project.org/repos",
                         getOption("repos")),
                 type="source")

rm(list=ls())
library("PLNmodels")
library(ggplot2)
library("lme4")
library("Matrix")
library("MASS")
library("glmmADMB")
library("optimx")
library("glmmTMB")
library("pscl")

## Data import
Combined_weekday_peak_model_data_train<-read.table("E:/Post-Doc/NCHRP/NCHRP 22-48/Paper/Combined_weekday_peak_model_data_train.csv",header=T,sep=",")
Combined_weekday_peak_model_data_test<-read.table("E:/Post-Doc/NCHRP/NCHRP 22-48/Paper/Combined_weekday_peak_model_data_test.csv",header=T,sep=",")

Combined_weekday_peak_model_data_train <- Combined_weekday_peak_model_data_train[,c('total_crashes','log_seg_length','lane_number_1','lane_number_2','rural_urban','log_volume','std_speed','RoadNumber')]
Combined_weekday_peak_model_data_train$Y <- Combined_weekday_peak_model_data_train$total_crashes * 2
Combined_weekday_peak_model_data_train$obs <- 1:nrow(Combined_weekday_peak_model_data_train)
Combined_weekday_peak_model_data_train$log_year <- log(2)
Combined_weekday_peak_model_data_train$obs <- factor(Combined_weekday_peak_model_data_train$obs)
Combined_weekday_peak_model_data_train$RoadNumber <- factor(Combined_weekday_peak_model_data_train$RoadNumber)

Combined_weekday_peak_model_data_test <- Combined_weekday_peak_model_data_test[,c('total_crashes','log_seg_length','lane_number_1','lane_number_2','rural_urban','log_volume','std_speed','RoadNumber')]
Combined_weekday_peak_model_data_test$Y <- Combined_weekday_peak_model_data_test$total_crashes * 2
Combined_weekday_peak_model_data_test$obs <- 1:nrow(Combined_weekday_peak_model_data_test)
Combined_weekday_peak_model_data_test$log_year <- log(2)
Combined_weekday_peak_model_data_test$obs <- factor(Combined_weekday_peak_model_data_test$obs)
Combined_weekday_peak_model_data_test$RoadNumber <- factor(Combined_weekday_peak_model_data_test$RoadNumber)

# Combined_weekday_peak_model_data_train$lane_number_1 <- factor(Combined_weekday_peak_model_data_train$lane_number_1)
# Combined_weekday_peak_model_data_train$lane_number_2 <- factor(Combined_weekday_peak_model_data_train$lane_number_2)
# Combined_weekday_peak_model_data_train$rural_urban <- factor(Combined_weekday_peak_model_data_train$rural_urban)

################################################################################################
# Poisson Log-normal model  #########
################################################################################################
gm2 <- glmer(Y ~ offset(log_seg_length) + offset(log_year) + lane_number_1 + lane_number_2 + rural_urban + log_volume + std_speed + (1|obs),
              family = poisson, data = Combined_weekday_peak_model_data_train, control=glmerControl(optimizer="bobyqa"))

# gm2 <- glmer(Y ~ log_seg_length + lane_number_1 + lane_number_2 + rural_urban + log_volume + std_speed + (1|obs),
#              family = poisson, data = Combined_weekday_peak_model_data_train)
# 
# allFit(show.meth.tab=TRUE)
# gm_all <- allFit(gm2)
# summary(gm_all)

summary(gm2)

p3 <- predict(gm2,Combined_weekday_peak_model_data_test,re.form=NA, type = "response")

mae <- function(error){
  mean(abs(error))
}

mae(Combined_weekday_peak_model_data_test$Y - p3)/2

################################################################################################
# Poisson model  #########
################################################################################################
poissonmodel <- glm(Y ~ offset(log_seg_length) + offset(log_year) + lane_number_1 + lane_number_2 + rural_urban + log_volume + std_speed,
             family = poisson, data = Combined_weekday_peak_model_data_train)
summary(poissonmodel)
BIC(poissonmodel)
logLik(poissonmodel)

p3 <- predict(poissonmodel,Combined_weekday_peak_model_data_test, type = "response")
mae(Combined_weekday_peak_model_data_test$Y - p3)/2

################################################################################################
# Negative Bionomial model  #########
################################################################################################
# nbmodel <- glm.nb(Y ~ offset(log_seg_length) + offset(log_year) + lane_number_1 + lane_number_2 + rural_urban + log_volume + std_speed,
#                     data = Combined_weekday_peak_model_data_train)
nbmodel <- glm(Y ~ offset(log_seg_length) + offset(log_year) + lane_number_1 + lane_number_2 + rural_urban + log_volume + std_speed,
                    family = negative.binomial(2), data = Combined_weekday_peak_model_data_train)

summary(nbmodel)
BIC(nbmodel)
logLik(nbmodel)

p3 <- predict(nbmodel,Combined_weekday_peak_model_data_test, type = "response")
mae(Combined_weekday_peak_model_data_test$Y - p3)/2


################################################################################################
# Random Effect Negative Bionomial model  #########
################################################################################################
gm2 <- glmer.nb(Y ~ offset(log_seg_length) + offset(log_year) + lane_number_1 + lane_number_2 + rural_urban + log_volume + std_speed + (1|RoadNumber),
             data = Combined_weekday_peak_model_data_train, control=glmerControl(optimizer="bobyqa"))

summary(gm2)

p3 <- predict(gm2,Combined_weekday_peak_model_data_test, type = "response", re.form= ~(1|RoadNumber))
mae(Combined_weekday_peak_model_data_test$Y - p3)/2


################################################################################################
# Random Effect Poisson Log-normal model  #########
################################################################################################
gm2 <- glmer(Y ~ offset(log_seg_length) + offset(log_year) + lane_number_1 + lane_number_2 + rural_urban + log_volume + std_speed + (1|obs) + (1|RoadNumber),
             family = poisson, data = Combined_weekday_peak_model_data_train)


summary(gm2)

p3 <- predict(gm2,Combined_weekday_peak_model_data_test, type = "response", re.form= ~(1|RoadNumber))
mae(Combined_weekday_peak_model_data_test$Y - p3)/2


################################################################################################
# Zero-Inflated Poisson Log-normal model  #########
################################################################################################
fit_zipoisson <- glmmTMB(Y ~ offset(log_seg_length) + offset(log_year) + lane_number_1 + lane_number_2 + rural_urban + log_volume + std_speed + (1|obs),
                         data=Combined_weekday_peak_model_data_train,
                         ziformula=~1 + rural_urban + log_volume,
                         family=poisson)
summary(fit_zipoisson)
p3 <- predict(fit_zipoisson,Combined_weekday_peak_model_data_test,re.form=NA, type = "response")
mae(Combined_weekday_peak_model_data_test$Y - p3)/2


################################################################################################
# Random Effect Zero-Inflated Poisson Log-normal model  #########
################################################################################################
fit_zipoisson <- glmmTMB(Y ~ offset(log_seg_length) + offset(log_year) + lane_number_1 + lane_number_2 + rural_urban + log_volume + std_speed + (1|obs) + (1|RoadNumber),
                         data=Combined_weekday_peak_model_data_train,
                         ziformula=~1,
                         family=poisson)
summary(fit_zipoisson)
p3 <- predict(fit_zipoisson,Combined_weekday_peak_model_data_test,re.form=NA, type = "response")
mae(Combined_weekday_peak_model_data_test$Y - p3)/2

################################################################################################
# Zero-Inflated Poisson model  #########
################################################################################################
m1 <- zeroinfl(Y ~ offset(log_seg_length) + offset(log_year) + lane_number_1 + lane_number_2 + rural_urban + log_volume + std_speed, 
               data = Combined_weekday_peak_model_data_train)

summary(m1)
p3 <- predict(m1,Combined_weekday_peak_model_data_test, type = "response")
mae(Combined_weekday_peak_model_data_test$Y - p3)/2


################################################################################################
# Test  #########
################################################################################################

poismod <- glmmadmb(Y ~ log_seg_length + lane_number_1 + lane_number_2 + rural_urban + log_volume + std_speed + (1|obs), family = "poisson", data = Combined_weekday_peak_model_data_train)
summary(poismod)



data(trichoptera)
trichoptera <- prepare_data(trichoptera$Abundance, trichoptera$Covariate)
trichoptera$Y <- trichoptera$Abundance
myPLN <- PLN(Offset ~ 1, data = trichoptera)
myPLN
varNames<-names(Combined_weekday_peak_model_data_train)
