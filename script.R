#### load required packages ####
library(caret) # used for training and testing models
library(healthcareai) # used for partition between training set and test set using groups

library(Hmisc) # used to check the correlation among variables
library(corrplot) # used to plot the correlation plot

library(MLeval) # used to plot the AUC-ROC and AUC-PR curves and to find the cut-off to optimize the informedness in 2018

library(pdp) # used to make Partial Dependence Plots
library(ggplot2) # used to make Partial Dependence Plots

#### read the csv file containing as exemple a subset of the observations with associated indices and observed B. oleae infestation ####
data<- read.csv("data_example.csv", sep=';')

### dataset partitioning ###
data<-subset(d, d$year<2018)

## test 18-19:
pred<- subset(d, d$year>=2018)

## test 20%:
data$group<- paste(data$farm_ID, data$year, sep="_")
data$group<-as.factor(data$group)

d<-split_train_test(data, inf_presence, percent_train = 0.8, seed=1, grouping_col=group)
trainSet<-d$train
testSet<-d$test

### check correlation among bioclimatic and geographical variables ###
### for variables highly correlated we selected the one with the highest rank according to a filter approach based on ROC 
res <- rcorr(as.matrix(trainSet[,c("gdd_oli","doy","cum_gdd","s_min_avg","w_mean_avg",
			"dis_sea","s_mean_avg","w_min_avg","w_day_frost","s_day_frost","dem",
			"idr_bal_30d","cum_prec","avg_max_7d","s_prec_cum","idr_bal_4","cum_max30",
			"avg_avg_7d","w_prec_cum","avg_min_7d","cum_avg26")]), type="spearman")
P<-as.data.frame(res$P)
r<-as.data.frame(res$r)


flattenCorrMatrix <- function(cormat, pmat) {
  ut <- upper.tri(cormat)
  data.frame(
    row = rownames(cormat)[row(cormat)[ut]],
    column = rownames(cormat)[col(cormat)[ut]],
    cor  =(cormat)[ut],
    p = pmat[ut]
  )
}

corr<-flattenCorrMatrix(res$r, res$P)

## subset of the correlation matrix for variables with a Spearman’s correlation coefficient >0.9 (absolute value)
subset(corr, abs(corr$cor)>0.9)

## correlation plot
cex.before <- par("cex")
par(cex = 1)
corrplot(res$r, method="color",  
         type="upper", order="hclust", 
         #addCoef.col = "black",# Add coefficient of correlation
         tl.col="black", tl.srt=45, #Text label color and rotation
         # Combine with significance
         p.mat = res$P, sig.level = 0.01, insig = "blank", 
         # hide correlation coefficient on the principal diagonal
         diag=FALSE, tl.cex = 1,
         cl.cex = 1
)

par(cex = cex.before)


### importance of bioclimatic and geographical variables with a filter approach calculating the area under the ROC curve ###
imp<-trainSet[,c("gdd_oli","doy","cum_gdd","s_min_avg","w_mean_avg",
			"dis_sea","s_mean_avg","w_min_avg","w_day_frost","s_day_frost","dem",
			"idr_bal_30d","cum_prec","avg_max_7d","s_prec_cum","idr_bal_4","cum_max30",
			"avg_avg_7d","w_prec_cum","avg_min_7d","cum_avg26","inf_presence")]
roc_imp <- filterVarImp(x = imp[,c("gdd_oli","doy","cum_gdd","s_min_avg","w_mean_avg",
			"dis_sea","s_mean_avg","w_min_avg","w_day_frost","s_day_frost","dem",
			"idr_bal_30d","cum_prec","avg_max_7d","s_prec_cum","idr_bal_4","cum_max30",
			"avg_avg_7d","w_prec_cum","avg_min_7d","cum_avg26")], y = imp$inf_presence)

# Variables removed: 
#avg_avg_7d
#w_min_avg
#w_day_frost
#cum_gdd


### creating folds for a 10-fold cross-validation grouped by farm ID x year ###
trainSet$group<- paste(trainSet$farm_ID, trainSet$year, sep="_")
trainSet$group<-as.factor(trainSet$group)

set.seed(7)
folds <- groupKFold(trainSet$group, k = 10)



### training the algorithms ###

control <- trainControl(method="cv", number=10, classProbs = TRUE, summaryFunction = twoClassSummary, 
                        savePredictions = T, index= folds)
metric <- "ROC"


# Random Forest
set.seed(7)
fit.rf <- train(inf_presence~doy+dis_sea+dem+cum_max30+cum_avg26+avg_max_7d+avg_min_7d+cum_prec+w_mean_avg+
                  w_prec_cum+s_day_frost+s_min_avg+s_mean_avg+s_prec_cum+idr_bal_4+idr_bal_30d+gdd_oli+perc_inf
                , 
                data=trainSet, method="rf", metric=metric, trControl=control2, tuneLength = 10)

# CART
set.seed(7)
fit.cart <- train(inf_presence~doy+dis_sea+dem+cum_max30+cum_avg26+avg_max_7d+avg_min_7d+cum_prec+w_mean_avg+
                    w_prec_cum+s_day_frost+s_min_avg+s_mean_avg+s_prec_cum+idr_bal_4+idr_bal_30d+gdd_oli+perc_inf
                  ,
                  data=trainSet, method="rpart", metric=metric, trControl=control)

# kNN
set.seed(7)
fit.knn <- train(inf_presence~doy+dis_sea+dem+cum_max30+cum_avg26+avg_max_7d+avg_min_7d+cum_prec+w_mean_avg+
                   w_prec_cum+s_day_frost+s_min_avg+s_mean_avg+s_prec_cum+idr_bal_4+idr_bal_30d+gdd_oli+perc_inf
                 , 
                 data=trainSet,method="knn", metric=metric, trControl=control, 
                 preProcess = c('center', 'scale'))

# neural network
set.seed(7)
fit.nn <- train(inf_presence~doy+dis_sea+dem+cum_max30+cum_avg26+avg_max_7d+avg_min_7d+cum_prec+w_mean_avg+
                  w_prec_cum+s_day_frost+s_min_avg+s_mean_avg+s_prec_cum+idr_bal_4+idr_bal_30d+gdd_oli+perc_inf
                , 
                data=trainSet, method="nnet", metric=metric, trControl=control,
                preProcess = c('center', 'scale'))

# reg tree C5.0
set.seed(7)
fit.C5 <- train(inf_presence~doy+dis_sea+dem+cum_max30+cum_avg26+avg_max_7d+avg_min_7d+cum_prec+w_mean_avg+
                  w_prec_cum+s_day_frost+s_min_avg+s_mean_avg+s_prec_cum+idr_bal_4+idr_bal_30d+gdd_oli+perc_inf
                , 
                data=trainSet, method="C5.0", metric=metric, trControl=control, tuneLength = 20)


### selecting the best model ###
results <- resamples(list(RF=fit.rf, NN=fit.nn, "k-NN"=fit.knn, CART=fit.cart, C5.0=fit.C5))
summary(results)
dotplot(results)
bwplot(results)

bwplot(results,
       metric = "ROC")
bwplot(results,
       metric = "Sens")
bwplot(results,
       metric = "Spec")

resMleval <- evalm(list(fit.rf, fit.nn, fit.knn, fit.cart, fit.C5),gnames=c('RF','NN','k-NN','CART','C5.0'), positive = "inf")
resMleval2 <- evalm(list(fit.rf, fit.nn, fit.knn, fit.cart, fit.C5),gnames=c('RF','NN','k-NN','CART','C5.0'), positive = "no")

### results of C5.0 on test sets ###
# on test 20%
pred_C5<- predict(fit.C5, testSet) 
res_C5<-confusionMatrix(pred_C5, testSet$inf_presence, mode = "everything", positive="inf")

# on test 18-19
pred_C5_1819<- predict(fit.C5, pred)
res_C5_1819<-confusionMatrix(pred_C5_1819, pred$inf_presence, mode = "everything", positive="inf")

### extract the probability of infestation predicted by the model on the two test sets ###
# on test 20%
testSet$pred_C5<- pred_C5
pred_C5_prob<- predict(fit.C5, testSet, type = "prob")
n<-c("C5_prob_inf", "C5_prob_no")
names(pred_C5_prob)<- (n)
testSet$C5_prob_no<- pred_C5_prob$C5_prob_no
testSet$C5_prob_inf<- pred_C5_prob$C5_prob_inf

# on test 18-19
pred$pred_C5<- pred_C5_1819
pred_C5_prob<- predict(fit.C5, pred, type = "prob")
n<-c("C5_prob_inf", "C5_prob_no")
names(pred_C5_prob)<- (n)
pred$C5_prob_no<- pred_C5_prob$C5_prob_no
pred$C5_prob_inf<- pred_C5_prob$C5_prob_inf

### results on test 18-19 divided by the two years ###
#2018
pred_18<- subset(pred, pred$year==2018)
pred_18<- droplevels(pred_18)
res_18<- confusionMatrix(pred_18$pred_C5, pred_18$inf_presence, mode = "everything", positive="inf")

C5_18<-cbind.data.frame(inf=pred_18$C5_prob_inf, no=pred_18$C5_prob_no, 
                        obs=pred_18$inf_presence, Group="C5.0")

C18<-evalm(C5_18, positive="inf")
## extracting all probabilities and metrics for each cut-off value (used in the paper to identify the cut-off value which optimized the informedness)
p18<-as.data.frame(C18$probs)

#2019
pred_19<- subset(pred, pred$year==2019)
pred_19<- droplevels(pred_19)
res_19<- confusionMatrix(pred_19$pred_C5, pred_19$inf_presence, mode = "everything", positive="inf")

### results on 2018 with the cut-off value which optimized the informedness ###
cut_inf<-0.37
cut_no<-0.63

pred_18$pred_cut<-as.factor(ifelse(pred_18$C5_prob_inf>=cut_inf, "inf", "no"))
res_rf_cut<-confusionMatrix(pred_18$pred_cut, pred_18$inf_presence, mode = "everything", positive="inf")

#### Importance of variables ####
varImp(fit.C5, metric = "splits", scale=FALSE)
plot(varImp(fit.C5, metric = "splits", scale=FALSE))

#### Partial dependence plots (pdp) of the top 9 important variables (covering more than 70% of splits) ####

p <- fit.C5 %>%  # the %>% operator is read as "and then"
  partial(train = trainSet, pred.var=c("perc_inf"), type=c("classification"), prob= TRUE,
          which.class= "inf", plot.engine = "ggplot2")
P<-autoplot(p, ylab=" ") + 
  theme_light() +
  geom_hline(yintercept = 0.5, lty = 2) +
  scale_y_continuous(breaks=seq(0.0, 0.7, 0.1), limits=c(0, 0.7))

p1 <- fit.C5 %>%  # the %>% operator is read as "and then"
  partial(train = trainSet, pred.var=c("dis_sea"), type=c("classification"), prob= TRUE,
          which.class= "inf", plot.engine = "ggplot2")
P1 <-autoplot(p1, ylab=" ") + 
  theme_light() +
  geom_hline(yintercept = 0.5, lty = 2) +
  scale_y_continuous(breaks=seq(0.0, 0.7, 0.1), limits=c(0, 0.7))

p2 <- fit.C5 %>%  
  partial(train = trainSet, pred.var=c("dem"), type=c("classification"), prob= TRUE,
          which.class= "inf", plot.engine = "ggplot2")
P2<-autoplot(p2, ylab=" ") + 
  theme_light() +
  geom_hline(yintercept = 0.5, lty = 2) +
  scale_y_continuous(breaks=seq(0.0, 0.7, 0.1), limits=c(0, 0.7))

p3 <- fit.C5 %>%  
  partial(train = trainSet, pred.var=c("doy"), type=c("classification"), prob= TRUE,
          which.class= "inf",plot.engine = "ggplot2")
P3<-autoplot(p3, ylab=" ") + 
  theme_light() +
  geom_hline(yintercept = 0.5, lty = 2) +
  scale_y_continuous(breaks=seq(0.0, 0.7, 0.1), limits=c(0, 0.7))

p4 <- fit.C5 %>%  
  partial(train = trainSet, pred.var=c("gdd_oli"), type=c("classification"), prob= TRUE,
          which.class= "inf",plot.engine = "ggplot2")
P4<-autoplot(p4, ylab=" ") + 
  theme_light() +
  geom_hline(yintercept = 0.5, lty = 2) +
  scale_y_continuous(breaks=seq(0.0, 0.7, 0.1), limits=c(0, 0.7))

p5 <- fit.C5 %>% 
  partial(train = trainSet, pred.var=c("w_mean_avg"), type=c("classification"), prob= TRUE,
          which.class= "inf", plot.engine = "ggplot2")
P5<-autoplot(p5, ylab=" ") + 
  theme_light() +
  geom_hline(yintercept = 0.5, lty = 2) +
  scale_y_continuous(breaks=seq(0.0, 0.7, 0.1), limits=c(0, 0.7))

p6 <- fit.C5 %>%  
  partial(train = trainSet, pred.var=c("s_prec_cum"), type=c("classification"), prob= TRUE,
          which.class= "inf", plot.engine = "ggplot2")
P6<-autoplot(p6, ylab=" ") + 
  theme_light() +
  geom_hline(yintercept = 0.5, lty = 2) +
  scale_y_continuous(breaks=seq(0.0, 0.7, 0.1), limits=c(0, 0.7))

p7 <- fit.C5 %>% 
  partial(train = trainSet, pred.var=c("w_prec_cum"), type=c("classification"), prob= TRUE,
          which.class= "inf", plot.engine = "ggplot2")
P7<-autoplot(p7, ylab=" ") + 
  theme_light() +
  geom_hline(yintercept = 0.5, lty = 2) +
  scale_y_continuous(breaks=seq(0.0, 0.7, 0.1), limits=c(0, 0.7))

p8 <- fit.C5 %>% 
  partial(train = trainSet, pred.var=c("idr_bal_30d"), type=c("classification"), prob= TRUE,
          which.class= "inf", plot.engine = "ggplot2")
P8<-autoplot(p8, ylab=" ") + 
  theme_light() +
  geom_hline(yintercept = 0.5, lty = 2) +
  scale_y_continuous(breaks=seq(0.0, 0.7, 0.1), limits=c(0, 0.7))


#### Multi-predictor pdp (applied to pair of variables with an opposite trend in pdp) ####
trainSet$perc_inf<- as.numeric(trainSet$perc_inf)

par<- partial(fit.C5, train = trainSet, pred.var =c("dis_sea", "gdd_oli"), chull = TRUE, type=c("classification"), prob= TRUE,
              which.class= "inf")
plot <- autoplot(par, contour = TRUE, legend.title = "Probability of infestation")

par2<- partial(fit.C5, train = trainSet, pred.var =c("perc_inf", "gdd_oli"), chull = TRUE, type=c("classification"), prob= TRUE,
              which.class= "inf")
plot <- autoplot(par, contour = TRUE, legend.title = "Probability of infestation")
