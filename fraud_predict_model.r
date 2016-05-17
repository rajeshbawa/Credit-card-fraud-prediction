#######################################################################
#######################Capital one datascience Challenge###############
#####Answer1####################
####packages you will need for this model are
require(ggplot2)
require(reshape)
require(leaps)
require(car)
require(qpcR)
require(MASS)
require(rpart)
require(rpart.plot)
require(randomForest)
require(caret)
require(robustbase)
setwd("~/Documents/CO_datascience/model_question")
train.data <- read.table("codetest_train.txt", header = T, sep="\t")
train.data[is.na(train.data)] <- "0" ###setting all NA to zero
#####on visual inspection, there are four columns with categorical data in it#####
####we will remove those for now to get correlation estimate####################
train.num.data <- train.data[,-c(63,123,217,239)]
####converting rest of the data to numeric
train.num.data <- sapply(train.num.data, as.numeric)
data.train1 <- cbind(train.num.data, train.data[,c(63,123,217,239)])
cor.train.num <- cor(train.num.data) #correlation between predictors and response variabes
#######build a heatmap to visualize the correlation between variables
m <-melt(cor.train.num)
jpeg("heatmap_correlation.jpg", height = 600, width = 600, quality = 100)
p <- ggplot(data=m, aes(x=X1, y=X2, fill=value)) + geom_tile()
plot(p)
dev.off()
###########################
#set up a coloring scheme using colorRampPalette
red=rgb(1,0,0); green=rgb(0,1,0); blue=rgb(0,0,1); white=rgb(1,1,1)
RtoWrange<-colorRampPalette(c(red, white ) )
WtoGrange<-colorRampPalette(c(white, green) ) 
p <- p + scale_fill_gradient2(low=RtoWrange(100), mid=WtoGrange(100), high="gray")
plot(p)
##############################
########## f_218 f_75, f_205 f_195, f_175 f_161, f_47 f_35, f_169 f_94, 
#####so correlation is mostly not a big issue in this data set######
######now lets run the full model first and look at the residuals
fullmodel.1 <- lm(target~. , data=data.train1, qr=T)
summary(fullmodel.1)
anova(fullmodel.1)
stack.diag1 <- ls.diag(fullmodel.1)
names(stack.diag1)
par(mfrow=c(1,2))
plot(fullmodel.1$fitted.values,stack.diag1$stud.res, ylab="Externally studentized residuals", xlab = "Fitted Values", main = "Residual vs Fitted", pch = 21, cex=0.5, col = "red")
abline(h =0, untf = FALSE, col="blue")
qqnorm(stack.diag1$stud.res, col="red", ylab = "Externally studentized residuals", xlab = "Theoretical quantiles", main = "Normal probability plot", cex=0.5)
qqline(stack.diag1$stud.res, col="blue", cex=3)
par(mfrow=c(2,2))
plot(fullmodel.1)
##################basically tells us that the full linear model is horrible
#####the externally standardized residuals are very wide and residuals deviate from normality
#####################################################
######run general transformation (log) of response and see the results
####adding a constant to the response and then doing the log transformation
new.target <- unlist(lapply(data.train1$target, function(x,y) log(x+y), y=30))
new.data.train <- cbind(new.target, data.train1)
new.data.train.1 <- new.data.train[, -2]
fullmodel.2 <- lm(new.target~., data=new.data.train.1, qr=T)
summary(fullmodel.2)
anova(fullmodel.2)
stack.diag2 <- ls.diag(fullmodel.2)
names(stack.diag2)
######
par(mfrow=c(1,2))
plot(fullmodel.2$fitted.values,stack.diag2$stud.res, ylab="Externally studentized residuals", xlab = "Fitted Values", main = "Residual vs Fitted", pch = 21, cex=0.5, col = "red")
abline(h =0, untf = FALSE, col="blue")
qqnorm(stack.diag2$stud.res, col="red", ylab = "Externally studentized residuals", xlab = "Theoretical quantiles", main = "Normal probability plot", cex=0.5)
qqline(stack.diag1$stud.res, col="blue", cex=3)
par(mfrow=c(2,2))
plot(fullmodel.2)
##################
####looking at the results we can say that just transforming the response doesn't do much
####we need a more rigrous criteria, possibly variable selection, followed by trying
###Linear, non-linear models
####################
###lets look at problematic data points in the above models
d1.fullmodel1 <- cooks.distance(fullmodel.1)
res.fullmodel1 <- rstandard(fullmodel.1)
dffits.fullmodel1 <- dffits(fullmodel.1)
influence.fullmodel1 <- as.data.frame(cbind(data.train1$target,d1.fullmodel1, res.fullmodel1, dffits.fullmodel1))
head(influence.fullmodel1[order(influence.fullmodel1$d1.fullmodel1, decreasing = T),], 10)
#####problematic points of log response model
d1.fullmodel2 <- cooks.distance(fullmodel.2)
res.fullmodel2 <- rstandard(fullmodel.2)
dffits.fullmodel2 <- dffits(fullmodel.2)
influence.fullmodel2 <- as.data.frame(cbind(new.data.train$new.target,d1.fullmodel2, res.fullmodel2, dffits.fullmodel2))
head(influence.fullmodel2[order(influence.fullmodel2$d1.fullmodel2, decreasing = T),], 10)
###############looking at the problematic data points and residual plots
####the transformation did nothing
##########################
#######Lets do variable selection#####
####all possible regression#####
###so for building the model
leaps.test1 <- regsubsets(target~., data=data.train1, nbest=10, really.big = T, method="forward")
leaps.summary <- summary(leaps.test1)
names(leaps.summary)
all.possible.model <- leaps.summary$which ###this contains all the best models recommended by leaps package
stats.all.possible.model <- cbind(leaps.summary$adjr2, leaps.summary$cp, leaps.summary$bic) ##stats of the corresponding models
colnames(stats.all.possible.model) <- c("AdjstR2", "Cp", "Bic")
jpeg("allpossible_regression.jpg", width = 1000, height = 800, quality = 100)
plot(leaps.test1, labels=leaps.test1$xnames, main=NULL, scale=c("bic", "Cp", "adjr2", "r2"), col=gray(seq(0, 0.9, length(leaps.test1$xnames))))
dev.off()
#######Looking at the stats of these models, I find three models, two with 7 variables and one with six var. the best ones
###these models were f_35, f_94, f_175, f_205, f_218, f_61, f_237
###second best f_35, f_169, f_175, f_205, f_218, f_61, f_237
###third best f_35, f_175, f_205, f_218, f_61, f_237
###Backward
#step1 <- step(lm(target~., data=data.train1), direction="backward") ###disabled because it takes a long time
#step1$anova
###backwards stepwise regression recommendes a very big model, but
####look at step$call and step$coefficents to get variables with largest coefficents.
###this was same as 1st model of all possible regression
####Forward
cnames <- colnames(data.train1[, -1])
var1 <- as.formula(paste(" ~ ",paste(cnames,collapse="+")))
step2 <- step(lm(target~1, data=data.train1), direction="forward", scope=var1)
step2$anova
####look at AIC changes in step$anova, based on that, it gives same model as 1st model of all
##possible regression
###########################
######other variable selection methods###
####Regression tree based approach
###based on complete data
tree=rpart(target~.,data=data.train1)
jpeg("regression_tree.jpg", width = 300, height = 300, quality = 100)
rpart.plot(tree)
dev.off()
#########
##three variables come out very important and these are
#####f_175, f_205, f_61
##using randomforest as one of the measures
###based on first 500 observations
rf=randomForest(target~., data=data.train1[c(1:500),], ntree= 5000, nodesize=5, importance=TRUE)
rf.importance <- as.data.frame(rf$importance)
jpeg("important_var.jpg", width = 500, height = 500, quality = 100)
#varImp(rf)
varImpPlot(rf,type=2)
dev.off()
#######it gives 2 more variables including the ones given in regression tree to be imp
### these variables are f_175, f_205, f_61, f_218, f_161
#############
##########
############
###based on all these selection methods, we pick up four models to test 
####first model
data.var.model1 <- lm(target~f_35 + f_175 + f_205 + f_218 + f_61 + f_237 + f_218*f_205, data = data.train1, qr=T)
summary(data.var.model1)
anova(data.var.model1) ###all the variables are significant here
####assumptions
stack.diag.1.1 <- ls.diag(data.var.model1)
names(stack.diag.1.1)
par(mfrow=c(1,2))
plot(data.var.model1$fitted.values,stack.diag.1.1$stud.res, ylab="Externally studentized residuals", xlab = "Fitted Values", main = "Residual vs Fitted", pch = 21, cex=0.5, col="red")
abline(h =0, untf = FALSE, col="blue")
qqnorm(stack.diag.1.1$stud.res, col="red", ylab = "Externally studentized residuals", xlab = "Theoretical quantiles", main = "Normal probability plot", cex=0.5)
qqline(stack.diag.1.1$stud.res, col="blue", cex=3)
par(mfrow=c(2,2))
plot(data.var.model1)
#######residuals don't look good at all
###Influence measures
vif.model1 <- vif(data.var.model1)
#press.model1 <- PRESS(data.var.model1)$stat ##press statistic
############looking at these two measures
####model2
data.var.model2 <- lm(target~f_35 + f_169 + f_175 + f_205 + f_218 + f_61 + f_237 + f_218*f_205, data = data.train1, qr=T)
summary(data.var.model2)
anova(data.var.model2) ###all the variables are significant here
####residuals
stack.diag.1.2 <- ls.diag(data.var.model2)
names(stack.diag.1.2)
par(mfrow=c(1,2))
plot(data.var.model2$fitted.values,stack.diag.1.2$stud.res, ylab="Externally studentized residuals", xlab = "Fitted Values", main = "Residual vs Fitted", pch = 21, cex=0.5, col="red")
abline(h =0, untf = FALSE, col="blue")
qqnorm(stack.diag.1.2$stud.res, col="red", ylab = "Externally studentized residuals", xlab = "Theoretical quantiles", main = "Normal probability plot", cex=0.5)
qqline(stack.diag.1.2$stud.res, col="blue", cex=3)
par(mfrow=c(2,2))
plot(data.var.model2)
######No change in residuals
###influence measures
vif.model2 <- vif(data.var.model2)
#press.model2 <- PRESS(data.var.model2)$stat
###################
##############Model3
data.var.model3 <- lm(target~f_35 + f_175 + f_205 + f_218 + f_61 + f_237 + f_218*f_205, data = data.train1, qr=T)
summary(data.var.model3)
anova(data.var.model3) ###all the variables are significant here
####residuals
stack.diag.1.3 <- ls.diag(data.var.model3)
names(stack.diag.1.3)
par(mfrow=c(1,2))
plot(data.var.model3$fitted.values,stack.diag.1.3$stud.res, ylab="Externally studentized residuals", xlab = "Fitted Values", main = "Residual vs Fitted", pch = 21, cex=0.5, col="red")
abline(h =0, untf = FALSE, col="blue")
qqnorm(stack.diag.1.3$stud.res, col="red", ylab = "Externally studentized residuals", xlab = "Theoretical quantiles", main = "Normal probability plot", cex=0.5)
qqline(stack.diag.1.3$stud.res, col="blue", cex=3)
par(mfrow=c(2,2))
plot(data.var.model3)
#####problem with the residuals
###Influence measures
vif.model3 <- vif(data.var.model3)
#press.model3 <- PRESS(data.var.model3)$stat
###############################
######Model4####
data.var.model4 <- lm(target~f_175 + f_205 + f_61, data = data.train1, qr=T)
summary(data.var.model4)
anova(data.var.model4) ###all the variables are significant here
##residuals
stack.diag.1.4 <- ls.diag(data.var.model4)
names(stack.diag.1.4)
par(mfrow=c(1,2))
plot(data.var.model4$fitted.values,stack.diag.1.4$stud.res, ylab="Externally studentized residuals", xlab = "Fitted Values", main = "Residual vs Fitted", pch = 21, cex=0.5, col="red")
abline(h =0, untf = FALSE, col="blue")
qqnorm(stack.diag.1.4$stud.res, col="red", ylab = "Externally studentized residuals", xlab = "Theoretical quantiles", main = "Normal probability plot", cex=0.5)
qqline(stack.diag.1.4$stud.res, col="blue", cex=3)
par(mfrow=c(2,2))
plot(data.var.model4)
#####influence measures
vif.model4 <- vif(data.var.model4)
#press.model4 <- PRESS(data.var.model4)$stat
#####just looking at ajusted R2 the model1 looks best. Adding more variables doesn't change adjustedR2 much and removing affects it
#####but the standard errors are high and the assumptions are not met
#########################################################
####Using robust regression
model1.robust <- rlm(target~f_35 + f_175 + f_205 + f_218 + f_61 + f_237 + f_218*f_205, data = data.train1)
summary(model1.robust)
anova(model1.robust)
plot(model1.robust) ##this doesn't work either (looking at the residuals)
######since normality is still a problem, lets look at glmrobust
####this is robust for non normal condition and heteroscedascity can be dealt with IRLS procedure
summary(model.glmrobust <- glmrob(target ~ f_35 + f_175 + f_205 + f_218 + f_61 + f_237, family = gaussian,
                                  data = data.train1, method= "Mqle", weights.on.x = "hat"))
######standard errors are considerably reduced
#######this model looks good
#########################
####predict the values#######################
testdata1 <- read.table("codetest_test.txt", header = T, sep="\t")
answer1.list <- as.matrix(predict(model.glmrobust, testdata1, type="response", se = F))
write.table(answer1.list, file="ans1_predict.txt", col.names = F, row.names = F, eol="\n", quote = F) ###write it to the file
