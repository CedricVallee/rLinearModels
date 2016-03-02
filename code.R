rawdata = read.csv("wine.csv")
summary(rawdata)

# We scale the dataset: zero mean and unit variance
library(dplyr)
v=c("fixed.acidity","volatile.acidity","citric.acid","residual.sugar","chlorides","free.sulfur.dioxide","total.sulfur.dioxide","density","pH","sulphates","alcohol")
data <- rawdata  %>% mutate_each_(funs(scale),vars=v)
summary(data)

# A) OLS
OLSmodel=lm(quality ~ fixed.acidity + volatile.acidity + citric.acid + residual.sugar + chlorides + free.sulfur.dioxide + total.sulfur.dioxide + density + pH + sulphates + alcohol, data)
summary(OLSmodel)
OLSmse=mean(OLSmodel$residuals^2)
barplot(OLSmodel$coefficients[v],main="OLS model",ylab="Coefficients",ylim=c(-0.25,0.25),las=2)

# We create variables to use the glmnet library
library(glmnet)
X <- model.matrix(quality~., -1, data=data)
y <- data$quality

# B) RR and 10-fold cross-validation for hyperparameter tuning
fit.ridge <- glmnet(X, y, alpha=0, nlambda=10000) #default: loss to use for CV is MSE; nfolds=10
plot(fit.ridge, xvar="lambda", label=TRUE)
cv.ridge <- cv.glmnet(X,y,alpha=0)
plot(cv.ridge)
cv.ridge$lambda.min
coef(cv.ridge)
RRmse <- cv.ridge$cvm[cv.ridge$lambda == cv.ridge$lambda.min]

# C) LASSO and 10-fold cross-validation for hyperparameter tuning
fit.lasso <- glmnet(X, y, nlambda=10000) #default: loss to use for CV is MSE; mfolds=10
plot(fit.lasso, xvar="lambda", label=TRUE)
cv.lasso <- cv.glmnet(X, y)
plot(cv.lasso) 
cv.lasso$lambda.min
coef(cv.lasso)
LASSOmse <- cv.lasso$cvm[cv.lasso$lambda == cv.lasso$lambda.min]

# Calculate the norms of the estimator for each method
z <- {OLSmodel$coefficients[v]}
sum(z^2) #L2 norm
sum(abs(z)) #L1 norm

z2 <- coef(cv.ridge)[2:13]
sum(z2^2) #L2 norm
sum(abs(z2)) #L1 norm

z3 <- coef(cv.lasso)[2:13]
sum(z3^2) #L2 norm
sum(abs(z3)) #L1 norm
