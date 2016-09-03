#
# EE622: Advanced Machine Learning
#        Prof. Amit Sethi
#        Assignment 1: HDLSS Lung Cancer Data Analysis
#
# Name: Duddu Sai Meher Karthik
# Roll No. : 130101084
#
#

library(Biobase)
library(GEOquery)

library(glmnet) #for models + regularization
library(mice) #for data imputation
library(lattice) #for 1D cluster visualization
library(dbscan) #for cluster analysis
library(caret) #confusion matrix calculation
library(proc) #ROC/AUC comparison (NOTE: couldn't complete this analysis)

gds2771 <- getGEO(filename='B:/Acad/Course Material/Semester 7/EE622/Assignment1/GDS2771.soft.gz') # Make sure path is correct as per your working folder. Could be './GDS2771.soft.gz'
eset2771 <- GDS2eSet(gds2771) # See http://www2.warwick.ac.uk/fac/sci/moac/people/students/peter_cock/r/geo/
data2771 <- cbind2(c('disease.state',pData(eset2771)$disease.state),t(Table(gds2771)[,2:194]))
colnames(data2771) = data2771[1, ] # the first row will be the header
data2771 = data2771[-1, ] 

# WRITE YOUR CODE BELOW THIS LINE

df <- as.data.frame(data2771)
df <- subset(df, df[,1] != "3") # Remove suspected cancer cases from data - 5 in number
df <- as.data.frame(apply(df, 2, function(x) as.numeric(x))) # Convert columns to numeric

#Handling missing data
# Step 1: Eliminate all genes which contain missing values for all samples
df <- df[, colSums(is.na(df)) < nrow(df)]
# Step 2: Impute values for all genes which contain partial missing values (using mean)
# Note: This dataset doesn't require this step, as after step 1, all na values disappear. Method kept for completeness
df <- if (sum(is.na(df)) > 0) mice(df, method = "mean") else df # Final number of genes = 22216

#Dividing dataset into testing and training sets
SPLIT.PERCENTAGE = 0.75 # 75-25 training-to-testing ratio
train.size = floor(nrow(df)*SPLIT.PERCENTAGE)
set.seed(101) #to ensure reproducibility
train.idx <- sample(nrow(df), size = train.size)
train_x <- as.matrix(df[train.idx, -1]) #The training dataset features
train_y <- df[train.idx, 1] #The training dataset annotation vector
test_x <- as.matrix(df[-train.idx, -1]) #The testing dataset features
test_y <- df[-train.idx, 1] #The testing dataset annotation vector


#Error computation function
get_error <- function(pred_y, test_y) {
  return (mean(pred_y != test_y))
}

#Experiment 1: Linear regression with Lasso and 10-fold cross-validation (mean-squared error)
#Note: All features normalized (standardized) before training, and rescaled before prediction
cvfit_e1 = cv.glmnet(x=train_x, y=train_y, nfolds=10, family="gaussian", type.measure="mse", alpha=1, standardize=TRUE)
pred_y <- predict(cvfit_e1, newx=test_x, s="lambda.min") #Choosing knee-point lambda value
#   Applying thresholding to the predicted values for comparison purposes (threshold - 1.5 for 2-class)
REGRESSION.THRESHOLD = 1.5
pred_y <- lapply(pred_y, function(x) {if (x > REGRESSION.THRESHOLD) 2 else 1} )
#   Deliverables
#   1. Accuracy in terms of error rate
err_per1 <- get_error(pred_y, test_y)
#   2. Graph of cross-validation of lambda and min lambda value
#    plot(cvfit_e1)
#    cvfit_e1$lambda.min
#   3. Number of non-zero coefficients
coef_e1 <- abs(coef(cvfit_e1, s="lambda.min"))
nz_idx <- which(coef_e1 != 0)
#   length(nz_idx)
#   plot(cvfit_e3$glmnet.fit$lambda, cvfit_e3$glmnet.fit$df, type="o", col="red", ylab="Number of non-zero coefficients", xlab="Lambda")
#   4. Top 10 important genes
nz_wt_sorted <- sort(coef_e1[nz_idx], decreasing = TRUE, index.return=TRUE)
nz_genes_e1 <- rownames(coef_e1)[nz_idx][nz_wt_sorted$ix[nz_wt_sorted$ix != 1]] #Exclude the 'Intercept' field
#   nz_genes_e1[1:10]


#Experiment 2: Linear regression with Ridge 
#Note: All features normalized (standardized) before training, and rescaled before prediction
cvfit_e2 = cv.glmnet(x=train_x, y=train_y, nfolds=10, type.measure="mse", family="gaussian", alpha=0, standardize=TRUE)
pred_y <- predict(cvfit_e2, newx=test_x, s="lambda.min") #Choosing knee-point lambda value
#   Applying thresholding to the predicted values for comparison purposes (threshold - 1.5 for 2-class)
REGRESSION.THRESHOLD = 1.5
pred_y <- lapply(pred_y, function(x) {if (x > REGRESSION.THRESHOLD) 2 else 1} )
#   Deliverables
#   1. Accuracy in terms of error rate
err_per2 <- get_error(pred_y, test_y)
#   2. Graph of cross-validation of lambda and min lambda value
#    plot(cvfit_e2)
#    cvfit_e2$lambda.min
#   3. Grouping effect in ridge
coef_e2 <- abs(coef(cvfit_e2, s="lambda.min"))
coefe <- as.matrix(scale(coef_e2[-1])) #To remove intercept and scale coefficients
cl <- dbscan::dbscan(coefe, eps=0.25, minPts=5)
stripplot(coefe, col="red", xlab="Grouping of coefficients in Ridge (Linear)")
nz_idx <- which(coef_e2 != 0)
#   length(nz_idx)
#   4. Top 10 important genes
nz_wt_sorted <- sort(coef_e2[nz_idx], decreasing = TRUE, index.return=TRUE)
nz_genes_e2 <- rownames(coef_e2)[nz_idx][nz_wt_sorted$ix[nz_wt_sorted$ix != 1]] #Exclude the 'Intercept' field
#   nz_genes_e2[1:10]
#   5. Cluster Analysis
coefe <- as.matrix(coef_e2[-1])


#Experiment 3: Choosing value of mixing parameter(alpha) of elasticnet using cross-validation
a <- matrix(NA,51,3)
j=1
i=0
for(i in seq(0,1,by=0.02))
{
  cvfit_e2 = cv.glmnet(x=train_x, y=train_y, nfolds=10, type.measure="mse", family="gaussian", alpha=i, standardize=TRUE)
  a[j,1] <- i
  a[j,2] <- cvfit_e2$lambda.min
  a[j,3] <- min(cvfit_e2$cvm)
  j=j+1
}
colnames(a) <- c("alpha","lambda_min","crossval_mean_error")
df_1 <- data.frame(a)
plot(crossval_mean_error ~ alpha,df_1,type="l")
cvm_min <- min(a[,"crossval_mean_error"])
index = which(a[,"crossval_mean_error"]==cvm_min)
alphamin = df_1[index,"alpha"]
# from the plot, the elastic net parameters are alphamin= 0.04
# applying the elastic net with 10 folds for the alphamin
cvfit_enet = cv.glmnet(x=train_x, y=train_y, nfolds=10, type.measure="mse", family="gaussian", alpha=alphamin, standardize=TRUE)
pred_y <- predict(cvfit_enet, newx=test_x, s="lambda.min") #Choosing knee-point lambda value
#   Applying thresholding to the predicted values for comparison purposes (threshold - 1.5 for 2-class)
REGRESSION.THRESHOLD = 1.5
pred_y <- lapply(pred_y, function(x) {if (x > REGRESSION.THRESHOLD) 2 else 1} )
#   Deliverables
#   1. Accuracy in terms of error rate
err_per_enet <- get_error(pred_y, test_y)
#   2. Graph of cross-validation of lambda
#    plot(cvfit_e2)
#    cvfit_e2$lambda.min
#   3. Number of non-zero coefficients
coef_enet <- abs(coef(cvfit_enet, s="lambda.min"))
nz_idx <- which(coef_enet != 0)
#   length(nz_idx)
#    plot(cvfit_enet$glmnet.fit$lambda, cvfit_enet$glmnet.fit$df, type="o", col="red", ylab="Number of non-zero coefficients", xlab="Lambda")
#   4. Grouping effect elasticnet
coefe <- as.matrix(scale(coef_enet[-1])) #To remove intercept and scale coefficients
cl <- dbscan::dbscan(coefe, eps=0.30, minPts=5)
stripplot(coefe, col="red", xlab="Grouping of coefficients in Ridge (Linear)")
#   5. Top 10 important genes
nz_wt_sorted <- sort(coef_enet[nz_idx], decreasing = TRUE, index.return=TRUE)
nz_genes_enet <- rownames(coef_enet)[nz_idx][nz_wt_sorted$ix[nz_wt_sorted$ix != 1]] #Exclude the 'Intercept' field  
#    nz_genes_enet[1:10]  


#Experiment 4: Logistic regression with Lasso
#Note: All features normalized (standardized) before training, and rescaled before prediction
cvfit_e3 = cv.glmnet(x=train_x, y=train_y, nfolds=10, family="binomial", type.measure="class", alpha=1, standardize=TRUE)
pred_y <- predict(cvfit_e3, newx=test_x, s="lambda.min", type="class") #Choosing knee-point lambda value
#   Deliverables
#   1. Accuracy in terms of error rate
err_per3 <- get_error(pred_y, test_y)
#   2. Graph of cross-validation of lambda and min lambda value
#    plot(cvfit_e3)
#    cvfit_e3$lambda.min
#   3. Variation in non-zero coefficients
coef_e3 <- abs(coef(cvfit_e3, s="lambda.min"))
nz_idx <- which(coef_e3 != 0)
#   length(nz_idx)
#    plot(cvfit_e3$glmnet.fit$lambda, cvfit_e3$glmnet.fit$df, type="o", col="red", ylab="Number of non-zero coefficients", xlab="Lambda")
#   4. Top 10 important genes
nz_wt_sorted <- sort(coef_e3[nz_idx], decreasing = TRUE, index.return=TRUE)
nz_genes_e3 <- rownames(coef_e3)[nz_idx][nz_wt_sorted$ix[nz_wt_sorted$ix != 1]] #Exclude the 'Intercept' field
#   nz_genes_e3[1:10]


#Experiment 5: Logistic regression with Ridge
#Note: All features normalized (standardized) before training, and rescaled before prediction
cvfit_e4 = cv.glmnet(x=train_x, y=train_y, nfolds=10, family="binomial", type.measure="class", alpha=0, standardize=TRUE)
pred_y <- predict(cvfit_e4, newx=test_x, s="lambda.min", type="class") #Choosing knee-point lambda value
#   Deliverables
#   1. Accuracy in terms of error rate
err_per4 <- get_error(pred_y, test_y)
#   2. Graph of cross-validation of lambda and min lambda value
#    plot(cvfit_e4)
#    cvfit_e4$lambda.min
#   3. Grouping effect in ridge
coef_e4 <- abs(coef(cvfit_e4, s="lambda.min"))
coefe <- as.matrix(scale(coef_e4[-1])) #To remove intercept and scale coefficients
cl <- dbscan::dbscan(coefe, eps=0.25, minPts=5)
stripplot(coefe)
#   4. Top 10 important genes
nz_idx <- which(coef_e4 != 0)
#   length(nz_idx)
nz_wt_sorted <- sort(coef_e4[nz_idx], decreasing = TRUE, index.return=TRUE)
nz_genes_e4 <- rownames(coef_e4)[nz_idx][nz_wt_sorted$ix[nz_wt_sorted$ix != 1]] #Exclude the 'Intercept' field
#   nz_genes_e4[1:10]


#Experiment 6: Choosing value of mixing parameter(alpha) of elasticnet using cross-validation
b <- matrix(NA,51,3)
j=1
i=0
for(i in seq(0,1,by=0.02))
{
  cvfit_enet2 = cv.glmnet(x=train_x, y=train_y, nfolds=10, type.measure="class", family="binomial", alpha=i, standardize=TRUE)
  b[j,1] <- i
  b[j,2] <- cvfit_enet2$lambda.min
  b[j,3] <- min(cvfit_enet2$cvm)
  j=j+1
}
colnames(b) <- c("alpha","lambda_min","crossval_mean_error")
df_2 <- data.frame(b)
plot(crossval_mean_error ~ alpha,df_2,type="o", col="red", xlab="Alpha", ylab="Crossvalidation Mean Error")
cvm_min <- min(b[,"crossval_mean_error"])
index = which(b[,"crossval_mean_error"]==cvm_min)
alphamin = df_2[index,"alpha"]
# from the plot, the elastic net parameters are alphamin= 0.28
cvfit_enet2 = cv.glmnet(x=train_x, y=train_y, nfolds=10, family="binomial", type.measure="class", alpha=alphamin, standardize=TRUE)
pred_y <- predict(cvfit_enet2, newx=test_x, s="lambda.min", type="class") #Choosing knee-point lambda value
#   Deliverables
#   1. Accuracy in terms of error rate
err_per_enet2 <- get_error(pred_y, test_y)
#   2. Graph of cross-validation of lambda and min lambda value
#    plot(cvfit_e1)
#    cvfit_e1$lambda.min
#   3. Number of non-zero coefficients
coef_e3 <- abs(coef(cvfit_e3, s="lambda.min"))
nz_idx <- which(coef_e3 != 0)
#   length(nz_idx)
#   plot(cvfit_enet$glmnet.fit$lambda, cvfit_enet$glmnet.fit$df, type="o", col="red", ylab="Number of non-zero coefficients", xlab="Lambda")
#   4. Grouping effect elasticnet
coefe <- as.matrix(scale(coef_enet[-1])) #To remove intercept and scale coefficients
cl <- dbscan::dbscan(coefe, eps=0.30, minPts=5)
stripplot(coefe, col="red", xlab="Grouping of coefficients in Ridge (Logistic)")
#   5. Top 10 important genes
nz_wt_sorted <- sort(coef_e3[nz_idx], decreasing = TRUE, index.return=TRUE)
nz_genes_enet2 <- rownames(coef_e3)[nz_idx][nz_wt_sorted$ix[nz_wt_sorted$ix != 1]] #Exclude the 'Intercept' field
#   nz_genes_e3[1:10]    

# Identify common genes in important genes found in all models
Reduce(intersect, list(nz_genes_e1, nz_genes_e2, nz_genes_e3[1:50], nz_genes_e4[1:50], nz_genes_enet[1:50], nz_genes_enet2))
