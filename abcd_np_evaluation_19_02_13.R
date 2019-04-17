

###################################################
###################################################
## ABCD Neurocognitive Prediction Challenge      ##
## Computing Validation MSE & R-squared          ##
##                                               ##
## Feb 13, 2019                                  ##
##                                               ##
###################################################
###################################################

## Read in and 'ground-truth' scores and predicted scores from contestant, should be one prediction per test subject
## The first input argument to the script should contain the ground-truth, and the second one the predicted scores
## Both files should have two columns. The first one should be the subject ID and the second one should be 'fluid.resid' for the ground-truth file; and 'predicted_score' for the predicted file

args <- commandArgs(TRUE)
gt_file <- args[1]
pred_file <- args[2]
pred = read.csv(pred_file,header=TRUE)
fluid_resid_test = read.csv(gt_file,header=TRUE)
test = merge(fluid_resid_test,pred,by=names(fluid_resid_test)[1])

## Only compute MSE and R-squared if the number of predictions made is at least 99% of the test sample
if(dim(test)[1] >= .99*dim(fluid_resid_test)[1]){
	test$predicted_score[is.na(test$predicted_score)] = 
		test$predicted_score[abs(test$predicted_score-test$fluid.resid) == max(abs(test$predicted_score-test$fluid.resid),na.rm=TRUE)]
	r.squared = cor(test$fluid.resid,test$predicted_score)^2
	mse = mean((test$fluid.resid - test$predicted_score)^2)
	cat("MSE: ", mse, "\nR-Squared: ", r.squared, "\n")
}
