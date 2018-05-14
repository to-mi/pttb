suppressPackageStartupMessages(library(R.matlab))
library(heuristica)
source("TTBABC/R/ttbabcfit.R")
source("TTBABC/R/ttbabcpredict.R")

# load input data:
# - methods (vector of strings)
# - criterion_col (scalar, index of criterion)
# - cue_cols (vector of length n_cues, indices of cues)
# - tr_data (N_tr x (n_cues + 1), table with cues and criterion as columns)
# - tr_data_cues_pw (N_pairs_tr x n_cues, pairwise differences, -1/0/1)
# - tr_data_criterion_pw (vector of length N_pairs_tr, pairwise differences, -1/1)
# - te_data (N_te x (n_cues + 1), table with cues and criterion as columns)
# - te_data_cues_pw (N_pairs_te x n_cues, pairwise differences, -1/0/1)
# - te_data_criterion_pw (vector of length N_pairs_te, pairwise differences, -1/1)
args <- commandArgs(trailingOnly=TRUE)
input <- readMat(sprintf('%s_input.mat', args[1]))

methods <- unlist(input$methods)
criterion_col <- input$criterion.col
cue_cols <- input$cue.cols
tr_data <- input$tr.data
tr_data_cues_pw <- input$tr.data.cues.pw
tr_data_criterion_pw <- as.vector(input$tr.data.criterion.pw)
te_data <- input$te.data
te_data_cues_pw <- input$te.data.cues.pw
te_data_criterion_pw <- as.vector(input$te.data.criterion.pw)

n_cues <- length(cue_cols)
n_methods <- length(methods)

# run the model(s)
yhat <- matrix(NA, ncol=n_methods, nrow=length(te_data_criterion_pw))
for (m_i in 1:n_methods) {
  switch(methods[m_i],
         "ttb" = {
           fit <- ttbModel(tr_data, criterion_col, cue_cols)
           yhat[, m_i] <- predictPairSummary(te_data, fit)[, "ttbModel"]
         },
         "ttbGreedy" = {
           fit <- ttbGreedyModel(tr_data, criterion_col, cue_cols)
           yhat[, m_i] <- predictPairSummary(te_data, fit)[, "ttbGreedyModel"]
           fit$cue_validities_unreversed <- NA
         },
         "unitWeight" = {
           fit <- unitWeightModel(tr_data, criterion_col, cue_cols)
           yhat[, m_i] <- predictPairSummary(te_data, fit)[, "unitWeightModel"]
         },
         "logReg" = {
           fit <- logRegModel(tr_data, criterion_col, cue_cols)
           yhat[, m_i] <- predictPairSummary(te_data, fit)[, "logRegModel"]
           fit <- fit$coefficients
         },
         "ttb_from_ttbabc" = {
           ###Order in dependency of correlation
           ttborder <- data.frame(n=1:n_cues, dir=cor(tr_data_cues_pw, tr_data_criterion_pw))
           ###if na, we assume it's 0, can happen sometimes, especially for rare cases (i.e. capital city)
           ttborder$dir <- ifelse(is.na(ttborder$dir), 0, ttborder$dir)
           ###order in dependency of validity
           ttborder <- ttborder[order(abs(ttborder$dir), decreasing=TRUE),]
           ###get the weights by dichotomizing
           ttborder$dir <- ifelse(ttborder$dir > 0, 1, -1)
           ##Predict TTB
           ###copy dt to avoid that the changed order messes things up later
           dtcopy <- te_data_cues_pw
           ###pre-order by weight magnitude
           dtcopy <- dtcopy[, ttborder$n]
           ###loop through order
           for (j in 1:nrow(ttborder)){
             ####Output=win or loss_j times weight
             dtcopy[, j] <- dtcopy[, j] * ttborder$dir[j]
           }
           ###Generate predictions by looping through ordered copy of test set
           predttb <- rep(0, nrow(dtcopy))
           ###loop through
           for (j in 1:nrow(ttborder)){
             ####prediction is the ordered copy times the weights
             predttb <- ifelse(predttb == 0, dtcopy[, j], predttb)
           }
           ###if 0, then guess
           # note: changes this such that each zero is predicted independently
           yhat[, m_i] <- ifelse(predttb==0, sample(c(-1, 1), length(predttb), replace=T), predttb)
           fit <- NA
         },
         "ttbabc" = {
           y <- as.data.frame(tr_data_criterion_pw)
           colnames(y) <- 'y'
           fit <- ttbabcfit(as.data.frame(tr_data_cues_pw), y, nsims=1000, epsilon=0.5, proportion=0.1, nsamples=100, countprior=c(1,2), progress=F)
           yhat[, m_i] <- ttbabcpredict(fit, as.data.frame(te_data_cues_pw))$class
         },
         stop("unknown method"))
}

# save predictions
writeMat(sprintf('%s_output.mat', args[1]), yhat=yhat, fit=fit)
