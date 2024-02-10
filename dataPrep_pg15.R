rm(list = ls())

library(CASdatasets)
library(tpSuite)
library(data.table)

data(pg15training)
inputDT <- as.data.table(pg15training)

setnames(inputDT, 'PolNum', 'polNumb')
setnames(inputDT, 'CalYear', 'uwYear')
setnames(inputDT, 'Gender', 'gender')
setnames(inputDT, 'Type', 'carType')
setnames(inputDT, 'Category', 'carCat')
setnames(inputDT, 'Occupation', 'job')
setnames(inputDT, 'Age', 'age')
setnames(inputDT, 'Group1', 'group1')
setnames(inputDT, 'Bonus', 'bm')
setnames(inputDT, 'Poldur', 'nYears')
setnames(inputDT, 'Value', 'carVal')
setnames(inputDT, 'Adind', 'cover')
setnames(inputDT, 'Density', 'density')
setnames(inputDT, 'Exppdays', 'exposure')
setnames(inputDT, 'Numtppd', 'claimNumbMD')
setnames(inputDT, 'Numtpbi', 'claimNumbBI')
setnames(inputDT, 'Indtppd', 'claimSizeMD')
setnames(inputDT, 'Indtpbi', 'claimSizeBI')


par(mfrow = c(3, 4))
barplot(table(inputDT$uwYear)/nrow(inputDT)*100, col = 'cornflowerblue', xlab = 'Underwriting year (uwYear)', ylab = 'Percent')
barplot(table(inputDT$gender)/nrow(inputDT)*100, col = 'cornflowerblue', xlab = 'Gender')
barplot(table(inputDT$carType)/nrow(inputDT)*100, col = 'cornflowerblue', xlab = 'Car type (carType)')
barplot(table(inputDT$carCat)/nrow(inputDT)*100, col = 'cornflowerblue', xlab = 'Car category (carCat)')
barplot(table(inputDT$job)/nrow(inputDT)*100, col = 'cornflowerblue', xlab = 'Job')
barplot(table(inputDT$ageGrouped)/nrow(inputDT)*100, col = 'cornflowerblue', xlab = 'Age')
barplot(table(inputDT$group1)/nrow(inputDT)*100, col = 'cornflowerblue', xlab = 'Group1')
barplot(table(inputDT$bm)/nrow(inputDT)*100, col = 'cornflowerblue', xlab = 'Bonus malus (bm)')
barplot(table(inputDT$nYears)/nrow(inputDT)*100, col = 'cornflowerblue', xlab = 'Number of policy years (nYears)')
barplot(table(inputDT$carValGrouped)/nrow(inputDT)*100, col = 'cornflowerblue', xlab = 'Car value (carVal)')
barplot(table(inputDT$cover)/nrow(inputDT)*100, col = 'cornflowerblue', xlab = 'Cover')
barplot(table(inputDT$densityGrouped)/nrow(inputDT)*100, col = 'cornflowerblue', xlab = 'Density')


sum(inputDT$exposure == 1)/nrow(inputDT)
par(mfrow = c(1, 1))

hist(inputDT$exposure, xlab = 'Exposure', col = 'cornflowerblue', main = '', ylim = c(0, 80000))
text(x = 0.97, y = 75000, label = paste0(round(sum(inputDT$exposure == 1)/nrow(inputDT)*100, digits = 2), '%'), pos = 3, cex = 2, col = "black")

xx <- barplot(table(inputDT$claimNumbBI), col = 'cornflowerblue', xlab = 'Claim number', ylim=c(0,105000), yaxt = "n")
text(x = xx, y = table(inputDT$claimNumbBI), label = paste0(round(table(inputDT$claimNumbBI)/nrow(inputDT)*100, digits = 2), '%'), pos = 3, cex = 2, col = "black")
# text(x = xx, y = c(85000, 4338, 168), label = paste0(round(table(inputDT$claimNumbBI)/nrow(inputDT)*100, digits = 2), '%'), pos = 3, cex = 2, col = "black")
axis(side = 2, at = c(seq(0, 80000, 20000), 95000), labels = c(seq(0, 80000, 20000), 95000))

hist(inputDT$claimSizeBI[inputDT$claimSizeBI != 0], breaks = 'FD', xlab = 'Claim size', col = 'cornflowerblue', main = '', ylim = c(0, 1620))
text(x = 1500, y = 1580, label = paste0(round(length(inputDT$claimSizeBI[inputDT$claimSizeBI != 0 & inputDT$claimSizeBI < 1000])/length(inputDT$claimSizeBI[inputDT$claimSizeBI != 0])*100, digits = 2), '%'), pos = 3, cex = 2, col = "black")



## Make the frequencies numbers (rather than factors)
dat$freqs <- as.numeric(as.character(dat$freqs))
## Find a range of y's that'll leave sufficient space above the tallest bar
ylim <- c(0, 1.1*max(dat$freqs))
## Plot, and store x-coordinates of bars in xx
xx <- barplot(dat$freqs, xaxt = 'n', xlab = '', width = 0.85, ylim = ylim,
              main = "Sample Sizes of Various Fitness Traits", 
              ylab = "Frequency")
## Add text at top of bars
text(x = xx, y = dat$freqs, label = dat$freqs, pos = 3, cex = 0.8, col = "red")
## Add x-axis labels 
axis(1, at=xx, labels=dat$fac, tick=FALSE, las=2, line=-0.5, cex.axis=0.5)

inputDT[claimNumbBI > 2, claimNumbBI := 2]
inputDT[, SubGroup2 := NULL]
inputDT[, Group2 := NULL]

library(R2DT)

asFactorDT(inputDT, c('polNumb', 'uwYear', 'gender', 'carType', 'carCat', 'job', 'group1', 'bm', 'nYears', 'cover'))
asNumericDT(inputDT, c('exposure', 'age', 'density', 'carVal', 'claimNumbMD', 'claimNumbBI', 'claimSizeMD', 'claimSizeBI'))

inputDT[, exposure := exposure/365]

#No missings in the data set

set.seed(100)
samp <- sample(c(1:nrow(inputDT)), round(0.9*nrow(inputDT)), replace = FALSE)
trainDT <- inputDT[samp, ]
testDT <- inputDT[setdiff(c(1:nrow(inputDT)), samp),]

library(rpart)
library(rpart.utils)
library(mgcv)

gamMain <- gam(formula = claimNumbBI ~ s(age) + s(density) + s(carVal) + uwYear + gender + carType + carCat + job + group1 + bm + nYears + cover + offset(exposure), family = poisson(link = log), data = trainDT)
# gamMain <- gam(formula = claimNumbBI ~ s(age) + s(density) + s(carVal) + offset(exposure), family = poisson(link = log), data = trainDT)
# bamMain <- bam(formula = claimNumb ~ s(age) + sex + s(experience) + s(carAge) + privateUse + coverType + noClaimsHistory + s(latitude, longitude) + offset(exposure), family = poisson(link = log), data = trainDT)

# group1, bm, (nYears)
# table(trainDT$cover)

testDT$predGam <- predict(gamMain, newdata = testDT, type = "response")

library(plyr)
library(ggplot2)

# plotUnivSpline(gamMain, trainDT, 'age')
# plotUnivSpline(gamMain, trainDT, 'carVal')
# plotUnivSpline(gamMain, trainDT, 'density')

splineEst <- extractSplineEstimate(gamMain, c('age', 'carVal', 'density'))

# splitsAge <- optNumbGroups(trainDT, gamMain, splineEst, 'age', xLimit = list(age = c(18, 75)), nGroups = c(4, 6, 8), addVar = TRUE)
# splitsCarVal <- optNumbGroups(trainDT, gamMain, splineEst, 'carVal', xLimit = list(carVal = c(1000, 40000)), nGroups = c(4, 6, 8), addVar = TRUE)
# splitsDensity <- optNumbGroups(trainDT, gamMain, splineEst, 'density', xLimit = list(density = c(0, 300)), nGroups = c(4, 6, 8), addVar = TRUE)

splitsAge <- groupingVars('age', 6, gamMain, trainDT, splineEst, sampleSizeBin = 5000)
splitsCarVal <- groupingVars('carVal', 6, gamMain, trainDT, splineEst, sampleSizeBin = 5000)
splitsDensity <- groupingVars('density', 6, gamMain, trainDT, splineEst, sampleSizeBin = 5000)

splitsAll <- list()
splitsAll[[1]] <- splitsAge
splitsAll[[2]] <- splitsCarVal
splitsAll[[3]] <- splitsDensity
names(splitsAll) <- c('age', 'carVal', 'density')

# transform2BinnedVar(inputDT, splitsAll, removeAlreadyPresent = TRUE)

transform2BinnedVar(trainDT, splitsAll, removeAlreadyPresent = TRUE)
transform2BinnedVar(testDT, splitsAll, removeAlreadyPresent = TRUE)
transform2BinnedVar(inputDT, splitsAll, removeAlreadyPresent = TRUE)

rm(gamMain);rm(splitsAge);rm(splitsCarVal);rm(splitsDensity);rm(splineEst);rm(pg15training)

splitsAll

# save.image('C:/Users/xrvrbk/Desktop/GA/workspace/pg15trainingPrepped.RData')
load('C:/Users/xrvrbk/Desktop/GA/workspace/pg15trainingPrepped.RData')
