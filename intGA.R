rm(list = ls())

library(Matrix)
library(data.table)
library(R2DT)
library(devFunc)
library(plyr)
library(mgcv)
library(speedglm)
library(feather)

dataDir <- "C://Users//xrvrbk//Desktop//GA//"
dirShape <- "C://Users//xrvrbk//Desktop//GA//shapeFiles//"

load('C:/Users/xrvrbk/Desktop/GA/workspace/pg15trainingPrepped.RData')

source(paste0(dataDir, 'intGAFuncs.R'))

seedNumb <- 3435
set.seed(seedNumb)

inputDT[, c('polNumb', 'claimNumbMD', 'claimSizeMD', 'claimSizeBI', 'age', 'density', 'carVal') := NULL]
setnames(inputDT, 'claimNumbBI', 'claimNumber')
setnames(inputDT, 'ageGrouped', 'age')
setnames(inputDT, 'carValGrouped', 'carVal')
setnames(inputDT, 'densityGrouped', 'density')

# claimNumbBI ~ s(age) + s(density) + s(carVal) + uwYear + gender + carType + carCat + job + group1 + bm + nYears + cover + offset(exposure)

allFirstOrderIntTerms <- function(variableNames){
  count <- 1
  if(length(variableNames) > 1){
    firstOrderTerms <- rep(NA, ((length(variableNames) - 1) * length(variableNames))/2)
    for(varLeft in 1:(length(variableNames) - 1)){
      for(varRight in (varLeft + 1):length(variableNames)){
        firstOrderTerms[count] <- paste(variableNames[varLeft], variableNames[varRight], sep = '*')
        count <- count + 1
      }
    }
    return(firstOrderTerms)
  }  
  return(NULL)
}

# checking if the data type is right (with the is... funtions) and rectifying it if they aren't (with the as... functions)

isFactorDT(inputDT, ,T)
isNumericDT(inputDT, ,T)

# setnames(inputDT, 'Clm_Count', 'claimNumber')
# setnames(inputDT, 'TLength', 'exposure')
# setnames(inputDT, 'AgeInsured', 'age')
# setnames(inputDT, 'SexInsured', 'sex')
# setnames(inputDT, 'Experience', 'experience')
# setnames(inputDT, 'VAge', 'carAge')
# setnames(inputDT, 'PrivateCar', 'privateUse')
# setnames(inputDT, 'Cover_C', 'coverType')
# setnames(inputDT, 'AvPayment', 'claimSize')
# setnames(inputDT, 'NCD_0', 'noClaimsHistory')
# 
# # checking if the data type is right (with the is... funtions) and rectifying it if they aren't (with the as... functions)
# 
# isFactorDT(inputDT, ,T)
# isNumericDT(inputDT, ,T)
# 
# asFactorDT(inputDT, c('privateUse', 'noClaimsHistory', 'coverType'))
# asNumericDT(inputDT, c('exposure', 'claimSize', 'carAge', 'age'))
# 
# table(inputDT$experience)
# table(inputDT$carAge)
# table(inputDT$age)
# 
# inputDT[carAge > 30, carAge := 31]
# inputDT[age < 20, age := 20]
# inputDT[age > 70, age := 71]
# inputDT[experience > 41, experience := 41]
# 
# table(inputDT$carAge)
# table(inputDT$age)
# 
# asFactorDT(inputDT, c('carAge', 'age', 'experience'))
# 
# isFactorDT(inputDT, ,T)
# isNumericDT(inputDT, ,T)
# 
# inputDT[, exposure := exposure/365]
# 
# set.seed(12)

# > splitsAll
# $age
# $age$ggplotObject
# 
# $age$splits
# [1] 23.5 25.5 27.5 29.5 31.5 33.5 68.5
# 
# 
# $experience
# $experience$ggplotObject
# 
# $experience$splits
# [1]  4.5  7.5 11.5 14.5 17.5 24.5 33.5
# 
# 
# $carAge
# $carAge$ggplotObject
# 
# $carAge$splits
# [1]  0.5  1.5  2.5  7.5  8.5  9.5 23.5

# changeLevelNamesDT(testDT, 'ageGrouped', as.character(1:10), c('<23', '[23, 25[', '[25, 27[', '[27, 29[', '[29, 31[', '[31, 33[', '[33, 35[', '[35, 65[', '[65, 72[', '>= 72'))

# varsGA <- c("age", "sex", "experience", "exposure", "claimNumb", "carAge", "privateUse", "noClaimsHistory", "coverType", "claimSize", "latitude", "longitude")
# varsGA <- c("age", "sex", "carAge", "privateUse", "noClaimsHistory", "coverType")
# vars <- c("age", "sex", "experience", "carAge", "privateUse", "noClaimsHistory", "coverType")
vars <- c('age', 'density', 'carVal', 'uwYear', 'gender', 'carType', 'carCat', 'job', 'group1', 'bm', 'nYears', 'cover')
varsGA <- c(vars, allFirstOrderIntTerms(vars))

trainSamp <- sample(c(1:nrow(inputDT)), round(0.8*nrow(inputDT)), replace = FALSE)
testSamp <- createTestSet(inputDT[,.SD,.SDcol = unique(vars)], trainSamp, (nrow(inputDT) - length(trainSamp)))

trainDT <- inputDT[trainSamp, ]
testDT <- inputDT[testSamp, ]

inputDTAll <- copy(inputDT)
inputDT <- copy(trainDT)

nGens <- 20
nModelsGen <- rep(10, nGens)
nCrossOverGen <- rep(1, nGens)
nMutationsGen <- rep(1, nGens)
nVarInit <- 10
nVarMax <- 21
nRedModels <- 5 
trainPerc <- 0.5
testPerc <- NULL
nCores <- NULL
nEstConcProb <- 1
# nEstConcProb <- 5
valOffSet <- 'exposure'
# nMinInOptModsGen <- c(0,0,0,0,0,0,0,1,0,0)
# nMinInOptModsGen <- NULL
nMinInOptModsGen <- rep(1, nGens)
includeIntsGen <- rep(FALSE, nGens)
# nModelsIntGen <- rep(c(0,2), nGens/2)
nModelsIntGen <- rep(0, nGens)
# nMaxIntGen <- rep(c(0,5), nGens/2)
nMaxIntGen <- rep(0, nGens)

nAddedBestModelsGen = rep(5, nGens)

#added 
ratioConcProbMSE <- 0.5 #1 = MSE, 0 = CP
ilst2Save <- NULL
distModel <- 'poisson'

list2Save <- NULL
zoneAgeFormGen <- NULL
updateOffsetZoneGen <- NULL
updateOffsetAgeBySex <- FALSE
upperBoundPred <- NULL
partialSaveGen <- NULL
concProbLow <- 0
concProbHigh <- 1

badOnes <- c()


#
# Genetic algorithm function
#

# inputDT = data.table object : Our database
#     At this point, only categorical vars are accepted.

# varsGA = the vars that will be selected or not in the genetic algo (exposure, claimSize et claimNumber not included).

# nVarInit = Number of vars used during the initialization phase

# nVarMax = Maximum number of vars during each generation

# nModelsGen = Number of models for each generation (needs to be pair number, length = nGens and the numbers can only be monotonuously decreasing)

# nGens = Number of generations

# nCrossOverGen = Number of cross-overs (length = nGens)

# nMutationsGen = Number of mutations (length = nGens)

# nCores : Number of cores of g/bam, if equal to NULL, it is just a regular g/bam

# nEstConcProb: number of times that the concProb is calculated

# distModel : selected distribution modele. Possibilities are 'poisson', gamma' and 'gaussian'. 
# The right C-index is chosen in both cases (concProbLow and concProbHigh have different meanings in both cases).

# trainPerc : percentage of inputDT that is included in the training data set (at each generation, another split is made)

# testPerc : percentage of inputDT that is included in the training data set. If equal to NULL, then the complement of the training set is used. 
# If trainPerc = 1, then testPerc = 1 as well.

# nRedModels : For models with more than nVarMax variables, random sample of nVarMax vars are taken from the selected set, 
# and this is repeated nRedModels times. The model (and their corresponding selection of vars) with the highest C-index is retained.

# nAddedBestModelsGen : Of all the past models, that have a C-index higher than the best model of the current generation, nAddedBestModelsGen are sampled
# and this number of models is in the running to be transmitted to the next generation (to improve the convergence of the GA). 
# A random sample is taken from the best models and the ones of the current generation. (length = nGens)

# valOffSet : This variable allows one to set the offset (set it equal to 'exposure'), being exposure in most cases (the right link function is automatically added to it). 
# Note that if disModel is gamma or gaussian, this argument should be set to NULL. One can also go for more exotic exposures such as log(s(lat,long) + s(age, by = sexe)). 

# nMinInOptModsGen : The minimal number of times a variable should appear in the set of final models of the generation and in all the model of the previous generations 
# that had a higher value in concProb than the concProb of the best model of the current generation (length = nGens)

# includeIntsGen : Boolean vector that indicates whether or not interactions are taken into account for the given generation. When equal to NULL, no interactions are 
# taken into account. Note that interactions are never considered for the init phase (length = nGens)

# nModelsIntGen: Number of models with randomly sampled interactions terms that are considered by the GA (we evaluate the effect of interaction effects by randomly 
# adding interaction terms to the main effects and to just consider the model with the highest concProb next). (length = nGens)

# nMaxIntGen: Maximum number of interaction terms that are added to the main effects, when testing nModelsIntGen. When it equals NULL, 
# no interaction terms are considered. (length = nGens)

# updateOffsetZoneGen Boolean vector that indicates whether or not the zone effect should be refitted during the considered generation. For the first generation, 
# no update of the zone is done in any case. At a certain generation, the very best past model is selected, and the zone is refitted for this case, 
# which will correspond to the zone effect that will be fixed in the offset, up until the zone effect will be refitted. If this argument equals NULL, 
# no update of the zone effect is performed during the GA. Note that this argument only makes sense for the Poisson model (length = nGens)

# updateOffsetAgeBySex Boolean (vector of length 1) indicating whether or not the offset is also updated with a refitted estimate of the age spline estimate, stratified 
# for gender, when the zone estimated are updated (as indicated by the argument updateOffsetZoneGen). At this moment not in use, and even hardcoded in the functions 
# to be equal to FALSE (this argument currently is completely useless). 

# partialSaveGen Boolean indicating whether or not the output of a certain generation should be saved or not. If equal to NULL, no partial saves are performed.
# the variable name of this partial save corresponds to the value of the argument partialSaveName (default name equals to TempGA). (length = nGens)

# list2Save Character vector that indicates which of the next outputs would be interesting to save:
#             list(matBinInit = matBinInit, listCIndexInit = listCIndexInit,
#             listMatMutFin = listMatMutFin, listMatReprodFin = listMatReprodFin, listMatFitFin = listMatFitFin, listModelsFin = listModelsFin,
#             listNewModelsFin = listNewModelsFin, listCIndexFin = listCIndexFin,
#             listPoidsCIndexFin = listPoidsCIndexFin, listvarsGAFin = listvarsGAFin, minimalSet = minimalSet,
#             listIntFin = listIntFin, newvalOffSet = inputDT$newvalOffSet, iGen = iGen, MSEFull, MSENull, MSEFullNew, CPFull, CPNull, CPFullNew)
# If it is equal to NULL, then the above list is constructed. 

# ratioConcProbMSE: Ratio between the concProb and MSE for the fitting criterion. When the value is equal to 0, only the concProb is taken into account. 
# When its value is different from 0 or 1, the normalized value for the concProb and the MSE is used. 

# zoneAgeFormGen: Is the zone effect (bivariate spline on latitude and longitude, and spline on age) applied for each fitted model. This is not recommended 
# when using large data sets, as is often the case for frequency data. The default value is NULL, meaning that both effects are not each time refitted. (length = nGens)

# upperBoundPred : when dealing with (strictly positive) continuous data, it is better to sometimes just truncate the predictions, to avoid values that do not 
# make economical sense. The default value is NULL, meaning that there is no truncation applied to the predictions.

functionGA <- function(inputDT, varsGA, nModelsGen, nGens, nCrossOverGen, nMutationsGen, concProbLow, concProbHigh,
                       nVarInit, nVarMax, nRedModels = 5, trainPerc = 0.8, testPerc = NULL,
                       distModel = 'poisson', nCores = NULL, nEstConcProb = 5,
                       nAddedBestModelsGen = rep(5, nGens), valOffSet = 'exposure',
                       nMinInOptModsGen = NULL, 
                       includeIntsGen = NULL, nModelsIntGen = NULL, nMaxIntGen = NULL, 
                       updateOffsetZoneGen = NULL, partialSaveGen = NULL, partialSaveName = NULL, 
                       list2Save = NULL,
                       parThreshold = NULL, typeThreshold = NULL,
                       ratioConcProbMSE = NULL, zoneAgeFormGen = NULL, 
                       upperBoundPred = NULL, badOnes = NULL){
  
  gamlssModel <- TRUE
  nModelsInit = nModelsGen[1]
  
  # Ajout 02/07 : distModel
  if(distModel == 'poisson'){
    distModel <- poisson(link='log')
    gamlssModel <- FALSE
  } else if(distModel== 'gamm\a'){
    distModel <- Gamma(link = 'log')
    gamlssModel <- FALSE
  } else if (distModel == 'gaussian'){
    distModel <- gaussian(link = 'identity')
    gamlssModel <- FALSE
  } else if (distModel == 'binomial'){
    distModel <- binomial(link = "logit")
    gamlssModel <- FALSE
  } else if(distModel == 'pareto'){ # Distribution de Pareto.
    if (is.null(parThreshold)){
      distModel <- PARETO2()
    } else{
      functionGenerateTrun(par = parThreshold, family = PARETO2(), name = 'tr', type = typeThreshold)
      distModel <- PARETO2tr()
    }
  } else if(distModel == 'weibul'){ # Distribution de Weibul.
    if (is.null(parThreshold)){
      distModel <- WEI()
    } else{
      functionGenerateTrun(par = parThreshold, family = WEI(), name = 'tr', type = typeThreshold)
      distModel <- WEItr()
    }
  } else if(distModel == 'gGamma'){ # Gamma generalisee.
    if (is.null(parThreshold)){
      distModel <- GG()
    } else{
      functionGenerateTrun(par = parThreshold, family = GG(), name = 'tr', type = typeThreshold)
      distModel <- GGtr()
    }
  }else if(distModel == 'gumbel'){ # Gumbel distribution
    if (is.null(parThreshold)){
      distModel <- GU()
    } else{
      functionGenerateTrun(par = parThreshold, family = GU(), name = 'tr', type = typeThreshold)
      distModel <- GUtr()
    }
  } else if(distModel == 'iGamma'){ # Inverse gamma
    if (is.null(parThreshold)){
      distModel <- IGAMMA()
    } else{
      functionGenerateTrun(par = parThreshold, family = IGAMMA(), name = 'tr', type = typeThreshold)
      distModel <- IGAMMAtr()
    }
  }else if(distModel == 'Gamma'){ # Gamma via GAMLSS
    if (is.null(parThreshold)){
      distModel <- GA()
    } else{
      functionGenerateTrun(par = parThreshold, family = GA(), name = 'tr', type = typeThreshold)
      distModel <- GAtr()
    }
  }else if(distModel == 'BCT'){ # BCT
    if (is.null(parThreshold)){
      distModel <- BCT()
    } else{
      functionGenerateTrun(par = parThreshold, family = BCT(), name = 'tr', type = typeThreshold)
      distModel <- BCTtr()
    }
  }else if(distModel == 'logNormale'){ # Lognormale.
    if (is.null(parThreshold)){
      distModel <- LOGNO()
    } else{
      functionGenerateTrun(par = parThreshold, family = LOGNO(), name = 'tr', type = typeThreshold)
      distModel <- LOGNOtr()
    }
  }else if(distModel == 'GB2'){ # Generalized Beta type 2.
    if (is.null(parThreshold)){
      distModel <- GB2()
    } else{
      functionGenerateTrun(par = parThreshold, family = GB2(), name = 'tr', type = typeThreshold)
      distModel <- GB2tr()
    }
  }else if(distModel == 'iGaussian'){ # inverse gaussienne.
    if (is.null(parThreshold)){
      distModel <- IG()
    } else{
      functionGenerateTrun(par = parThreshold, family = IG(), name = 'tr', type = typeThreshold)
      distModel <- IGtr()
    }
  } else if(distModel == 'exp'){ # distribution exponentielle.
    if (is.null(parThreshold)){
      distModel <- EXP()
    } else{
      functionGenerateTrun(par = parThreshold, family = EXP(), name = 'tr', type = typeThreshold)
      distModel <- EXPtr()
    }
  }else{
    stop('distModel should be poisson, gamma, gaussian or binomial to perform GAM. Not defined for GAMLSS otherwise.')
  }
  
  '%notin%' <- Negate('%in%')
  if (distModel$family[1] == 'poisson'){
    checkDT(inputDT, 'claimNumber')
    inputDT[, observed := claimNumber]
  }else if (distModel$family[1] != 'binomial'){
    checkDT(inputDT, 'claimSize')
    inputDT[, observed := claimSize]
  } else{ # distModel$family[1] = binomial -> Proba.
    checkDT(inputDT, 'claimSizeProb')
    inputDT[, observed := claimSizeProb]
    inputDT[, exposure := 1] # On force l'exposition a 1 pour evaluer concProb.
  }
  
  #
  # Tests of the different values/objects
  #
  
  checkCharVec(list(varsGA))
  checkDT(inputDT, unique(unlist(strsplit(varsGA, '[*]')))) 
  
  checkNumOrIntVec(list(nModelsInit))
  checkLength(list(nModelsInit), 1)
  checkRanges(list(nModelsInit), list(c('>',0)))
  
  checkNumOrIntVec(list(nModelsGen))
  checkLength(list(nModelsGen), nGens)
  for(iList in 1:length(nModelsGen)) checkRanges(as.list(nModelsGen)[iList], list(c('>',0)))
  
  checkNumOrIntVec(list(nCrossOverGen))
  checkLength(list(nCrossOverGen), nGens)
  for(iList in 1:length(nCrossOverGen)) checkRanges(as.list(nCrossOverGen)[iList], list(c('>',0)))
  
  checkNumOrIntVec(list(nMutationsGen))
  checkLength(list(nMutationsGen), nGens)
  for(iList in 1:length(nMutationsGen)) checkRanges(as.list(nMutationsGen)[iList], list(c('>',0)))
  
  checkNumOrIntVec(list(nVarInit))
  checkLength(list(nVarInit), 1)
  checkRanges(list(nVarInit), list(c('>', 0,'<=',length(varsGA))))
  
  checkNumOrIntVec(list(nVarMax))
  checkLength(list(nVarMax), 1)
  checkRanges(list(nVarMax), list(c('>', 0,'<=',length(varsGA))))
  
  checkNumOrIntVec(list(nEstConcProb))
  checkLength(list(nEstConcProb), 1)
  checkRanges(list(nEstConcProb), list(c('>', 0)))
  
  if(!is.null(nMinInOptModsGen)){
    checkNumOrIntVec(list(nMinInOptModsGen))
    checkLength(list(nMinInOptModsGen), nGens)
    for(iList in 1:length(nMinInOptModsGen)) checkRanges(as.list(nMinInOptModsGen)[iList], list(c('>=',0)))
  }
  
  if((sum(nModelsGen%%2)>0) || (nModelsInit%%2 == 1)){ 
    stop('All element of nModelsGen/nModelsInit should be even.')
  } else if (all(isFactorDT(inputDT, unique(unlist(strsplit(varsGA, '[*]'))))) == FALSE){
    stop('All covariate in varsGA should be categorical (except exposure, claimSize and claimNumber)')
  }
  
  for (iLengMod in 1:length(nModelsGen)){
    if(iLengMod == 1){
      if(nModelsGen[iLengMod] > nModelsInit){
        stop('Number of model should be decreasing : nModelsGen[1] <= nModelsInit')
      }
      else{}
    }
    else{
      if(nModelsGen[iLengMod] > nModelsGen[iLengMod-1]){
        stop('Number of model should be decreasing : nModelsGen[counter] <= nModelsGen[counter-1]')
      }
      else{}
    }
  }
  
  checkLength(list(nAddedBestModelsGen), nGens)
  
  if (!is.null(list2Save)){
    
    if(!all(names(list2Save) %in% c('matBinInit', 'listCIndexInit', 'listMatMutFin', 'listMatReprodFin', 'listMatFitFin', 
                                    'listModelsFin', 'listNewModelsFin', 'listCIndexFin', 'listPoidsCIndexFin', 'listvarsGAFin',
                                    'listIntFin', 'newvalOffSet', 'iGen', 'MSEFull', 'MSENull', 'CPFull', 'CPNull'))){
      stop('list2Save doesn t contain all/the right arguments !')
    }
    
    if(!all(nModelsGen <= length(list2Save$listNewModelsFin[[list2Save$iGen]]))){
      stop('Elements of nModelsGen should be <= last number of models (before save).')
    }
  }
  
  listVarLevels <- extractLevelDT(inputDT[,.SD,.SDcol = c(unique(c(unlist(strsplit(unlist(varsGA), '[*]')))))])
  lengthMaxVarLevels <- max(unlist(llply(1:100000, function(xx) sum(length(unlist(listVarLevels[sample(1:length(c(varsGA)), nVarMax)]))) - 1))*5)
  
  if (nrow(inputDT)*trainPerc*1.05 < (lengthMaxVarLevels)){ # * 1.05 pour petite marge d erreur, pas trop restrictif tout de meme !
    trainPerc <- 1
    cat('Pas assez de lignes pour le nombre de levels consideres : trainPerc = testPerc = 1. Meme training et test sets. \n') 
  }
  
  if ((!is.null(updateOffsetZoneGen)) && (!is.null(zoneAgeFormGen))){
    stop('Update du zonier en offset et prise en compte du zonier sous forme de spline dans les formules des GAM/GAMLSS !!! Si updateOffsetZoneGen != NULL alors zoneAgeFormGen = NULL (et inversement).')
  }
  
  #
  #  Initialisation matrix + initialisation GA.
  #
  
  if (is.null(list2Save)){
    
    cat('list2Save est NULL : GA classique avec initialisation. \n')
    
    listModelsFin <- list() # Il s agit d une liste de liste avec tous les modeles, ainsi on garde acces a la totalite. # Tous les modeles pour chaque generation.
    listNewModelsFin <- list() # Liste avec les modeles optimaux compte tenu de nVarMax.
    listCIndexFin <- list() # Pareil : Liste des C index pour chaque modele de chaque generation.
    listPoidsCIndexFin <- list() # Liste des poids pour le CIndex, pour classer les modeles.
    listMatReprodFin <- list() # Liste des matrices apres reproduction
    listMatMutFin <- list() # Liste des matrices apres mutation.
    listMatFitFin <- list() # Liste des matrices apres fitting/C-index.
    listvarsGAFin <- list() # Liste des Variables : Besoin de ca car on change l ordre des variables durant la reproduction.
    listIntFin <- list() # Liste des interactions pour chaque generation.
    
    #
    #  Compute MSE/C-Index for the claim size model
    #
    
    '%notin%' <- Negate('%in%')
    
    # the next lines until 760 was commented out 
    
    #if (distModel$family[1] %notin% c('poisson', 'binomial') && ratioConcProbMSE %notin% c(0,1)){ # Uniquement C-index dans ce cas. Pas besoin de calculer MSE dans ce contexte.
    if(distModel$family[1] %in% c('poisson', 'binomial') && ratioConcProbMSE %notin% c(0,1)){ 

      if(trainPerc == 1){
        trainSamp = 1:nrow(inputDT)
        testSamp = 1:nrow(inputDT)
      }else if(is.null(testPerc)){ 
      	trainSamp = sample(1:nrow(inputDT), trainPerc*nrow(inputDT))
        testSamp = createTestSet(inputDT[,.SD,.SDcol = unique(c(unlist(strsplit(unlist(varsGA), '[*]'))))], trainSamp, lengthTestSamp = (nrow(inputDT) - length(trainSamp)))
      } else{ 
      	trainSamp = sample(1:nrow(inputDT), trainPerc*nrow(inputDT))
        testSamp = createTestSet(inputDT[,.SD,.SDcol = unique(c(unlist(strsplit(unlist(varsGA), '[*]'))))], trainSamp, lengthTestSamp = testPerc*nrow(inputDT))
      }

      # Null model

      cat('Computation MSE/CP for the NULL model. \n')

      if(!is.null(updateOffsetZoneGen) || !is.null(zoneAgeFormGen)){
        formNull <- functionMakeFormula(distModel, offset = NULL, zoneAgeForm = TRUE)
      }else{
        formNull <- functionMakeFormula(distModel, NULL, offset = valOffSet)
      }

      fitNull <- functionFitIter(formNull, distModel, inputDT, trainSamp, testSamp, nCores, gamlssModel, upperBoundPred)
      MSENull <- sum((fitNull$predModel - inputDT$observed[testSamp])^2*inputDT$exposure[testSamp], na.rm = TRUE)/sum(inputDT$exposure[testSamp], na.rm = TRUE)
      
      listCIndexNull <- list()
      length(listCIndexNull) <- nEstConcProb
      llply(1:length(listCIndexNull), function(xx) xx <- NA)

      inputDT[testSamp, predicted := fitNull$predModel]
      
      expSplits <- unique(quantile(inputDT[testSamp, exposure], seq(0.01, 0.99, 0.01)))[-1]
      quantSplits <- seq(0.05, 0.95, 0.05)
      CPNull <- concProbGrid(inputDT[testSamp, ], lowCat = concProbLow, highCat = concProbHigh, expSplits, quantSplits)$concProbGlobal
      
      # Full model

      cat('Computation MSE/CP for the full model. \n')
      
      listVarAllLevels <- extractLevelDT(inputDT[,.SD,.SDcol = c(unique(unlist(strsplit(varsGA, '[*]'))))])
      nbTotalLevels <- (sum(length(unlist(listVarAllLevels))) - 1)

      if(nrow(inputDT[trainSamp,])*1.05 > nbTotalLevels*5){ # On tolere 5pc d erreur.

        if(!is.null(updateOffsetZoneGen) || !is.null(zoneAgeFormGen)){
          formFull <- functionMakeFormula(distModel, listVecVariables = list(varsGA), offset = NULL, zoneAgeForm = TRUE)
        } else {
          formFull <- functionMakeFormula(distModel, listVecVariables = list(varsGA), offset = NULL)
        }

        fitFull <- functionFitIter(formFull, distModel, inputDT, trainSamp, testSamp, nCores, gamlssModel, upperBoundPred)
        MSEFull <- sum((fitFull$predModel - inputDT$observed[testSamp])^2*inputDT$exposure[testSamp], na.rm = TRUE)/sum(inputDT$exposure[testSamp], na.rm = TRUE)
        inputDT[testSamp, predicted := fitFull$predModel]
        
        expSplits <- unique(quantile(inputDT[testSamp, exposure], seq(0.01, 0.99, 0.01)))[-1]
        quantSplits <- seq(0.05, 0.95, 0.05)
        CPFull <- concProbGrid(inputDT[testSamp, ], lowCat = concProbLow, highCat = concProbHigh, expSplits, quantSplits)$concProbGlobal
        
        inputDT[, predicted := NULL] 
        
      } else {
        
        CPFull <- 1
        MSEFull <- 0

      }
    } else {
      MSEFull <- NULL
      MSENull <- NULL 
      CPFull <- NULL
      CPNull <- NULL
    }
        
    cat('MSE Full model : ', MSEFull, '\n')
    cat('MSE Null model : ', MSENull, '\n')
    cat('C-index Full model : ', CPFull, '\n')
    cat('C-index Null model : ', CPNull, '\n')

    MSEFullNew <- MSEFull
    CPFullNew <- CPFull
        
    #
    #  Initialization : first generation.
    #
    
    nVar <- length(varsGA)
    
    listSampleInit <- llply(as.list(rep(nVar, nModelsInit)), function(xx){sort(sample(1:xx, nVarInit),decreasing = FALSE)})
    listModelsInit <- llply(listSampleInit, function(xx){varsGA[xx]})
    matBinInit <- matrix(rep(0,nVar*nModelsInit), nrow = nModelsInit)
    for (iRow in 1:nModelsInit){
      matBinInit[iRow,listSampleInit[[iRow]]] = rep(1,nVarInit) # On met les 1 ou les variables sont selectionnees, on laisse 0 ailleurs.
    }
    
    cat('Start initialization \n')
    
    procTime = proc.time()[[3]]
    
    fitInit <- functionFit(inputDT, listModelsInit, nVarMax, nRedModels, trainPerc, testPerc, distModel, nCores, 
                           concProbLow, concProbHigh, nEstConcProb, 
                           valOffSet, gamlssModel, MSEFull, MSENull, MSEFullNew, CPFull, CPNull, CPFullNew, ratioConcProbMSE, 
                           zoneAgeFormGen[1], upperBoundPred)
    
    cat('Time init : ', (proc.time()[[3]] - procTime)/60,'\n')
    
    listCIndexInit <- fitInit$listCIndex
    
    listIntInit <- fitInit$listInterModels
    
    prevGen <- 0 
    
    print('End initialization')

    MSEFullNew <- fitInit$MSEFullNew
    CPFullNew <- fitInit$CPFullNew

    MSEFull <- MSEFullNew
    CPFull <- CPFullNew
    
  } else {
        
    cat('list2Save != NULL : Start GA from previous generations (warm start). \n')
    
    matBinInit <- list2Save$matBinInit
    listCIndexInit <- list2Save$listCIndexInit
    listModelsFin <- list2Save$listModelsFin # Il s agit d une liste de liste avec tous les modeles, ainsi on garde acces a la totalite.
    # Tous les modeles pour chaque generation.
    listNewModelsFin <- list2Save$listNewModelsFin # Liste avec les modeles optimaux compte tenu de nVarMax.
    listCIndexFin <- list2Save$listCIndexFin # Pareil : Liste des C index pour chaque modele de chaque generation.
    listPoidsCIndexFin <- list2Save$listPoidsCIndexFin # Liste des poids pour le CIndex, pour classer les modeles.
    listMatReprodFin <- list2Save$listMatReprodFin # Liste des matrices apres reproduction
    listMatMutFin <- list2Save$listMatMutFin # Liste des matrices apres mutation.
    listMatFitFin <- list2Save$listMatFitFin # Liste des matrices apres fitting/C-index.
    listvarsGAFin <- list2Save$listvarsGAFin # Liste des Variables : Besoin de ca car on change l ordre des variables durant la reproduction.
    listIntFin <- list2Save$listIntFin # Liste des interactions pour chaque generation.
    MSEFull <- list2Save$MSEFull
    MSENull <- list2Save$MSENull
    MSEFullNew <- list2Save$MSEFullNew
    CPFull <- list2Save$CPFull
    CPNull <- list2Save$CPNull
    CPFullNew <- list2Save$CPFullNew
    
    if (!is.null(list2Save$newvalOffSet)){
      inputDT[, newvalOffSet := list2Save$newvalOffSet]
    }
    
    prevGen <- list2Save$iGen
    
  }
  
  for(iGen in 1:nGens){
    
    iGen <- iGen + prevGen
    cat('iGen :', iGen, '\n')
    procTime = proc.time()[[3]]
    
    if(!is.null(updateOffsetZoneGen) && (sum(updateOffsetZoneGen[1:(iGen - prevGen)])>0)){valOffSet <- 'newvalOffSet'} 
    
    if(iGen > 1){
      
      resGA <- iterGA(inputDT, listvarsGAFin[[iGen-1]], listMatFitFin[[iGen-1]], listCIndexFin, nCrossOverGen[iGen - prevGen], nMutationsGen[iGen - prevGen], 
                      nModelsGen[iGen - prevGen], nVarMax, nRedModels, trainPerc, testPerc, distModel, nCores, 
                      concProbLow, concProbHigh, nEstConcProb,
                      iGen, listNewModelsFin, nAddedBestModelsGen[iGen - prevGen], valOffSet,
                      nMinInOptModsGen[iGen - prevGen],
                      includeIntsGen[iGen - prevGen], nModelsIntGen[iGen - prevGen], nMaxIntGen[iGen - prevGen], listIntFin, 
                      updateOffsetZoneGen[iGen - prevGen], gamlssModel,
                      MSEFull, MSENull, MSEFullNew, CPFull, CPNull, CPFullNew, ratioConcProbMSE, zoneAgeFormGen[iGen - prevGen], upperBoundPred, badOnes) # si x = NULL, x[1] = x[2] = ... = NULL -> OK.
      
    } else {
      
      resGA <- iterGA(inputDT, varsGA, matBinInit, listCIndexInit, nCrossOverGen[iGen - prevGen], nMutationsGen[iGen - prevGen], 
                      nModelsGen[iGen - prevGen], nVarMax, nRedModels, trainPerc, testPerc, distModel, nCores, 
                      concProbLow, concProbHigh, nEstConcProb,
                      iGen, listModelsInit, nAddedBestModelsGen[iGen - prevGen], valOffSet,
                      nMinInOptModsGen[iGen - prevGen],
                      includeIntsGen[iGen - prevGen], nModelsIntGen[iGen - prevGen], nMaxIntGen[iGen - prevGen], listIntInit, 
                      updateOffsetZoneGen[iGen - prevGen], gamlssModel,
                      MSEFull, MSENull, MSEFullNew, CPFull, CPNull, CPFullNew, ratioConcProbMSE, zoneAgeFormGen[iGen - prevGen], upperBoundPred, badOnes)
    }
    
    cat('Time iter : ', (proc.time()[[3]] - procTime)/60,'\n')
    
    listMatMutFin[[iGen]] <- resGA$matMut
    listMatReprodFin[[iGen]] <- resGA$matReprod
    listMatFitFin[[iGen]] <- resGA$matNewFit
    listCIndexFin[[iGen]] <- resGA$listCIndexIter
    listModelsFin[[iGen]] <- resGA$listModelsIter
    listNewModelsFin[[iGen]] <- resGA$listNewModelsIter
    listPoidsCIndexFin[[iGen]] <- resGA$listPoidsCIIter
    listvarsGAFin[[iGen]] <- resGA$varsGA
    listIntFin[[iGen]] <- resGA$listInterModels

    MSEFullNew <- resGA$MSEFullNew
    CPFullNew <- resGA$CPFullNew
    
    MSEFull <- MSEFullNew
    CPFull <- CPFullNew
        
    if (!is.null(partialSaveGen) && partialSaveGen[iGen - prevGen] == TRUE){
      list2Save <- list(matBinInit = matBinInit, listCIndexInit = listCIndexInit, 
                        listMatMutFin = listMatMutFin, listMatReprodFin = listMatReprodFin, listMatFitFin = listMatFitFin, listModelsFin = listModelsFin,
                        listNewModelsFin = listNewModelsFin, listCIndexFin = listCIndexFin,
                        listPoidsCIndexFin = listPoidsCIndexFin, listvarsGAFin = listvarsGAFin, 
                        listIntFin = listIntFin, newvalOffSet = inputDT$newvalOffSet, iGen = iGen, MSEFull = MSEFull, MSENull = MSENull, MSEFullNew = MSEFullNew, CPFull = CPFull, CPNull = CPNull, CPFullNew = CPFullNew) 
      if (is.null(partialSaveName)){
        save(list2Save, file = paste0(paste0('TempGA', iGen), '.RData'))
      }else{
        save(list2Save, file = paste0(paste0(partialSaveName, iGen), '.RData'))
      }
    }
  }
  
  return(list(matBinInit = matBinInit, listCIndexInit = listCIndexInit, 
              listMatMutFin = listMatMutFin, listMatReprodFin = listMatReprodFin, listMatFitFin = listMatFitFin, listModelsFin = listModelsFin,
              listNewModelsFin = listNewModelsFin, listCIndexFin = listCIndexFin,
              listPoidsCIndexFin = listPoidsCIndexFin, listvarsGAFin = listvarsGAFin, 
              listIntFin = listIntFin, newvalOffSet = inputDT$newvalOffSet, MSEFull = MSEFull, MSENull = MSENull, CPFull = CPFull, CPNull = CPNull))
  
}

# save.image(paste0(paste0('C:/Users/xrvrbk/Desktop/GA/workspace/intGAFr_MSE_concProb_SN', seedNumb), '.RData'))
# load(paste0(paste0('C:/Users/xrvrbk/Desktop/GA/workspace/intGAFr_MSE_concProb_SN', seedNumb), '.RData'))
