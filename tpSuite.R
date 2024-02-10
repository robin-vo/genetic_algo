

#' Adding the longitude and latitude coordinate to the input data set when the (Belgian) postal code is known
#'
#' @param inputDT data.table object containing the data of interest. This is an obligatory argument, without default value.
#' @param dirShape Character vector of length 1 indicating directory where all the shape files are stored. This is an obligatory argument, with NULL as default value.
#' @return The longitude and latitude coordinates are added as 2 separate columns to the 'inputDT' argument. Note that no output object is thrown by the 'addCoordinates' function, the 2 columns are directly added to the object that was passed to the 'inputDT' argument.
#' @examples
#'nSamples <- 50000
#'observedValues <- rpois(nSamples, lambda = 1)
#'library(data.table)
#'inputDT <- as.data.table(data.frame(exposure = round(runif(nSamples), digits = 2),
#'observed = observedValues,
#'postalCode = sample(c('1000', '3000', '5000'), nSamples, replace = TRUE)))
#'addCoordinates(inputDT)
#'#addCoordinates(inputDT, '/home/robin/bla/')

addCoordinates <- function(inputDT, dirShape = NULL){

  checkDT(inputDT, 'postalCode')
  if(!isNumericDT(inputDT, 'postalCode')) asNumericDT(inputDT, 'postalCode')
  checkCharVec(list(dirShape))

  checkLength(list(dirShape), 1)
  lastCharDir <- substr(dirShape, start = nchar(dirShape), stop = nchar(dirShape))
  if(lastCharDir == '\\' | lastCharDir == '/'){
    dirShape <- substr(dirShape, start = 1, stop = nchar(dirShape) - 1)
  }

  if(sum(names(inputDT) == 'LAT')) inputDT[, LAT := NULL]
  if(sum(names(inputDT) == 'LONG')) inputDT[, LONG := NULL]
  if(sum(names(inputDT) == 'lat')) inputDT[, lat := NULL]
  if(sum(names(inputDT) == 'long')) inputDT[, long := NULL]

  didItLoad <- suppressWarnings(try(postalCode <- as.data.table(read.csv(paste(dirShape, "zipcodes.csv", sep = "\\"))[,c("CODPOSS","LAT", "LONG")]), silent = TRUE))
  if('try-error' %in% class(didItLoad)) stop("Please set a valid directory to the argument 'dirShape'.")

  postalCode <- as.data.table(read.csv(paste(dirShape, "zipcodes.csv", sep = "\\"))[,c("CODPOSS","LAT", "LONG")])
  setnames(postalCode, c('CODPOSS', 'LONG', 'LAT'), c('postalCode', 'longitude', 'latitude'))
  setkey(inputDT, 'postalCode')
  setkey(postalCode, 'postalCode')
  return(merge(inputDT, postalCode, all.x = TRUE))
}

#' Categorization of continuous according to some splitting points.
#'
#' @param inputDT data.table object containing the data of interest. This is an obligatory argument, without default value.
#' @param splits List of numeric vectors, ideally being the output object of the 'groupingVars' function. The names of this object need to correspond to column names of the 'inputDT' argument. This is an obligatory argument, without default value.
#' @param removeAlreadyPresent Logical vector of length 1 indicating whether or not continuous variables of the 'inputDT' argument that were already categorized by this function during a previous call, and which are about to be recategorized again, should remove or not the categorization of the previous call. The default value is TRUE, and is an optional argument.
#' @return The variables whose names are found in the 'split' argument as names of the object that is passed to this argument, are splitted by means of the splitting points that are supplied by the same 'split' argument. The resulting variable is added to the object that is passed to the 'inputDT' argument, and the respective column name corresponds to the variable name augmented with the suffix 'Grouped', without any space or character in between. Note that the grouping is always performed in the following manner: if the 'split' argument supplies the splitting points a, b, and c, then the continuous variable is split up in the following intervals ]-infinity, a[, [a, b[ and [b, +infinity[.
#' @examples

transform2BinnedVar <- function(inputDT, splits, removeAlreadyPresent = TRUE){

  checkLogicVec(list(removeAlreadyPresent))
  if(!is.list(splits)) stop("The 'split' argument is a list of numeric vectors.")

  varNames <- names(splits)
  checkDT(inputDT, varNames)
  checkLogicVec(list(removeAlreadyPresent))
  checkLength(list(removeAlreadyPresent), 1)

  colsAlreadyTransformed <- which(names(inputDT) %in% paste(varNames, 'Grouped', sep = ''))
  if(length(colsAlreadyTransformed) > 0) set(inputDT, NULL, colsAlreadyTransformed, NULL)
  nColOriginal <- ncol(inputDT)
  varNamesOriginal <- names(inputDT)
  for(iVar in 1:length(varNames)){

    if(removeAlreadyPresent){
      alreadyPresentCol <- which(names(inputDT) %in% c(paste(varNames[iVar], 'Grouped', sep = ''), 'newcol'))
      if(length(alreadyPresentCol) != 0){
        set(inputDT, NULL, alreadyPresentCol, NULL)
        nColOriginal <- nColOriginal - 1
      }
    }
    selectedVar <- inputDT[, .SD, .SDcol = which(varNamesOriginal == varNames[iVar])]
    selectedVarGrouped <- selectedVar
    if(sum(names(splits[[iVar]]) %in% c('splits')) > 0){
      selectedSplits <- sort(splits[[iVar]]$splits)
    } else {
      selectedSplits <- sort(splits[[iVar]])
    }
    if(sum(is.na(selectedSplits)) == 0){
      nGroups <- length(selectedSplits) + 1
      for(iSplit in (nGroups - 1):1){
        selectedVarGrouped[selectedVar < selectedSplits[iSplit]] <- iSplit
      }
      selectedVarGrouped[selectedVar >= selectedSplits[nGroups-1]] <- nGroups

      inputDT[ , newcol := character(.N)]
      set(inputDT, NULL, as.integer(nColOriginal + iVar), as.factor(selectedVarGrouped[[1]]))
      setnames(inputDT, 'newcol', paste(varNames[iVar], 'Grouped', sep = ''))
    } else {
      warning(paste("No splits were provided by the 'splits' argument for the variable ", varNames[iVar], sep = ''))
    }
  }
}

checkPlotDirAndName <- function(plotDir = NULL, plotName = NULL){
  if((is.null(plotDir) & !is.null(plotName)) | (!is.null(plotDir) & is.null(plotName))) stop('If one want to plot the figure automatically, then one needs to provide a value for both the "plotDir" argument and the "plotName" argument.')
  if(!is.null(plotName)){
    checkCharVec(list(plotName))
    checkLength(list(plotName), 1)
    if(length(unlist(strsplit(plotName, '[.]'))) > 1) stop('The "plotName" argument should not contain the extension.')
  }
  if(!is.null(plotDir)){
    checkCharVec(list(plotDir))
    checkLength(list(plotDir), 1)
    if(str_sub(plotDir, start= -1) != '/') plotDir <- paste(plotDir, '/', sep = "")
  }
}

is.ggplotObject <- function(ggplotObject){
  sum(names(ggplotObject) %in% c("data", "layers", "scales", "mapping", "theme", "coordinates", "facet", "plot_env", "labels")) == 9
}


#' Constructing a ggplot-styled map of the Belgium, splitted up by INS region, to represent a zone by means of a continuous variable or simply the exposure
#'
#' @param inputDT data.table object containing the data of interest. Note that the 'inputDT' argument needs to contain an 'exposure ' and 'postalCode' column in order to the function to work. This is an obligatory argument, without default value.
#' @param predictions Numeric vector of length corresponding to the number of rows of the 'inputDT' argument. This argument correspond to the predictions (mostly the spline covariate effect estimates) made for every observation of the 'inputDT' argument. In case that the NULL value is supplied to this argument, the exposure is plotted. This is an optional argument, without default value.
#' @param plotTitle Character vector of length 1 consisting of the plot title of the plot. This argument needs to be supplied in conjunction with the 'plotDir' argument. This is an optional argument, with '' as default value.
#' @param dirShape Character vector of length 1 consisting of the directory where one can find the shape files. This is an optional argument, with NULL as default value.
#' @param plotDir Character vector of length 1 consisting of the directory where one wants to save the ggplot object to. This argument needs to be supplied in conjunction with the 'plotName' argument. This is an optional argument, with NULL as default value.
#' @param plotName Character vector of length 1 consisting of the file name of the saved ggplot object to. This argument needs to be supplied in conjunction with the 'plotDir' argument. This is an optional argument, with NULL as default value.
#' @param pdf2 Logical vector of length 1 indicating whether or not one wants to save the ggplot object as a pdf as well, on top of the png version of the ggplot object which is produced in any case. This is an optional argument, with TRUE as default value.
#' @param lowColor Character vector of length 1 indicating the color of the lowest value of the scale of the 'predictions' argument. This argument will only be taken into account if a valid value is also provided for the 'highColor' argument, and when the 'predictions' argument is not considered to be a factor. This is an optional argument, with NULL as default value.
#' @param highColor Character vector of length 1 indicating the color of the highest value of the scale of the 'predictions' argument. This argument will only be taken into account if a valid value is also provided for the 'lowColor' argument, and when the 'predictions' argument is not considered to be a factor. This is an optional argument, with NULL as default value.
#' @param predAsFact Logical vector of length 1 indicating whether or the 'predictions' argument needs to be considered as a continuous or a factor variable. This is an optional argument, with FALSE as default value.
#' @param predFactColors Character vector of length equal to the number of different values of the 'predictions' argument. Only if the 'predAsFact' argument is toggled to TRUE, this argument is taken into account. This is an optional argument, with NULL as default value.
#' @param randomPick Logical vector of length 1 indicating whether one wants to let a random color generator decide upon the color of the different levels of the 'zoneGrouped' factor column of the 'inputDT' argument. If this argument is set to be TRUE, this function will generate an output object that should be used as an input for the first argument of the 'randomColorPick' function (of this package). This is an optional argument, with FALSE as default value.
#' @return A ggplot object is constructed with a continuous color scheme, plotting the exposure or the predictions for the different INS regions. Note that the 'inputDT' argument needs to contain an 'exposure ' and 'postalCode' column in order to the function to work.
#' @examples
#' library(data.table)
#' dirShape <- '/home/robin/shapeFiles/'
#' postalCode <- read.csv(paste(dirShape, "postins.csv", sep = "\\"))[,c("CODPOSS","INS")]
#' inputDT <- data.table(postalCode = postalCode$CODPOSS, exposure = runif(nrow(postalCode)))
#' predictions <- as.numeric(as.character(postalCode$CODPOSS))
#' plottingSpatialEffect(inputDT)
#' plottingSpatialEffect(inputDT, predictions, dirShape = dirShape)
#' plottingSpatialEffect(inputDT, predictions, lowColor = 'red', highColor = 'green', dirShape = dirShape)
#' plottingSpatialEffect(inputDT, predictions, lowColor = 'green', highColor = 'purple', dirShape = dirShape)
#'
#' predictions <- sample(LETTERS[1:10], nrow(postalCode), replace = TRUE)
#' plottingSpatialEffect(inputDT, predictions, predAsFact = TRUE, dirShape = dirShape)

plottingSpatialEffect <- function(inputDT, predictions = NULL, plotTitle = '', dirShape = NULL, plotDir = NULL, plotName = NULL, pdf2 = TRUE, lowColor = NULL, highColor = NULL, predAsFact = FALSE, predFactColors = NULL, randomPick = FALSE){

  checkDT(inputDT, c('exposure', 'postalCode'))
  checkLength(list(pdf2, randomPick, predAsFact, dirShape), 1)
  if(!is.null(plotTitle)) checkLength(list(plotTitle), 1)
  if(!is.null(dirShape)) checkLength(list(dirShape), 1)
  if(!is.null(plotDir)) checkLength(list(plotDir), 1)
  if(!is.null(plotName)) checkLength(list(plotName), 1)
  checkPlotDirAndName(plotDir, plotName)

  if(!is.null(predictions) & !predAsFact) checkNumVec(list(predictions))
  if(!is.null(predictions) & predAsFact) if(!is.factor(predictions)) stop("Please transform argument 'predictions' to a factor.")
  checkCharVec(list(plotTitle, dirShape))
  checkLogicVec(list(pdf2, randomPick, predAsFact))

  if(!is.null(highColor)){ checkLength(list(highColor), 1) ; if(sum(highColor %in% colors()) == 0) stop('Please provide a valid color name for the "highColor" argument. The valid values can be found by running the "colors()" function on the command line.')}
  if(!is.null(lowColor)){ checkLength(list(lowColor), 1) ; if(sum(lowColor %in% colors()) == 0) stop('Please provide a valid color name for the "lowColor" argument. The valid values can be found by running the "colors()" function on the command line.')}
  if(!is.null(lowColor) & is.null(highColor)) stop('Both "lowColor" and "highColor" either need to be NULL, either need to have a valid color name. The valid values can be found by running the "colors()" function on the command line.')
  if(is.null(lowColor) & !is.null(highColor)) stop('Both "lowColor" and "highColor" either need to be NULL, either need to have a valid color name. The valid values can be found by running the "colors()" function on the command line.')

  lastCharDir <- substr(dirShape, start = nchar(dirShape), stop = nchar(dirShape))
  while(lastCharDir == '\\' | lastCharDir == '/'){
    dirShape <- substr(dirShape, start = 1, stop = nchar(dirShape) - 1)
    lastCharDir <- substr(dirShape, start = nchar(dirShape), stop = nchar(dirShape))
  }

  belgium <- readOGR(dsn = dirShape, layer = 'communes_08', verbose = FALSE)
  postalCode <- read.csv(file.path(dirShape, "postins.csv"))[,c("CODPOSS","INS")]
  belgium@data <- left_join(belgium@data, postalCode, by = 'INS')

  exposure <- aggregate(exposure ~ postalCode, FUN = sum, data = inputDT)
  exposure$postalCode <- as.numeric(as.character(exposure$postalCode))
  names(belgium@data)[7] <- "postalCode"
  belgium@data <- left_join(belgium@data, exposure, by = 'postalCode')
  for(i in 1:nrow(belgium)) belgium$relativeExposure[i] <- belgium$exposure[i]/sum(belgium$exposure)
  belgiumFortified <- suppressMessages(fortify(belgium))
  belgium$id <- row.names(belgium)
  belgiumFortified <- left_join(belgiumFortified, belgium@data, by = 'id')

  if(is.null(predictions)){

    mapExposure <- ggplot(belgiumFortified, aes(long, lat, group = group, fill = exposure, text = paste(paste(paste('Postal Code: ', postalCode, sep = ''), ' -- ', sep = ''), Name1, sep = ''))) + xlab('Longitude') + ylab('Latitude') + geom_polygon() +
      ggtitle(plotTitle) + scale_fill_continuous(name = '')
    if(!is.null(lowColor) & !is.null(highColor)){
      mapExposure <- mapExposure + scale_colour_gradient(low = lowColor, high = highColor, name = "")
    } else {
      mapExposure <- mapExposure + scale_colour_gradient(name = "")
    }
    if(!is.null(plotDir) & !is.null(plotName)){
      if(pdf2) ggsave(paste(plotDir, paste(plotName, '.pdf', sep = ''), sep = ""))
      ggsave(paste(plotDir, paste(plotName, '.png', sep = ''), sep = ""))
    }
    return(mapExposure)
  } else {

    checkLength(list(predictions), nrow(inputDT))
    if(sum(names(inputDT) == 'predicted') == 1) inputDT[, predicted := NULL] #part of debug
    inputDT$predicted <- predictions
    predicted <- aggregate(predicted ~ postalCode, FUN = getMode, data = inputDT)
    #predicted <- aggregate(predicted ~ postalCode, FUN = mean, data = inputDT)
    predicted$postalCode <- as.numeric(as.character(predicted$postalCode))
    belgiumFortified <- left_join(belgiumFortified, predicted, by = 'postalCode')

    if(!predAsFact){
      mapPredictions <- ggplot(belgiumFortified, aes(long, lat, group = group, fill = predicted, text = paste(paste(paste('Postal Code: ', postalCode, sep = ''), ' -- ', sep = ''), Name1, sep = ''))) + xlab('Longitude') + ylab('Latitude') + geom_polygon() + ggtitle(plotTitle)
      if(!is.null(lowColor) & !is.null(highColor)){
        mapPredictions <- mapPredictions + scale_fill_continuous(low = lowColor, high = highColor, name = "")
      } else {
        mapPredictions <- mapPredictions + scale_fill_continuous(name = "")
      }
    } else {
      if(length(unique(predictions)) > 20) warning('The number of unique values of the "predictions" argument is larger than 20. Probably it is not a good idea to treat this variable as a factor.')
      if(!is.null(predFactColors)){
        if(length(unique(predictions)) != length(predFactColors)) stop('The number of colours in the "predFactColors" argument should correspond to the number of different values of the "predictions" argument.')
        if(sum(predFactColors %in% colors()) != length(unique(predictions))) stop('Some of the values of the "predFactColors" argument are not valid color names. The valid values can be found by running the "colors()" function on the command line.')
      } else {
        colorPalette <- colors()[-grep('white', colors())]
        colorPalette <- colorPalette[-grep('grey', colorPalette)]
        colorPalette <- colorPalette[-grep('gray', colorPalette)]
        colorPalette <- colorPalette[-grep('ivory', colorPalette)]
        colorPalette <- colorPalette[-grep('silk', colorPalette)]
        colorPalette <- colorPalette[-grep('bisque', colorPalette)]
        colorPalette <- colorPalette[-grep('lightyellow', colorPalette)]
        colorPalette <- colorPalette[-grep('honeydew', colorPalette)]
        colorPalette <- colorPalette[-grep('palegolden', colorPalette)]
        colorPalette <- colorPalette[-grep('mint', colorPalette)]
        colorPalette <- colorPalette[-grep('lavender', colorPalette)]
        colorPalette <- colorPalette[-grep('lightcyan', colorPalette)]
        colorPalette <- colorPalette[-grep('lemon', colorPalette)]
        colorPalette <- colorPalette[-grep('linen', colorPalette)]
        colorPalette <- colorPalette[-grep('snow', colorPalette)]
        colorPalette <- colorPalette[-grep('almond', colorPalette)]
        colorPalette <- colorPalette[-grep('wheat', colorPalette)]
        colorPalette <- colorPalette[-grep('tan', colorPalette)]
        colorPalette <- colorPalette[-grep('azure', colorPalette)]
        colorPalette <- colorPalette[-grep('papaya', colorPalette)]
        colorPalette <- colorPalette[-grep('lightgolden', colorPalette)]

        predFactColors <- colorPalette[sample(1:length(colorPalette), length(unique(predictions)))]
        warnings("If you don't like the randomly chosen color palette, enter 'colors()' in the command line to see the possible colors and rerun this function by supplying valid color names to the 'predFactColors' argument.")
      }
      mapPredictions <- ggplot(belgiumFortified, aes(long, lat, group = group, fill = factor(predicted), text = paste(paste(paste('Postal Code: ', postalCode, sep = ''), ' -- ', sep = ''), Name1, sep = ''))) + xlab('Longitude') + ylab('Latitude') + scale_fill_manual(name = "Predictions", values = predFactColors) + geom_polygon() + ggtitle(paste(plotTitle, ': Predictions', sep = ''))
    }

    if(!is.null(plotDir) & !is.null(plotName)){
      if(pdf2) ggsave(paste(plotDir, paste(plotName, '.pdf', sep = ''), sep = ""))
      ggsave(paste(plotDir, paste(plotName, '.png', sep = ''), sep = ""))
    }
    if(randomPick == TRUE & predAsFact == TRUE){
      return(list(belgiumFortified = belgiumFortified, plotTitle = plotTitle, plotDir = plotDir, plotName = plotName, pdf2 = pdf2))
    } else {
      return(mapPredictions)
    }
  }
}

binningUnivSpline <- function(inputDT, nGroups = NULL, sampleSizeBin = min(nrow(inputDT), 50000), nSamples = 10, maxIter = 500){

  checkDT(inputDT, c('y', 'x'))
  if(is.null(nGroups)) stop("A value needs to be passed to the 'nGroups' argument.")
  checkNumVec(as.list(sampleSizeBin, nSamples, maxIter))

  breakPoints <- matrix(NA, nrow = nSamples, ncol = nGroups - 1)
  for(iSample in 1:nSamples){
    selectedIndices <- sample(1:nrow(inputDT), sampleSizeBin)
    complexityParam <- 1
    nIters <- 1
    notDone <- TRUE
    while(notDone){
      resultTree <- rpart(y ~ x, data = inputDT[selectedIndices,], cp = complexityParam)
      splitsTree <- sort(unique(unlist(rpart.lists(resultTree))))
      if(is.character(splitsTree) | (length(splitsTree) + 1) < nGroups){
        complexityParam <- complexityParam*0.5
      } else if((length(splitsTree) + 1) > nGroups){
        complexityParam <- complexityParam*1.5
      } else{
        notDone <- FALSE
      }
      if(nIters == maxIter) notDone <- FALSE
      nIters <- nIters + 1
    }
    if(nIters <= maxIter) breakPoints[iSample, ] <- splitsTree
  }
  library(cluster)
  if(sum(is.na(breakPoints)) == prod(dim(breakPoints))){
    finalBreakPoints <- NA
  } else {
    rows2Keep <- which(aaply(breakPoints, 1, function(xx) sum(is.na(xx)) == 0))
    finalBreakPoints <- pam(breakPoints[rows2Keep,], 1)$medoids
  }
  return(as.numeric(finalBreakPoints))
}

#do also add an example to the 'transform2BinnedVar' function of the dataPrep package, this is not done yet. The output object of this function is the input for the 'splits' argument of the latter function.

groupingVars <- function(var2BeGrouped, nGroups, gamFit, inputDT, splineEst, sampleSizeBin = min(50000, nrow(inputDT)), nSamples = 10, maxIter = 500, xLimit = NULL, showPlot = FALSE, plotDir = NULL, plotName = NULL, pdf2 = TRUE){

  checkLength(list(var2BeGrouped, sampleSizeBin, showPlot, pdf2), 1)
  if(!is.null(xLimit)){
    checkLength(list(xLimit), 2)
    checkNumOrIntVec(list(xLimit))
  }

  checkCharVec(list(var2BeGrouped))
  checkDT(inputDT, var2BeGrouped)
  checkPlotDirAndName(plotDir, plotName)
  checkLogicVec(list(showPlot, pdf2))
  checkNumOrIntVec(list(nGroups, sampleSizeBin, nSamples, maxIter))

  checkRanges(list(nGroups), list(c('>', 1)))
  checkRanges(list(sampleSizeBin), list(c('>', 1)))
  checkWholeNumb(list(nGroups, sampleSizeBin, nSamples, maxIter))

  if(sum(class(gamFit) %in% c('gam', 'bam')) == 0) stop('The "gamFit" argument should be an output object of the mgcv::[g,b]am function.')
  allSplineVar <- unlist(llply(gamFit$smooth, function(xx) xx$term))
  for(elementSplineEst in names(splineEst)) checkValues(list(elementSplineEst), list(allSplineVar))
  checkValues(list(var2BeGrouped), list(allSplineVar))
  checkValues(list(var2BeGrouped), list(names(splineEst)))

  #inputSplit <- extractData4ModelFit(inputDT, var2BeGrouped, NULL, NULL)
  inputSplit <- inputDT[as.integer(rownames(splineEst[[which(names(splineEst) == var2BeGrouped)]])), ]
  inputSplit <- inputSplit[, .SD, .SDcol = which(names(inputSplit) == var2BeGrouped)]
  names(inputSplit) <- 'x'
  inputSplit[, y := exp(splineEst[[which(names(splineEst) == var2BeGrouped)]])]
  splitsTree <- binningUnivSpline(inputSplit, nGroups, sampleSizeBin = min(nrow(inputDT), sampleSizeBin), nSamples = nSamples, maxIter = maxIter)

  if(!sum(is.na(splitsTree))){
    library(ggplot2)
    ggplotObject <- plotUnivSpline(gamFit, inputDT, var2BeGrouped, simultaneousCI = FALSE, transparence = 0.5, colorLine = 'black', colorRibbon = 'steelblue', lineWidth = 1.2, showPlot = TRUE, nSamples = 1000, xLimits = xLimit)
    ggplotObject <- ggplotObject[[1]] + geom_vline(xintercept = splitsTree, color = 'forestgreen', size = 1.1)

    plotName <- sprintf("binned %s with %d groups", var2BeGrouped, nGroups)
    if(showPlot == TRUE) print(ggplotObject)
    if(!is.null(plotDir)){
      if(pdf2) ggsave(paste(plotDir, paste(plotName, '.pdf', sep = ''), sep = ""))
      ggsave(paste(plotDir, paste(plotName, '.png', sep = ''), sep = ""))
    }
    return(list(ggplotObject = ggplotObject, splits = splitsTree))
  } else {
    warning(paste(paste(paste(paste('Variable ', var2BeGrouped, sep = ''), ' could not be split into ', sep = ''), nGroups, sep = ''), ' groups.', sep = ''))
  }
}

groupingSpatialSpline <- function(splineEst, nGroups = NULL, sampleSizeBin = 50000, nSamples = 10, verbose = TRUE, asList = TRUE){

  if(is.null(nGroups)) stop("A value needs to be passed to the 'nGroups' argument.")
  checkNumVec(as.list(splineEst, sampleSizeBin, nSamples))
  checkLength(list(nGroups), 1)
  sampleSizeBin = min(length(splineEst), sampleSizeBin)

  breakPoints <- matrix(NA, nrow = nSamples, ncol = nGroups - 1)
  for(iSample in 1:nSamples){
    if(verbose) print(paste('Run', iSample, sep = ''))
    selectedIndices <- sample(1:length(splineEst), sampleSizeBin)
    fisherResult <- classIntervals(splineEst[selectedIndices], n = nGroups, style = 'fisher')
    breakPoints[iSample, ] <- fisherResult$brks[-c(1, length(fisherResult$brks))]
  }
  finalBreakPoints <- pam(breakPoints, 1)$medoids

  if(asList == TRUE){
    finalResult <- list()
    finalResult[[1]] <- finalBreakPoints
    names(finalResult) <- 'zone'
    return(finalResult)
  } else {
    return(as.numeric(finalBreakPoints))
  }
}

binningSpatialEffect <- function(inputDT, gamFit, nGroups = 5, dirShape = NULL, keyWordColSelection = NULL, sampleSizeBin = min(nrow(inputDT), 50000), nSamples = 10, verbose = FALSE, asList = TRUE, addVar = FALSE){

  checkDT(inputDT)
  if(sum(class(gamFit) %in% c('gam', 'bam')) == 0) stop('The "gamFit" argument should be an output object of the mgcv::gam function.')
  checkLength(list(nGroups), 1)
  for(iGroup in nGroups) checkRanges(list(iGroup), list(c('>', 0)))
  checkWholeNumb(list(nGroups, sampleSizeBin))

  if(is.null(keyWordColSelection)) keyWordColSelection <- 'latitude'
  if(sum(names(inputDT) %in% c('latitude', 'longitude')) != 2){
    addCoordinates(inputDT, dirShape = dirShape)
  }
  splineEst <- extractSplineEstimate(gamFit, keyWordColSelection)
  splineEst <- unlist(splineEst[which(names(splineEst) == keyWordColSelection)])
  splitsSpatial <- groupingSpatialSpline(splineEst = splineEst, nGroups = nGroups, sampleSizeBin = sampleSizeBin, nSamples = nSamples, verbose = verbose, asList = TRUE) #classicTP
  inputDT[, zone := splineEst]
  if(!addVar){
    transform2BinnedVar(inputDT, splitsSpatial)
  }
  return(list(splits = splitsSpatial))
}

# gamModelFit <- function(inputDT, splineVarNames = NULL, nonSplineVarNames = NULL, interactionTerms = NULL, model = NULL, nCores = 1, varSelection = FALSE, totalCost = TRUE, dirShape = NULL, chunckSize = 5000, addBivarSpatialSpline = FALSE){
#
#   checkDT(inputDT, c('exposure', 'claimNumber', 'claimSize'))
#
#   if(sum(inputDT$exposure == 0) != 0) stop("The 'exposure' column of the 'inputDT' argument can not have a value equal to 0.")
#   if(sum(inputDT$exposure < 0) != 0) stop("The 'exposure' column of the 'inputDT' argument can not have negative values.")
#   if(sum(inputDT$exposure > 1) != 0) stop("The 'exposure' column of the 'inputDT' argument can not have values higher than 1.")
#
#   #checkNumVec(as.list(inputDT[,.SD,.SDcols = which(names(inputDT) %in% c('exposure', 'claimNumber', 'claimSize'))]))
#   checkCharVec(list(nonSplineVarNames, splineVarNames, interactionTerms, model))
#   if(!is.null(dirShape)) checkCharVec(list(dirShape))
#   checkLogicVec(list(totalCost))
#   model <- tolower(model)
#   if(length(match(model, c('poisson', 'gamma'))) != 1) stop("The 'model' argument should either be 'gamma' or 'poisson'.")
#   if(!is.null(splineVarNames)){
#     checkDT(inputDT, splineVarNames)
#     checkNumVec(as.list(inputDT[,.SD,.SDcols = which(names(inputDT) %in% splineVarNames)]))
#     splineVarNames <- setdiff(splineVarNames, c('longitude', 'latitude', 'LONGITUDE', 'LATITUDE', 'long', 'lat', 'LONG', 'LAT'))
#   }
#   #if(is.null(nonSplineVarNames)){
#   #  nonSplineVarNames <- setdiff(names(inputDT), c('exposure', 'claimNumber', 'claimSize', splineVarNames))
#   #} else {
#   if(!is.null(nonSplineVarNames)){
#     checkDT(inputDT, nonSplineVarNames)
#     if(length(setdiff(splineVarNames, nonSplineVarNames)) != length(splineVarNames)) stop("The 'splineVarNames' and 'nonSplineVarNames' are not permitted to contain the same variable name.")
#   }
#   if(sum(names(inputDT) == 'postalCode') & addBivarSpatialSpline){
#     if(sum(names(inputDT) %in% c('longitude', 'latitude')) < 2) inputDT <- addCoordinates(inputDT, dirShape)
#     inputDT[, postalCode := NULL]
#   }
#   VarPartFormulaModel <- ''
#   if(length(nonSplineVarNames) != 0){
#     mainEffectsVar <- setdiff(nonSplineVarNames, c('longitude', 'latitude', 'postalCode', splineVarNames))
#     if(length(mainEffectsVar) != 0) VarPartFormulaModel <- paste(mainEffectsVar, collapse = ' + ')
#     if(!is.null(interactionTerms)){
#       if(sum(grepl('*', interactionTerms)) != length(interactionTerms)) stop("Not every element of the 'interactionTerms' argument is a valid interaction term.")
#       interactionTermsVars <- strsplit(interactionTerms, '\\*')
#       for(iInt in 1:length(interactionTerms)){
#         if(sum(mainEffectsVar %in% interactionTermsVars[[iInt]]) != 2) stop(paste(paste("The element ", interactionTerms[[iInt]], sep = ''), " of the 'interactionTerms' argument contains variable names that are not part of the valid main effect variables (being part of the 'inputDT' argument, yet not of the 'splineVarNames' argument).", sep = ''))
#       }
#       VarPartFormulaModel <- paste(paste(VarPartFormulaModel, ' + ', sep = ''), paste(interactionTerms, collapse = ' + '), sep = '')
#     }
#   }
#   if(length(splineVarNames) != 0){
#     VarPartFormulaSplines <- aaply(splineVarNames, 1, function(xx) paste(paste('s(', xx, sep = ''), ')', sep = ''))
#     VarPartFormulaModel <- paste(VarPartFormulaModel, paste(' + ', paste(VarPartFormulaSplines, collapse = ' + '), sep = ''), sep = '')
#   }
#   if(sum(names(inputDT) %in% c('longitude', 'latitude')) == 2 & addBivarSpatialSpline){
#     VarPartFormulaModel <- paste(VarPartFormulaModel, ' + s(longitude, latitude)', sep = '')
#   }
#   createdCluster <- makeCluster(min(detectCores(), nCores))
#
#   if(model == 'poisson'){
#     formulaModel <- as.formula(paste(paste('claimNumber ~ ', VarPartFormulaModel, sep = ''), ' + offset(exposure)', sep = ''))
#     fitModel <- bam(formulaModel, data = inputDT, family = poisson(link = log), chunk.size = min(5000, nrow(inputDT)), cluster = createdCluster, select = varSelection)
#     #fitModel <- bam(formulaModel, data = inputDT, family = poisson(link = log), chunk.size = min(5000, nrow(inputDT)), cluster = createdCluster, keepData = TRUE)
#   } else if(model == 'gamma'){
#     formulaModel <- as.formula(paste('claimSize ~ ', VarPartFormulaModel, sep = ''))
#     if(totalCost == TRUE) inputDT[claimNumber > 1, claimSize := claimSize/claimNumber]
#     fitModel <- bam(formulaModel, weights = claimNumber, data = inputDT[claimNumber > 0,], family = Gamma(link = log), chunk.size = min(chunckSize, nrow(inputDT)), cluster = createdCluster, select = varSelection)
#     #fitModel <- bam(formulaModel, weights = claimNumber, data = inputDT[claimNumber > 0,], family = Gamma(link = log), chunk.size = min(chunckSize, nrow(inputDT)), cluster = createdCluster, keepData = TRUE)
#   }
#
#   stopCluster(createdCluster)
#   return(fitModel)
# }
#
# plotBivarSpatialSpline <- function(gamFit, inputDT, keyWordColSelection = 'latitude', returnSplineEstimates = FALSE, plotTitle = '', dirShape = NULL, plotDir = NULL, plotName = NULL, pdf2 = TRUE, lowColor = NULL, highColor = NULL, predAsFact = FALSE, predFactColors = NULL){
#
#   checkPlotDirAndName(plotDir, plotName)
#   checkDT(inputDT, c('latitude'))
#   if(sum(class(gamFit) %in% 'gam') == 0) stop('The "gamFit" argument should be an output object of the mgcv::gam function.')
#
#   gamFitModelMatrix <- model.matrix(gamFit)
#   selectionCols <- grep(keyWordColSelection, names(gamFit$coefficients))
#   fittedSpline <- gamFitModelMatrix[, selectionCols]%*%gamFit$coefficients[selectionCols]
#   if(is.null(gamFit$na.action)){
#     plottingSpatialEffect(inputDT, fittedSpline, plotTitle = plotTitle, dirShape = dirShape, plotDir = plotDir, plotName = plotName, pdf2 = pdf2, lowColor = lowColor, highColor = highColor, predAsFact = predAsFact, predFactColors = predFactColors)
#   } else {
#     plottingSpatialEffect(inputDT[-gamFit$na.action,], fittedSpline, plotTitle = plotTitle, dirShape = dirShape, plotDir = plotDir, plotName = plotName, pdf2 = pdf2, lowColor = lowColor, highColor = highColor, predAsFact = predAsFact, predFactColors = predFactColors)
#   }
#   if(returnSplineEstimates) return(fittedSpline)
# }

plotUnivSpline <- function(gamFit, inputDT, selectedVar, simultaneousCI = FALSE, transparence = 0.5, colorLine = 'black', colorRibbon = 'steelblue', lineWidth = 1.2, showPlot = TRUE, nSamples = 1000, plotDir = NULL, plotName = NULL, pdf2 = TRUE, xLimits = NULL, yLimits = NULL){

  checkPlotDirAndName(plotDir, plotName)

  if(sum(class(gamFit) %in% 'gam') == 0) stop('The "gamFit" argument should be an output object of the mgcv::gam function.')
  if(!is.data.table(inputDT)) stop('The "inputDT" argument should be a data.table object.')
  if(length(selectedVar) < 1 & selectedVar %in% names(inputDT) != length(selectedVar)) stop('The "selectedVar" argument should be a character vector consisting of column names of the "inputDT" argument.')

  if(!is.null(xLimits)){
    if(length(xLimits) != 2 | !is.numeric(xLimits)) stop('The "xLimits" argument should be a numeric vector of length 2.')
  }
  if(!is.null(yLimits)){
    if(length(yLimits) != 2 | !is.numeric(yLimits)) stop('The "yLimits" argument should be a numeric vector of length 2.')
  }

  for(iVar in 1:length(selectedVar)){
    varNumber <- 1
    goOn <- TRUE
    while(goOn){
      modelVar <- try(attr(gamFit$terms[varNumber], 'term.labels'), silent = TRUE)
      if(class(modelVar) == "try-error") stop(paste(selectedVar[iVar], 'is not part of the model variables of the "gamFit" argument.'))
      if(selectedVar[iVar] == modelVar) goOn <- FALSE
      varNumber <- varNumber + 1
    }
  }

  includedVars <- attr(gamFit$terms, 'dataClasses')
  numericVarNames <- names(includedVars)[which(includedVars == 'numeric')]
  numericVarNames <- numericVarNames[!grepl('offset', numericVarNames)]
  categoricalVarNames <- names(gamFit$xlevels)

  referenceClass <- laply(gamFit$xlevels, function(xx) xx[1])
  refDF <- data.frame(t(c(1, rep(0, length(numericVarNames)), referenceClass)))
  names(refDF) <- c('exposure', numericVarNames, categoricalVarNames)

  ggplotObject <- list()
  coVarMatrix <- vcov(gamFit)
  N <- 10000

  for(iVar in 1:length(selectedVar)){

    newDataVar <- seq(min(inputDT[[which(names(inputDT) == selectedVar)]], na.rm = TRUE), max(inputDT[[which(names(inputDT) == selectedVar)]], na.rm = TRUE), length = nSamples)
    newData <- data.frame(refDF, temp = rep(-1, length(newDataVar)))
    newData <- newData[,-ncol(newData)]
    newData[, which(names(refDF) == selectedVar)] <- newDataVar
    newData <- as.data.table(newData)

    categoricalCols <- which(names(newData) %in% names(gamFit$xlevels))

    numericCols <- 1:ncol(newData)
    numericCols <- numericCols[-categoricalCols]
    names(newData)[numericCols]

    for(iCol in numericCols){
      newData[[iCol]] <- as.numeric(newData[[iCol]])
    }

    if(simultaneousCI == TRUE){

      predictions <- predict(gamFit, newData, se.fit = TRUE)
      fitSE <- predictions$se.fit

      normalSamples <- rMVN(N, mu = rep(0, nrow(coVarMatrix)), sig = coVarMatrix)
      Cg <- predict(gamFit, newData, type = "lpmatrix")
      simulatedDiff <- Cg %*% t(normalSamples)
      absoluteDiff <- abs(sweep(simulatedDiff, 1, fitSE, FUN = "/"))
      maxDiff <- apply(absoluteDiff, 2L, max)
      criticalValue <- quantile(maxDiff, prob = 0.95, type = 8)

      upperBound <- splineEst + criticalValue*stError
      lowerBound <- splineEst - criticalValue*stError

    } else {

      predictions <- predict(gamFit, newData, type = "terms", se.fit = TRUE)
      fitSE <- predictions$se.fit

      selectedVarSpline <- paste(paste('s(', selectedVar, sep = ''), ')', sep = '')
      selectedCol <- grep(selectedVarSpline, colnames(fitSE), fixed = TRUE)
      stError <- fitSE[, selectedCol]

      Xp <- predict(gamFit, newdata = newData, type = "lpmatrix")

      selectedCol <- grep(selectedVarSpline, colnames(Xp), fixed = TRUE)
      splineEst <- Xp[, selectedCol]%*%gamFit$coefficients[selectedCol]

      upperBound <- splineEst + 2*stError
      lowerBound <- splineEst - 2*stError

    }
    ggplotData <- as.data.table(data.frame(xVar = newDataVar, yVar = splineEst, upperBound = upperBound, lowerBound = lowerBound))

    if(is.null(xLimits) & is.null(yLimits)){
      ggplotObject[[iVar]] <- ggplot(ggplotData, aes(x = xVar, y = yVar)) + geom_line(lwd = lineWidth, color = colorLine) +
        geom_ribbon(aes(ymin = lowerBound, ymax = upperBound), alpha = transparence, fill = colorRibbon) +
        xlab(selectedVar[iVar]) + ylab('Predictions + uncertainty')
    } else if(is.null(xLimits) & !is.null(yLimits)){
      ggplotObject[[iVar]] <- ggplot(ggplotData, aes(x = xVar, y = yVar)) + geom_line(lwd = lineWidth, color = colorLine) +
        geom_ribbon(aes(ymin = lowerBound, ymax = upperBound), alpha = transparence, fill = colorRibbon) +
        xlab(selectedVar[iVar]) + ylab('Predictions + uncertainty') + coord_cartesian(ylim = yLimits)
    } else if(!is.null(xLimits) & is.null(yLimits)){
      ggplotObject[[iVar]] <- ggplot(ggplotData, aes(x = xVar, y = yVar)) + geom_line(lwd = lineWidth, color = colorLine) +
        geom_ribbon(aes(ymin = lowerBound, ymax = upperBound), alpha = transparence, fill = colorRibbon) +
        xlab(selectedVar[iVar]) + ylab('Predictions + uncertainty') + coord_cartesian(xlim = xLimits)
    } else if(!is.null(xLimits) & !is.null(yLimits)){
      ggplotObject[[iVar]] <- ggplot(ggplotData, aes(x = xVar, y = yVar)) + geom_line(lwd = lineWidth, color = colorLine) +
        geom_ribbon(aes(ymin = lowerBound, ymax = upperBound), alpha = transparence, fill = colorRibbon) +
        xlab(selectedVar[iVar]) + ylab('Predictions + uncertainty') + coord_cartesian(xlim = xLimits) + coord_cartesian(ylim = yLimits)
    }
    if(showPlot == TRUE) print(ggplotObject[[iVar]])
    if(!is.null(plotDir) & !is.null(plotName)){
      if(pdf2) ggsave(paste(plotDir, paste(plotName, '.pdf', sep = ''), sep = ""))
      ggsave(paste(plotDir, paste(plotName, '.png', sep = ''), sep = ""))
    }
  }
  return(ggplotObject)
}

changeLevelNamesDT <- function(inputDT, categoricalVar, oldLevelName, newLevelName, dropCheck = FALSE){

  checkCharVec(list(categoricalVar, oldLevelName, newLevelName))
  checkDT(inputDT, categoricalVar)
  checkLength(list(categoricalVar), 1)
  checkEqualLength(list(oldLevelName, newLevelName))

  indexFactor <- which(names(inputDT) %in% categoricalVar)
  noFactor <- !isFactorDT(inputDT[,.SD,.SDcols = indexFactor], NULL, FALSE)
  if(sum(noFactor) != 0) stop(paste(paste('The variable ', categoricalVar[noFactor], sep = ''), ' of the "inputDT" argument is not a factor.', sep = ''))

  selectedFactor <- inputDT[,.SD,.SDcols = indexFactor]
  if(!dropCheck){
    for(iRun in 1:length(oldLevelName)){
      if(!isPresentCharVec(levels(selectedFactor[[1]]), oldLevelName[iRun])) stop(sprintf("The value %s of the %dth element of the 'oldLevelName' argument does not correspond to a valid level of the 'categoricalVar' argument, %s.", oldLevelName[iRun], iRun, categoricalVar))
    }
  }
  levels(selectedFactor[[1]]) <- c(levels(selectedFactor[[1]]), newLevelName)
  for(iRun in 1:length(oldLevelName)){
    set(selectedFactor, as.integer(which(selectedFactor[[1]] == oldLevelName[iRun])), 1L, newLevelName[iRun])
  }
  set(inputDT, NULL, as.integer(indexFactor), selectedFactor)
  removeEmptyLevelsDT(inputDT, categoricalVar)

}

newArgCheck <- function() {
  argcheck <- new.env()
  assign("n_warn", 0, envir = argcheck)
  assign("warn_msg", NULL, envir = argcheck)
  assign("n_error", 0, envir = argcheck)
  assign("error_msg", NULL, envir = argcheck)
  assign("n_message", 0, envir = argcheck)
  assign("message_msg", NULL, envir = argcheck)
  class(argcheck) <- c("ArgCheck", "environment")
  return(argcheck)
}

isPresentCharVec <- function(inputVector, value2BeChecked){
  if(!is.character(inputVector) & !is.factor(inputVector)){
    stop("The 'inputVector' argument should be a factor or a character vector.")
  }
  if(!is.character(value2BeChecked) & !is.factor(value2BeChecked)){
    stop("The 'value2BeChecked' argument should be a factor or a character vector.")
  }
  if(length(which(inputVector == value2BeChecked)) > 0){
    return(TRUE)
  } else {
    return(FALSE)
  }
}

finishArgCheck <- function(argcheck, justWarnings = FALSE){

  fn_call <- sys.call(-1)
  fn_call <- utils::capture.output(fn_call)
  if(!"ArgCheck" %in% class(argcheck)) stop("'argcheck' must be an object of class 'ArgCheck'")
  argcheck <- mget(ls(envir = argcheck), envir = argcheck)
  if(argcheck$n_warn > 0)
    warning(paste0(c("", fn_call, paste0(1:argcheck$n_warn, ": ", argcheck$warn_msg)), collapse = "\n"), call. = FALSE)
  if(argcheck$n_message > 0)
    message(paste0(c("", fn_call, paste0(1:argcheck$n_message, ": ", argcheck$message_msg)), collapse = "\n"))
  if(argcheck$n_error > 0){
    if(justWarnings == FALSE){
      stop(paste0(c("", fn_call, paste0(1:argcheck$n_error, ": ", argcheck$error_msg)), collapse = "\n"), call. = FALSE)
    } else {
      warning(paste0(c("", fn_call, paste0(1:argcheck$n_error, ": ", argcheck$error_msg)), collapse = "\n"), call. = FALSE)
    }
  }
}

checkEqualLength <- function(listObjects, ...){

  namesListObjects <- deparse(substitute(listObjects))
  if(str_count(namesListObjects, "list[(]") > 1){
    stop('All the objects to be checked need to be put in a list. A list of lists is not permitted by this function.')
  }
  if(nargs() > 1){
    stop("Just one separate argument needs to be provided for.")
  }
  if(length(listObjects) == 1){
    stop("The names of at least 2 different objects should be stored in the 'listObjects' argument.")
  }

  namesListObjects <- unlist(strsplit(namesListObjects, '[,]'))
  namesListObjects <- gsub("\\s", "", namesListObjects)
  namesListObjects <- str_replace_all(namesListObjects, fixed("list("), "")
  namesListObjects <- str_replace_all(namesListObjects, fixed(")"), "")
  isObject <- aaply(namesListObjects, 1, function(xx) exists(xx, envir = parent.frame()))

  checkList <- newArgCheck()

  lengthObjects <- laply(listObjects, length)
  nElements <- length(lengthObjects)
  comparison <- rep(NA, (nElements*(nElements - 1))/2)
  counter <- 1
  for(iLeft in 1:(nElements - 1)){
    for(iRight in (iLeft + 1):nElements){
      if(!identical(lengthObjects[iLeft], lengthObjects[iRight])){
        if(isObject[iLeft] & isObject[iRight]){
          addError(msg = paste(paste(paste(paste("The length of argument '", namesListObjects[iLeft], sep = ''), "' differs in length from the length of argument '", sep = ''), namesListObjects[iRight], sep = ''), "'.", sep = ''), argcheck = checkList)
        } else if(isObject[iLeft] & !isObject[iRight]){
          addError(msg = paste(paste(paste(paste("The length of argument '", namesListObjects[iLeft], sep = ''), "' differs in length from the length of the element with index ", sep = ''), iRight, sep = ''), '.', sep = ''), argcheck = checkList)
        } else if(!isObject[iLeft] & isObject[iRight]){
          addError(msg = paste(paste(paste(paste("The length of the element with index ", iLeft, sep = ''), " differs in length from the length of argument '", sep = ''), namesListObjects[iRight], sep = ''), "'.", sep = ''), argcheck = checkList)
        } else{
          addError(msg = paste(paste(paste(paste("The length of the element with index ", iLeft, sep = ''), " differs in length from the length of the element with index ", sep = ''), iRight, sep = ''), ".", sep = ''), argcheck = checkList)
        }
      }
    }
  }
  finishArgCheck(checkList)
}

addLevelDT <- function(inputDT, catVar, newLevel){

  checkCharVec(list(catVar, newLevel))
  checkLength(list(catVar, newLevel), 1)
  checkDT(inputDT, catVar)

  indCatVar <- which(names(inputDT) == catVar)
  selFact <- inputDT[, .SD, .SDcols = indCatVar]
  levels(selFact[[1]]) <- unique(c(levels(selFact[[1]]), newLevel))
  set(inputDT, NULL, as.integer(indCatVar), selFact)

}

extractSplineEstimate <- function(gamFit, keyWordColSelection = NULL){

  allSmoothTerms <- laply(gamFit$smooth, function(xx) xx$vn[1])
  allSmoothTerms[which(!allSmoothTerms %in% c('longitude', 'latitude'))]
  if(is.null(keyWordColSelection)){
    keyWordColSelection <- allSmoothTerms
  } else {
    present <- (keyWordColSelection %in% allSmoothTerms)
    if(length(present) != length(keyWordColSelection)){
      sprintf("The elements %s of the 'keyWordColSelection' argument are not part of the smooth terms of the 'gamFit' argument.", paste(keyWordColSelection[!present], collapse = ', '))
    }
  }
  if(sum(class(gamFit) %in% 'gam') == 0) stop('The "gamFit" argument should be an output object of the mgcv::gam function.')
  gamFitModelMatrix <- model.matrix(gamFit)
  fittedSpline <- list()
  length(fittedSpline) <- length(keyWordColSelection)
  for(iSpline in 1:length(fittedSpline)){
    selectionCols <- grep(keyWordColSelection[iSpline], names(gamFit$coefficients))
    fittedSpline[[iSpline]] <- gamFitModelMatrix[, selectionCols]%*%gamFit$coefficients[selectionCols]
  }
  names(fittedSpline) <- keyWordColSelection
  return(fittedSpline)
}

#' Checking if all elements of a list are all numeric or integer vectors with just whole numbers
#'
#' @param listNum A list of the vectors of which one wishes to check if their data type is numeric
#' @param namesListElements Character vector containing the names of the variables of which the data type is checked. Optional parameter, with as default value NULL. This argument should be used when the variable of which the data type is checked is not an object that was provided as an argument to the function, or when the list elements of the first argument do not have a name attached to it.
#' @return No value is returned if all vectors have the numeric or integer data type, containing just whole numbers. If not, an error message is thrown for each element of the list that does not pertain to the numeric data type.
#' @examples
#' arg1 <- 2
#' checkWholeNumb(list(arg1))
#' checkWholeNumb(list(2))
#'
#' arg2 <- 0.8
#' \donttest{checkWholeNumb(list(arg2))}
#' \donttest{checkWholeNumb(list(1, arg2))}

checkWholeNumb <- function(listNum, namesListElements = NULL){

  if(!is.list(listNum)) stop("The argument 'listNum' needs to be a list.")

  for(iList in 1:length(listNum)){
    if(is.list(listNum[[iList]])){
      stop(paste(paste('The data type of element with index ', iList, sep = ''), ' is NOT a vector. A list of lists is not permitted by this function.', sep = ''))
    }
  }

  checkList <- newArgCheck()
  argNames <- deparse(substitute(listNum))
  argNames <- unlist(strsplit(argNames, '[,]'))
  argNames <- gsub("\\s", "", argNames)
  argNames <- str_replace_all(argNames, fixed("list("), "")
  argNames <- str_replace_all(argNames, fixed(")"), "")
  isObject <- aaply(argNames, 1, function(xx) exists(xx, envir = parent.frame()))

  if(!is.null(namesListElements)){
    if(!is.character(namesListElements)) stop("The argument 'namesListElements' should be a character vector or string.")
    if(length(listNum) != length(namesListElements)) stop("The argument 'listNum' should have the same length as argument 'namesListElements'.")
  }

  if(is.list(listNum)){
    for(iList in 1:length(listNum)){
      checkNumOrIntVec(list(listNum[[iList]]))
      if(!all(listNum[[iList]] == round(listNum[[iList]])) & !is.null(listNum[[iList]])){
        if(isObject[iList]){
          addError(msg = paste(paste("The argument '", argNames[iList], sep = ''), "' does NOT consist of whole numbers.", sep = ''), argcheck = checkList)
        } else {
          if(is.null(namesListElements)){
            addError(msg = paste(paste('The element with index ', iList, sep = ''), ' does NOT consist of whole numbers.', sep = ''), argcheck = checkList)
          } else {
            addError(msg = paste(paste("The argument '", namesListElements[iList], sep = ''), "' does NOT consist of whole numbers.", sep = ''), argcheck = checkList)
          }
        }
      }
    }
  } else if(is.vector(listNum) & length(listNum) == 1){
    checkNumOrIntVec(list(listNum))
    if(!all(listNum == round(listNum)) & !is.null(listNum)){
      if(isObject){
        addError(msg = paste(paste("The argument '", argNames, sep = ''), "' does NOT consist of whole numbers.", sep = ''), argcheck = checkList)
      } else {
        if(is.null(namesListElements)){
          addError(msg = paste(paste('The element with index ', '1', sep = ''), ' does NOT consist of whole numbers.', sep = ''), argcheck = checkList)
        } else {
          addError(msg = paste(paste("The argument '", namesListElements, sep = ''), "' does NOT consist of whole numbers.", sep = ''), argcheck = checkList)
        }
      }
    }
  } else {
    addError(msg = 'The "listNum" argument should be of a list.', argcheck = checkList)
  }
  finishArgCheck(checkList)
}

addError <- function(msg, argcheck){
  if(!"ArgCheck" %in% class(argcheck)) stop("'argcheck' must be an object of class 'ArgCheck'")
  assign("n_error", get("n_error", envir = argcheck) + 1, envir = argcheck)
  assign("error_msg", c(get("error_msg", envir = argcheck), msg), envir = argcheck)
}

optNumbGroups <- function(inputDT, gamFit, splineEst, splineVarNames, nGroups = seq(4, 10, 2), plotDir = NULL, sampleSizeBin = min(50000, nrow(inputDT)), minIncr = 0.05, minIncrRel = 1.05, showExplVar = FALSE, minExplVar = 0.6, addVar = FALSE, verbose = TRUE, nSamples = 10, maxIter = 500, xLimit = NULL){

  checkDT(inputDT, splineVarNames[splineVarNames != 'zone'])
  if(sum(class(gamFit) %in% c('gam', 'bam')) == 0) stop('The "gamFit" argument should be an output object of the mgcv::gam function.')
  allSplineVar <- unlist(llply(gamFit$smooth, function(xx) xx$term))
  for(elementSplineEst in names(splineEst)) checkValues(list(elementSplineEst), list(allSplineVar))

  checkCharVec(list(splineVarNames))
  if(length(splineVarNames[splineVarNames != 'zone'])){
    splineVarNamesRed <- splineVarNames[splineVarNames != 'zone']
    for(iCheck in 1:length(splineVarNamesRed)) checkValues(list(splineVarNamesRed[iCheck]), list(names(splineEst)))
  }

  checkNumOrIntVec(list(nGroups, sampleSizeBin, nSamples, maxIter))
  for(iGroup in nGroups) checkRanges(list(iGroup), list(c('>', 0)))
  checkWholeNumb(list(nGroups, sampleSizeBin, nSamples, maxIter))

  checkLength(list(showExplVar, minIncr, minIncrRel, addVar, sampleSizeBin), 1)
  checkNumVec(list(minIncr, minIncrRel))
  checkRanges(list(minIncr), list(c('>=', 0, '<=', 1)))

  checkRanges(list(minIncrRel), list(c('>=', 1)))
  checkLogicVec(list(showExplVar, addVar))
  checkEqualLength(list(splineVarNames, xLimit))

  splittingPoints <- list()
  explVarList <- list()

  length(splittingPoints) <- length(splineVarNames)
  names(splittingPoints) <- splineVarNames
  length(explVarList) <- length(splineVarNames)
  names(explVarList) <- splineVarNames

  for(iSpline in 1:length(splineVarNames)){

    if(verbose) print(paste(paste(iSpline, '.: ', sep = ''), splineVarNames[iSpline], sep = ''))
    splittingPointsTemp <- list()
    iRun <- 1
    explVar <- rep(NA, length(nGroups))
    for(iGroup in nGroups){
      if(verbose) print(paste('  Group ', iGroup, sep = ''))
      splits <- list()
      tempDT <- copy(inputDT)
      if(splineVarNames[iSpline] %in% c('longitude', 'latitude', 'zone')){
        splittingPointsTemp[[iRun]] <- groupingSpatialSpline(splineEst = splineEst[which(names(splineEst) %in% c('longitude', 'latitude'))][[1]], nGroups = iGroup, sampleSizeBin = min(nrow(tempDT), sampleSizeBin), nSamples = nSamples, verbose = FALSE, asList = FALSE) #classicTP
        splits[[1]] <- splittingPointsTemp[[iRun]]
        inputDT[, zone := splineEst[which(names(splineEst) %in% c('longitude', 'latitude'))][[1]]]
        tempDT[, zone := splineEst[which(names(splineEst) %in% c('longitude', 'latitude'))][[1]]]
        names(splits) <- 'zone'
        groupedVarName <- 'zoneGrouped'
        names(splittingPoints)[iSpline] <- 'zone'
      } else {
        splittingPointsTemp[[iRun]] <- groupingVars(splineVarNames[iSpline], iGroup, gamFit, tempDT, splineEst, sampleSizeBin = min(nrow(tempDT), sampleSizeBin), nSamples = nSamples, maxIter = maxIter, showPlot = FALSE, xLimit = xLimit[[iSpline]])
        if(is.list(splittingPointsTemp[[iRun]])){
          splits[[1]] <- splittingPointsTemp[[iRun]]$splits
        } else {
          splits[[1]] <- NA
        }
        names(splits) <- splineVarNames[iSpline]
        groupedVarName <- paste(splineVarNames[iSpline], 'Grouped', sep = '')
      }

      if(sum(is.na(splits[[1]])) == 0){

        transform2BinnedVar(tempDT, splits)
        levs <- extractLevelDT(tempDT, groupedVarName)[[1]]
        sdGroup <- rep(NA, length(levs))
        nObs <- rep(NA, length(levs))
        for(iLev in 1:length(levs)){
          workDT <- copy(tempDT)
          #workDT <- extractData4ModelFit(workDT, splineVarNames[iSpline], NULL, NULL)
          workDT <- workDT[as.integer(rownames(splineEst[[which(names(splineEst) == splineVarNames[iSpline])]])), ]
          if(splineVarNames[iSpline] == 'zone'){
            workDT[, splineEst := splineEst[which(names(splineEst) %in% c('longitude', 'latitude'))][[1]]]
          } else {
            workDT[, splineEst := splineEst[which(names(splineEst) == splineVarNames[iSpline])]]
          }
          splineEstSubset <- subsetDT(workDT, which(names(workDT) == groupedVarName), levs[iLev])$splineEst
          sdGroup[iLev] <- sd(splineEstSubset, na.rm = T)
          nObs[iLev] <- length(splineEstSubset)
        }
        if(splineVarNames[iSpline] == 'zone'){
          explVar[iRun] <- 1 - (sum(sdGroup*nObs)/sum(nObs))/sd(splineEst[which(names(splineEst) %in% c('longitude', 'latitude'))][[1]])
        } else {
          explVar[iRun] <- 1 - (sum(sdGroup*nObs)/sum(nObs))/sd(splineEst[which(names(splineEst) == splineVarNames[iSpline])][[1]])
        }
      }
      iRun <- iRun + 1
    }

    if(sum(is.na(explVar)) == length(explVar)){
      splittingPoints[[iSpline]] <- NA
      explVarList[[iSpline]] <- NA
      warning(paste("No valid splits for any of the tested groups (cfr. 'nGroups' argument) were found for the variable ", splineVarNames[iSpline], sep = ''))
    } else {

      selInds <- !is.na(explVar)
      explVarSub <- explVar[selInds]
      nGroupsSub <- nGroups[selInds]

      if(sum(selInds) > 1){
        selGroup <- which(explVarSub[2:length(nGroupsSub)] - explVarSub[1:(length(nGroupsSub) - 1)] < minIncr)
        selGroupRel <- which(explVarSub[2:length(nGroupsSub)] < minIncrRel* explVarSub[1:(length(nGroupsSub) - 1)])

        if(length(selGroup) > 1) selGroup <- max(selGroup)
        if(length(selGroupRel) > 1) selGroupRel <- max(selGroupRel)
        if(length(selGroup) == 0) selGroup <- length(nGroupsSub)
        if(length(selGroupRel) == 0) selGroupRel <- length(nGroupsSub)
        splittingPoints[[iSpline]] <- splittingPointsTemp[[which(selInds)[min(selGroup, selGroupRel)]]]
        explVarList[[iSpline]] <- explVarSub[min(selGroup, selGroupRel)]
        if(explVarSub[min(selGroup, selGroupRel)] < minExplVar) warning(paste(paste(paste(paste('Even the optimal grouping for the variable ', splineVarNames[iSpline], sep = ''), ' did not exceed the minimal value for the explained variance', sep = ''), "as set by the 'minExplVar' argument. Please increase the number of possible groups, as indicated by the 'nGroups' argument.")))
      } else {
        splittingPoints[[iSpline]] <- splittingPointsTemp[[which(selInds)]]
        explVarList[[iSpline]] <- explVarSub[selInds]
        if(explVar[selInds] < minExplVar) warning(paste(paste(paste(paste('Even the optimal grouping for the variable ', splineVarNames[iSpline], sep = ''), ' did not exceed the minimal value for the explained variance', sep = ''), "as set by the 'minExplVar' argument. Please increase the number of possible groups, as indicated by the 'nGroups' argument.")))
      }
    }
    if(!is.null(plotDir)) groupingVars(splineVarNames[iSpline], length(unlist(splittingPoints[[iSpline]])) + 1, gamFit, inputDT, splineEst, plotDir = plotDir, plotName = paste(splineVarNames[iSpline], 'Grouped', sep = ''), sampleSizeBin = min(nrow(inputDT), sampleSizeBin), nSamples = nSamples, maxIter = maxIter, showPlot = FALSE, xLimit = xLimit[[iSpline]])
  }

  if(addVar){
    transform2BinnedVar(inputDT, splittingPoints)
  }

  if(showExplVar){
    return(list(splits = splittingPoints, explVar = explVarList))
  } else {
    return(list(splits = splittingPoints))
  }
}

getMode <- function(vector){
  uniqueValues <- unique(vector)
  uniqueValues[which.max(tabulate(match(vector, uniqueValues)))]
}


#' Selecting a subset of a single column, selected by its numeric index, of a data.table object, optionally subsetted by some value
#'
#' @param inputDT data.table object containing the data of interest. This is an obligatory argument, without default value.
#' @param rowValue Vector of length 1 that contains the row value of the selected column that should be selected. This is an optional argument, with NULL as default value (in which case all rows are selected).
#' @param colNbr Integer or numeric vector of length 1 that contains the column number of the 'inputDT' object that is selected. This is an obligatory argument, without default value.
#' @param missingsColNbrGone Logical vector of length 1 indicating whether or not one wants to remove the missing values of the selected column. The default value is TRUE and this is an optimal argument only.
#' @return The subset of the 'inputDT' data.table object. If a non-existant value is chosen for the selected column, an empty data.table object is returned.
#' @examples
#' library(data.table)
#' inputDT <- as.data.table(data.frame(x = 10:1, y = LETTERS[1:10]))
#' subsetDT(inputDT, 1, 10)
#' subsetDT(inputDT, 2, 'A')
#' subsetDT(inputDT, 1, 'A')

subsetDT <- function(inputDT, colNbr, rowValue = NULL, missingsColNbrGone = TRUE){
  checkDT(inputDT)
  if(!is.null(rowValue)){
    checkLength(list(rowValue, colNbr), c(1, 1))
  }
  checkLength(list(missingsColNbrGone), 1)
  checkLogicVec(list(missingsColNbrGone))
  if(is.factor(rowValue)) rowValue <- as.character(rowValue)
  if((isCharacterDT(inputDT)[colNbr] | isFactorDT(inputDT)[colNbr]) & !is.character(rowValue)){
    stop("If the 'rowValue' argument is a character or factor value, then the selected column needs be a character or factor variable. This requirement is violated here, so please fix it.")
  } else if(isNumericDT(inputDT)[colNbr] & !is.numeric(rowValue)){
    stop("If the 'rowValue' argument is a numeric value, then the selected column needs be a numeric variable. This requirement is violated here, so please fix it.")
  } else if(isIntegerDT(inputDT)[colNbr] & !is.integer(rowValue)){
    stop("If the 'rowValue' argument is an integer value, then the selected column needs be an integer variable. This requirement is violated here, so please fix it.")
  } else if(isLogicalDT(inputDT)[colNbr] & !is.logical(rowValue)){
    stop("If the 'rowValue' argument is a logical value, then the selected column needs be a logical variable. This requirement is violated here, so please fix it.")
  }
  checkNumOrIntVec(list(colNbr))
  checkRanges(list(colNbr), list(c('>=', 0, '<=', ncol(inputDT))))
  if(is.null(rowValue)){
    if(missingsColNbrGone){
      inputDT <- inputDT[inputDT[, .SD, .SDcols = colNbr]]
      return(inputDT[complete.cases(inputDT[,.SD, .SDcols = colNbr])])
    } else {
      return(inputDT[inputDT[, .SD, .SDcols = colNbr]])
    }
  } else {
    if(missingsColNbrGone){
      inputDT <- inputDT[inputDT[, .I[.SD == rowValue], .SDcols = colNbr]]
      return(inputDT[complete.cases(inputDT[,.SD, .SDcols = colNbr])])
    } else {
      return(inputDT[inputDT[, .I[.SD == (as.character(rowValue))], .SDcols = colNbr]])
    }
  }
}

#' Constructing a ggplot-styled map of the Belgium, with colors for the different zones that are selected randomly
#'
#' @param plottingSpatialEffectCyclingOutputObject Output object of the 'plottingSpatialEffect' function (generated by toggling the 'randomPick' argument to TRUE). This is an obligatory argument, without default value.
#' @param predFactColors Character vector of length equal to the number of different values of the 'predictions' argument. Only if the 'predAsFact' argument is toggled to TRUE, this argument is taken into account. This is an optional argument, with NULL as default value.
#' @param printColourNames Logical vector of length 1 indicating whether or not one wants to print the name of the randomly selected color names. As such, one can learn the actual names of nice-looking colors. This is an optional argument, with FALSE as default value.
#' @param noSaving Logical vector of length 1 indicating whether or not one wants to save the produced ggplot object. The arguments 'plotTitle', 'plotDir', 'plotName' and 'pdf2', necessary to save the plot, are automatically supplied by the 'plottingSpatialEffectCyclingOutputObject' argument, such that any changes regarding these arguments need to be done by the 'plottingSpatialEffect' function. This is an optional argument, with FALSE as default value.
#' @return A ggplot object is constructed with a different color for each zone (the predictions need to be a factor), each color being randomly selected.
#' @examples
#' library(data.table)
#' dirShape <<- '/home/robin/shape/'
#' postalCode <- read.csv(paste(defaultdirectoryShapeFile, "postins.csv", sep = "\\"))[,c("CODPOSS","INS")]
#' inputDT <- data.table(postalCode = postalCode$CODPOSS, exposure = runif(nrow(postalCode)))
#' predictions <- as.numeric(as.character(postalCode$CODPOSS))
#' # If doesn't know which colour should be chosen, one can use the
#' # 'randomPick' argument and the 'randomColorPick' function.
#' inputCyclingFunction <- plottingSpatialEffect(inputDT, predictions, predAsFact = TRUE, randomPick = TRUE)
#' randomColorPick(inputCyclingFunction, , TRUE)

randomColorPick <- function(plottingSpatialEffectCyclingOutputObject, predFactColors = NULL, printColourNames = FALSE, noSaving = FALSE){

  checkLogicVec(list(printColourNames, noSaving))
  checkLength(list(printColourNames, noSaving), 1)

  belgiumFortified <- plottingSpatialEffectCyclingOutputObject$belgiumFortified
  plotTitle <- plottingSpatialEffectCyclingOutputObject$plotTitle
  plotDir <- plottingSpatialEffectCyclingOutputObject$plotDir
  plotName <- plottingSpatialEffectCyclingOutputObject$plotName
  pdf2 <- plottingSpatialEffectCyclingOutputObject$pdf2

  if(is.null(predFactColors)){

    colorPalette <- colors()[-grep('white', colors())]
    colorPalette <- colorPalette[-grep('grey', colorPalette)]
    colorPalette <- colorPalette[-grep('gray', colorPalette)]
    colorPalette <- colorPalette[-grep('ivory', colorPalette)]
    colorPalette <- colorPalette[-grep('silk', colorPalette)]
    colorPalette <- colorPalette[-grep('bisque', colorPalette)]
    colorPalette <- colorPalette[-grep('lightyellow', colorPalette)]
    colorPalette <- colorPalette[-grep('honeydew', colorPalette)]
    colorPalette <- colorPalette[-grep('palegolden', colorPalette)]
    colorPalette <- colorPalette[-grep('mint', colorPalette)]
    colorPalette <- colorPalette[-grep('lavender', colorPalette)]
    colorPalette <- colorPalette[-grep('lightcyan', colorPalette)]
    colorPalette <- colorPalette[-grep('lemon', colorPalette)]
    colorPalette <- colorPalette[-grep('linen', colorPalette)]
    colorPalette <- colorPalette[-grep('snow', colorPalette)]
    colorPalette <- colorPalette[-grep('almond', colorPalette)]
    colorPalette <- colorPalette[-grep('wheat', colorPalette)]
    colorPalette <- colorPalette[-grep('tan', colorPalette)]
    colorPalette <- colorPalette[-grep('azure', colorPalette)]
    colorPalette <- colorPalette[-grep('papaya', colorPalette)]
    colorPalette <- colorPalette[-grep('lightgolden', colorPalette)]

    predFactColors <- colorPalette[sample(1:length(colorPalette), length(unique(belgiumFortified$predicted)))]

  } else {
    checkLength(list(predFactColors), length(unique(inputDT$predicted)))
    if(sum(predFactColors %in% colors()) != length(unique(inputDT$predicted))) stop('Some of the values of the "predFactColors" argument are not valid color names. The valid values can be found by running the "colors()" function on the command line.')
  }

  mapPredictions <- ggplot(belgiumFortified, aes(long, lat, group = group, fill = factor(predicted), text = paste(paste(paste('Postal Code: ', postalCode, sep = ''), ' -- ', sep = ''), Name1, sep = ''))) + xlab('Longitude') + ylab('Latitude') + scale_fill_manual(name = "Predictions", values = predFactColors) + geom_polygon() + ggtitle(paste(plotTitle, ': Predictions', sep = ''))

  if(!noSaving){
    if(!is.null(plotDir) & !is.null(plotName)){
      if(pdf2) ggsave(paste(plotDir, paste(plotName, '.pdf', sep = ''), sep = ""))
      ggsave(paste(plotDir, paste(plotName, '.png', sep = ''), sep = ""))
    }
  }
  if(printColourNames == TRUE) print(predFactColors)
  return(mapPredictions)

}
