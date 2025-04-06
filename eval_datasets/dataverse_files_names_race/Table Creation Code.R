rm(list = ls())
library(readr)
library(tidyverse)

# constants
threshold <- 25
summedTableDir <- 'path/to/summed/tables'
dictDir <- 'dictionary/dir'
  
# read in the files
load(paste(summedTableDir, 'final_summed_tables.rData', sep = '/'))
colNames <- c("Voters_FirstName", "Voters_MiddleName", "Voters_LastName")

########################################################
##                  helper functions                  ##
########################################################

# name-race probs
makeNameRaceProbs <- function(tab) {
  
  base <- data.frame(tab[,-1]/matrix(rowSums(tab[,-1]), ncol = 5, nrow = nrow(tab)))
  base$name <- tab$name
  base[,c('name', 'whi', 'bla', 'his', 'asi', 'oth')]
  
}

# race-name probs
makeRaceNameProbs <- function(tab, aggregates) {
  
  base <- data.frame(tab[,-1]/matrix(aggregates, ncol = 5, nrow = nrow(tab), byrow = TRUE))
  base$name <- tab$name
  base[,c('name', 'whi', 'bla', 'his', 'asi', 'oth')]
  
}

###################################################
##              punctuation removal              ##
###################################################

finalSummedTables <- lapply(finalSummedTables, FUN = function(nameTable) {
  
  # remove all the punctuation
  nameTable$name <- gsub("[[:punct:][:blank:]]+", "", nameTable$name)
  
  # re-collapse everything 
  nameTable <- nameTable %>% 
    group_by(name) %>%
    summarize(whi = sum(whi), bla = sum(bla), his = sum(his), 
              asi = sum(asi), oth = sum(oth))
  
  # return the result
  nameTable
})

########################################################
##           table filtering and formatting           ##
########################################################

# filter out all the rows below threshold and collapse them together
aggregateRecords <- lapply(finalSummedTables, FUN = function(x) {colSums(x[,-1])})
finalSummedTables.filtered <- lapply(finalSummedTables, FUN = function(tab) {
  allOther <- colSums(tab[rowSums(tab[,-1]) < threshold,-1])
  tab[rowSums(tab[,-1]) >= threshold,] %>% 
    add_row(name = 'ALL OTHER NAMES', 
            whi = allOther[1], bla = allOther[2], his = allOther[3], asi = allOther[4], oth = allOther[5])
})

# build the name-race tables
first_nameRaceProbs <- makeNameRaceProbs(finalSummedTables.filtered[[1]])
middle_nameRaceProbs <- makeNameRaceProbs(finalSummedTables.filtered[[2]])
last_nameRaceProbs <- makeNameRaceProbs(finalSummedTables.filtered[[3]])

# build the race-name tables
first_raceNameProbs <- makeRaceNameProbs(finalSummedTables.filtered[[1]], aggregateRecords[[1]])
middle_raceNameProbs <- makeRaceNameProbs(finalSummedTables.filtered[[2]], aggregateRecords[[2]])
last_raceNameProbs <- makeRaceNameProbs(finalSummedTables.filtered[[3]], aggregateRecords[[3]])

# save all the dictionaries
save(first_nameRaceProbs, file = paste(dictDir, 'first_nameRaceProbs.rData', sep = '/'))
save(middle_nameRaceProbs, file = paste(dictDir, 'middle_nameRaceProbs.rData', sep = '/'))
save(last_nameRaceProbs, file = paste(dictDir, 'last_nameRaceProbs.rData', sep = '/'))
save(first_raceNameProbs, file = paste(dictDir, 'first_raceNameProbs.rData', sep = '/'))
save(middle_raceNameProbs, file = paste(dictDir, 'middle_raceNameProbs.rData', sep = '/'))
save(last_raceNameProbs, file = paste(dictDir, 'last_raceNameProbs.rData', sep = '/'))


# save all the dictionaries as CSVs
write_csv(first_nameRaceProbs, file = paste(dictDir, 'first_nameRaceProbs.csv', sep = '/'))
write_csv(middle_nameRaceProbs, file = paste(dictDir, 'middle_nameRaceProbs.csv', sep = '/'))
write_csv(last_nameRaceProbs, file = paste(dictDir, 'last_nameRaceProbs.csv', sep = '/'))
write_csv(first_raceNameProbs, file = paste(dictDir, 'first_raceNameProbs.csv', sep = '/'))
write_csv(middle_raceNameProbs, file = paste(dictDir, 'middle_raceNameProbs.csv', sep = '/'))
write_csv(last_raceNameProbs, file = paste(dictDir, 'last_raceNameProbs.csv', sep = '/'))


