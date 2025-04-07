library(readr)
library(tidyverse)

# constants
dir <- 'path/to/voter/files'
outdir <- 'path/to/summed/tables'
states <- c('AL', 'FL', 'GA', 'LA', 'NC', 'SC')
colNames <- c("Voters_FirstName", "Voters_MiddleName", "Voters_LastName")

############################################################
##           build and sum the tables over time           ##
############################################################

stateTables <- lapply(states, FUN = function(state) {
  
  print(state)
  
  # get the list of voter files
  fileList <- list.files(paste(dir, state, sep = ''))
  
  # iterate through the files and read them in 
  tableList <- lapply(fileList, FUN = function(file) {
    fileData <- read_csv(paste(dir, state, '/', file, sep = ''))
    
    # get all the first, middle, and last name tables
    lapply(colNames, FUN = function(colName) {
      
      # cast the names appropriately
      fileData[[colName]][is.na(fileData[[colName]])] <- '' # deal with null names
      fileData[[colName]] <- toupper(iconv(fileData[[colName]]))
      
      # do some formatting to make things easier
      t <- as.data.frame.matrix(table(fileData[[colName]], fileData$CountyEthnic_Description))
      t$name <- rownames(t); rownames(t) <- c(); 
      t <- t[, c(ncol(t), 1:(ncol(t) - 1))]
    })
  })
  
  # sum the tables across states
  summedTables <- lapply(1:length(colNames), FUN = function(i) {
    base <- tableList[[1]][[i]]
    raceGroups <- colnames(base)[-1]
    
    # iterate through later tables and sum
    for(j in 2:length(tableList)) {
      
      # merge the frames
      merged <- full_join(base, tableList[[j]][[i]], by = 'name')
      merged[is.na(merged)] <- 0
      
      # sum up the new results
      for(r in raceGroups) {
        merged[[r]] <- rowSums(merged[,grep(r, colnames(merged))])
      }
      base <- merged[,c("name", raceGroups)]
    }
    
    # formatting
    if('Korean' %in% names(base)) {
      data.frame(name = base$name,
                 whi = base$`White Self Reported`,
                 bla = base$`African or Af-Am Self Reported`,
                 his = base$`Hispanic`, 
                 asi = base$`East Asian` + base$Korean,
                 oth = base$`Other Undefined Race` + base$`Native American (self reported)`)
    } else {
      data.frame(name = base$name,
                 whi = base$`White Self Reported`,
                 bla = base$`African or Af-Am Self Reported`,
                 his = base$`Hispanic`, 
                 asi = base$`East Asian`,
                 oth = base$`Other Undefined Race` + base$`Native American (self reported)`)
    }
    
  })
 
})

##############################################################
##           build and sum the tables over states           ##
##############################################################

# sum the tables across states
finalSummedTables <- lapply(1:length(colNames), FUN = function(i) {
  
  base <- stateTables[[1]][[i]]
  raceGroups <- colnames(base)[-1]

  # iterate through later tables and sum
  for(j in 2:length(stateTables)) {
    
    # merge the frames
    merged <- full_join(base, stateTables[[j]][[i]], by = 'name')
    merged[is.na(merged)] <- 0
    
    # sum up the new results
    for(r in raceGroups) {
      merged[[r]] <- rowSums(merged[,grep(r, colnames(merged))])
    }
    
    base <- merged[,c("name", raceGroups)]
  }
  
  return(base)
})

save(finalSummedTables, file = paste(outdir, 'final_summed_tables.rData', sep = '/'))
