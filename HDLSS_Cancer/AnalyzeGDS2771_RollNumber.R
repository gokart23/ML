# 1. Install packages to read the NCBI's GEO microarray SOFT files in R
# 1.1. Uncomment only once to install stuff
#source("https://bioconductor.org/biocLite.R")
#biocLite("GEOquery")
#biocLite("Affyhgu133aExpr")

# Libraries necessary for running the program

library(Biobase)
library(GEOquery)

# 2. Read data and convert to dataframe.

# Note: Useful info about reading this data can be found here: https://www.bioconductor.org/packages/devel/bioc/vignettes/GEOquery/inst/doc/GEOquery.html
gse4115 <- getGEO(filename="B:/Acad/Course Material/Semester 7/EE622/Assignment1/GSE4115_family.soft.gz", GSEMatrix = TRUE)
gsmList <- GSMList(gse4115)
probesets <- Table(GPLList(gse4115)[[1]])$ID
data.matrix <- do.call('cbind',lapply(gsmList,function(x) {tab <- Table(x)
                          mymatch <- match(probesets,tab$ID_REF)
                          return(tab$VALUE[mymatch])})
                      ) # Make the data matrix
data.matrix <- apply(data.matrix,2,function(x) {as.numeric(as.character(x))}) #Convert all character strings to numbers
data.matrix <- log2(data.matrix) # Scale the numeric values
df <- as.data.frame(data.matrix) #Make a data frame out of it

# 2.1 Extract cancer annotations
cancer.status <- as.data.frame( Map(function(x) {return ((Meta(x)$characteristics_ch1)[6])}, gsmList) )
cancer.status <- t(cancer.status)

# 2.2 Display GSM Description (optional)
# length(gsmList)
# Columns(gsmList[[1]])

# 2.3 Removing suspected cancer samples
suspected <- is.na(cancer.status)
df <- df[!suspected]
cancer.status <- cancer.status[!suspected]
# Convert cancer labels to integers
cancer.status <- as.numeric(factor(cancer.status, levels=unique(cancer.status)))

# Go through the necessary steps to make a compliant ExpressionSet
# rownames(data.matrix) <- probesets
# colnames(data.matrix) <- names(gsmList)
# pdata <- data.frame(samples=names(gsmList))
# rownames(pdata) <- names(gsmList)
# pheno <- as(pdata,"AnnotatedDataFrame")
# eset <- new('ExpressionSet',exprs=data.matrix,phenoData=pheno)

# 2.4. View data frame (optional; can be commented)
# View(data2771)

# WRITE YOUR CODE BELOW THIS LINE

