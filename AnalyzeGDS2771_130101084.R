# 1. Install packages to read the NCBI's GEO microarray SOFT files in R
# 1.Ref. http://www2.warwick.ac.uk/fac/sci/moac/people/students/peter_cock/r/geo/

# 1.1. Uncomment only once to install stuff

#source("https://bioconductor.org/biocLite.R")
#biocLite("GEOquery")
#biocLite("Affyhgu133aExpr")


# 1.2. Use packages # Comment to save time after first run of the program in an R session

library(Biobase)
library(GEOquery)

# Add other libraries that you might need below this line

library(glmnet)

# 2. Read data and convert to dataframe. Comment to save time after first run of the program in an R session
# 2.1. Once download data from ftp://ftp.ncbi.nlm.nih.gov/geo/datasets/GDS2nnn/GDS2771/soft/GDS2771.soft.gz
# 2.Ref.1. About data: http://www.ncbi.nlm.nih.gov/sites/GDSbrowser?acc=GDS2771
# 2.Ref.2. Study that uses that data http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3694402/pdf/nihms471724.pdf
# 2.Warning. Note that do not use FULL SOFT, only SOFT, as mentioned in the link above. 2.2.R. http://stackoverflow.com/questions/20174284/error-in-gzfilefname-open-rt-invalid-description-argument

gds2771 <- getGEO(filename='B:/Acad/Course Material/Semester 7/EE622/Assignment1/GDS2771.soft.gz') # Make sure path is correct as per your working folder. Could be './GDS2771.soft.gz'
eset2771 <- GDS2eSet(gds2771) # See http://www2.warwick.ac.uk/fac/sci/moac/people/students/peter_cock/r/geo/

# 2.2. View data (optional; can be commented). See http://www2.warwick.ac.uk/fac/sci/moac/people/students/peter_cock/r/geo/
# eset2771 # View some meta data
featureNames(eset2771)[1:10] # View first feature names
sampleNames(eset2771) # View patient IDs. Should be 192
pData(eset2771)$disease.state #View disease state of each patient. Should be 192

# 2.3. Convert to data frame by concatenating disease.state with data, using first row as column names, and deleting first row
data2771 <- cbind2(c('disease.state',pData(eset2771)$disease.state),t(Table(gds2771)[,2:194]))
colnames(data2771) = data2771[1, ] # the first row will be the header
data2771 = data2771[-1, ] 

# 2.4. View data frame (optional; can be commented)
# View(data2771)

# WRITE YOUR CODE BELOW THIS LINE

df <- as.data.frame(data2771)
df <- subset(df, df[,1] != "3") # Remove suspected cancer cases from data - 5 in number
df[seq(2, length(colnames(df)))] <- sapply(df[seq(2, length(colnames(df)))], as.numeric) # Convert columns to numeric

#Handling missing data
# Step 1: Eliminate all genes which contain missing values for all samples
df <- df[, colSums(is.na(df)) < nrow(df)]
# Step 2: Impute values for all genes which contain partial missing values (using mean)
# Note: This dataset doesn't require this step, as after step 1, all na values disappear. Method kept for completeness
df <- if (sum(is.na(df)) > 0) mice(df, method = "mean") else df

