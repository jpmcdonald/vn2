#################
#WORKING EXAMPLE#
#####################################################################################################################################################################################################################################################################
##A company has 10 sales reps with varying degrees of skill. Each order they close will net the company a $50,000 sale. Over the course of a year, they will make 2000 cold calls. What is the distribtuion of earnings they will obtain collectively for the company?


#Creating some data to work with
set.seed(1)
names <- c("Moe", "Larry", "Curly", "Shep", "Sisko", "Adriana", "Lea", "Pat", "Perry", "Elizabeth")
age <- floor(rnorm(10, 35, 10))
salary <- round(rnorm(10, 60000, 15000),2)
closure_rate <- round(rnorm(10, .06, .015),2)

df <- data.frame(names,closure_rate,salary,age)

#Now, lets generate some SIPs based on our problem. We'll create SIPs for both the amount earned and the number of sales, which in this case will be perfectly correlated. Using the uniform distribution provided in R we can generate our simulation:
SIPdf <- data.frame(matrix(ncol = 10, nrow = 2000))
colnames(SIPdf) <- names
for(i in 1:10) {
  SIPdf[,i] <- runif(2000)
  SIPdf[,i] <- ifelse(SIPdf[,i] > df[i,2], 0, 50000)
}

#Next, lets use the functions below to generate our SLURP. Both of these functions are included in the "SLURPCreationFunctions.R" file. We'll also include the average and median, which are generated from the SIPs themselves:

SIPMetaDF(df, names, c("closure_rate","salary","age"))

CreateSLURP(SIPdf, testslurp.xml, average = TRUE, median = TRUE, meta = MetaDF)

#Using the code will generate an xml file in your current Working Directory with name "testslurp". You can import this .xml file into Excel with the SIPmath Tools to generate a library for further modeling/analysis.

