#Authors:Aaron Brown, Daniel O'Neil

##########################
#SLURP CREATION FUNCTIONS#
##################################################################################################################################################################################################################################
##Creates the data frame for metadata. df is the original data, IDvect is a vector containing the row ID's of the original data, and metanamesvect is a vector that states the columns in string form from which to pull the data.

SIPMetaDF <- function(df, IDvect, metanamesvect) {
  xdf <- data.frame(matrix(nrow = length(metanamesvect), ncol = length(IDvect)))
  colnames(xdf) <- IDvect
  for(j in 1:length(IDvect)) {
    for(i in 1:length(metanamesvect)) {
      xdf[i,j] <- as.character(df[match(IDvect[j], df[[1]]), match(metanamesvect[i], colnames(df))])
    }
  }
  xdf$Named <- metanamesvect
  colnames(xdf)[length(IDvect)+1] <- "Meta"
  MetaDF <<-xdf
  return(xdf)
}

#Example: SIPMetaDF(data, c(0001,0002,0003), c("Location", "Date")) will return a dataframe where the jobs 0001, 0002, and 0003 have location and date information.


###
CreateSLURP <- function(dataframe,filename,index=FALSE,provenance="",csvr=4,average=FALSE,median=FALSE,meta=NULL) {
  #The CreateSLURP function will break the columns of a DataFrame into SIPS, by taking each column and pasting it into string.
  #The paste function will add all of the additional information to the string including necessary formatting.
  #The string will be saved to a variable to be added to the SLURP string, which has its own set of formatting.
  #The "write" function will save the string to a file in the current working directory with the name set by "filename".

  SIPS <- NULL      #SIPS is the variable to add all SIPS into 1 string. The format of the vector at the end is c( SIPS(1st SIP), SIPS(2nd SIP),... SIPS(n SIP) )
  metas <- NULL
  res <- ""            #res is the variable to take the vector of SIPS and create a single string from the vector of strings.
  res.meta <- ""
  if (index==TRUE) start<- 2 else start<- 1          #If there is an index, we dont want to add the index to the SLURP. Start at column 2 if there is an index

  metadata.function <- function(i) {                 #for loop iterrating over the number of columns:1 to n without an index, 2 to n with index
    for (j in 1:nrow(meta[i])) {
      metas <- c(metas,
                 paste(" ",meta[j,length(meta)],'="',meta[j,i],'"',collapse = "",sep = ""))                          #Paste metadata (in development)
      }
    for (metadata in metas) {
    res.meta <- paste(res.meta,metadata,sep = "")
    }
  return(res.meta)}

  for (i in start:ncol(dataframe)) {                 #for loop iterrating over the number of columns:1 to n without an index, 2 to n with index
    SIPS <- c(SIPS,
              paste( "<SIP name=",'"',colnames(dataframe[i]),'"',                                  #Paste the column name with SIP name
                     " count=",'"',length(dataframe[,i]),'"',                                      #Paste in the count of items in the column
                     " type=",'"',"CSV",'"',                                                       #Paste type with hardcoded default CSV
                     if (provenance!="") paste(" provenance=",'"',provenance,'"',sep = "")         #If Provenance is blank then skip, otherwise paste Provenance
                     else "",
                     metadata.function(i),                                                         #If there is metadata added, then use it by calling the metadata.function function
                     if (average==TRUE) paste(" average=",'"',mean(dataframe[,i]),'"')             #If average is true, take mean of column otherwise skip
                     else "",
                     if (median==TRUE) paste(" median=",'"',median(dataframe[,i]),'"')             #If median is true, take median of column otherwise skip
                     else "",
                     "> ",
                     paste(
                       if (is.numeric(csvr)==TRUE)
                         round(
                           dataframe[,i],                                                          #Paste the data from the current column
                           digits = as.numeric(csvr))                                              #Round by the CSVR argument
                       ,collapse = ",", sep = ", "),                                               #Separate the data with a comma
                     " </SIP>",                                                                    #End each string with the ending XML
                     "\n",                                                                         #At the end of the function, the write function will add each SIP to a new line with this
                     sep = "",
                     collapse = "") )
  }
  for (items in SIPS) {
    res <- paste(res,items, sep = "")
  }
  write(
    paste( "<SLURP name=",'"',deparse(substitute(dataframe)),'"',
           " provenance=",'"',provenance,'"',
           if (index==TRUE) paste(" count=",'"',ncol(dataframe)-1,'"')
           else paste(" count=",'"',ncol(dataframe),'"'),
           "> ",
           "\n",
           res,
           "</SLURP>",
           "\n",
           sep = "",
           collapse = ""),
    deparse(substitute(filename)),sep = "\n") }                                                    #Puts it all together and outputs the file in the current working directory


# Example: CreateSLURP(test.df,testdfxml21.xml,provenance = "Testing with 1000 values",csvr = 4,average = TRUE,median = FALSE,meta = MetaDF) outputs an XML file that named "testdfxml21.xml" can be read into Excel using the SIPmath tools to generate a library. It'll have the metadata included from the SIPMetaDf function.

