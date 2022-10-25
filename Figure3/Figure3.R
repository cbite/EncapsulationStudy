########## Figure 3 ###################################
library(xlsx)

path_to_working_directory="C:/Users/tkuijpe1/Desktop/Encapsulation Figures/"
setwd(path_to_working_directory)
data=read.csv("ranking_list.csv_Cell count.csv")
data_to_plot=data[,c('feature.idx','median','mean','ranksum')]

# Rank the data based on the mean cell count
data_to_plot$Rank<-rank(data_to_plot$ranksum)

# create a scatterplot
library(ggplot2)
ggplot(data_to_plot, aes(x=Rank, y=mean))+theme_classic() + geom_point(color = "Black", size = 2)+ scale_x_continuous(name="Ranksum of the TopoUnit", limits=c(0, 2178)) +
  scale_y_continuous(name="Mean Cell Count per analyzed area", limits=c(3, 17))
