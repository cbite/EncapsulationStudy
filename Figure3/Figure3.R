########## Figure 3 ###################################
####### Script written by T.J.M. Kuijpers #############
#######################################################

## Load the libraries
library(xlsx)
library(ggplot2)

path_to_working_directory="C:/Users/tkuijpe1/Desktop/Encapsulation Figures/"
setwd(path_to_working_directory)
data=read.csv("ranking_list.csv_Cell count.csv")
data_to_plot=data[,c('feature.idx','median','mean','ranksum')]
color_coding=read.csv("ColorCodingFeatures.csv")

# Rank the data based on the mean cell count
data_to_plot$Rank<-rank(data_to_plot$ranksum)
data_to_plot_merged_with_color=merge(data_to_plot,color_coding,by.x='feature.idx',by.y='FeatureIDx')
# create a scatterplot

svg("RankingPlot.svg")
ggplot(data_to_plot_merged_with_color, aes(x=Rank, y=mean))+theme_classic(base_size = 14) + geom_point(color=data_to_plot_merged_with_color$Color, size = data_to_plot_merged_with_color$PointSize/1.5)+ scale_x_continuous(name="Ranksum of the TopoUnit", limits=c(0, 2178)) +
  scale_y_continuous(name="Mean Cell Count per analyzed area", limits=c(0, 20))
dev.off()