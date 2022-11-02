#####################################################
############# Figure 6A #############################
#####################################################
# Original script + work: Aleksei Vasilevich        #  
# Transformed to reproducible code: T.J.M. Kuijpers #
#####################################################

# Import the libraries
library("corrplot")
library("FactoMineR")
library("factoextra")
library("caret")
library("randomForest")
library(rpart)
library(partykit)
library("corrplot")

# Set the working directory
setwd("C:/Users/tkuijpe1/Desktop/Encapsulation Figures/")

########################## Load the dataset and prepare for further analysis ##################

# Read the dataset (derived from Cellprofiler + post-pipeline)
dataset <- read.csv(file="Ranking_for_paper.csv",sep=";",header=TRUE,stringsAsFactors=FALSE)
dataset.internalidx<-read.csv(file="ranking_list.csv_Cell count.csv",header=TRUE, stringsAsFactors=FALSE)

# Merge the feature idx and internal idx
featuredata=merge(dataset.internalidx[,c("internal.idx","feature.idx")],dataset)
# Remove the first columns
featuredata=featuredata[,-1]

########################## Prepare the dataset for the Neural Network ###########################
neuralnetwork_data=featuredata
colnames(neuralnetwork_data)=c("FeatureIDx","Label")
neuralnetwork_data$Label=ifelse(neuralnetwork_data$Label=='Low',0,1)
# Check the neural network data
table(neuralnetwork_data$Label)
# Save the input data for the neural network
write.csv(neuralnetwork_data, file= "Featureidx_with_labels.csv", row.names = F)


################ Figure 6A: PCA plot ###############################################################
#full_feature_data=read.csv("TopoFeatureFull.csv", header=TRUE, stringsAsFactors=FALSE)
load("TopoFeatureFull.RData")
full_feature_data=TopoFeatureFull[,-c(3683:3720)]

# Merge the feature data with the ranking data
dataset_pca=merge(full_feature_data,featuredata[,c("feature.idx","Class")], by.y="feature.idx", by.x="Metadata_FeatureIdx" )
dataset_pca=dataset_pca[,colnames(dataset_pca)[!grepl("feature.dx",colnames(dataset_pca))]]
dataset_pca=dataset_pca[,colnames(dataset_pca)[!grepl("Texture",colnames(dataset_pca))]]

dataset_pca=dataset_pca[,colnames(dataset_pca)[!grepl("Zernike",colnames(dataset_pca))]]
dataset_pca=dataset_pca[,colnames(dataset_pca)[!grepl("Center",colnames(dataset_pca))]]
dataset_pca=dataset_pca[,colnames(dataset_pca)[!grepl("FCPLOG",colnames(dataset_pca))]]
dataset_pca=dataset_pca[,colnames(dataset_pca)[!grepl("Group_Indexa",colnames(dataset_pca))]]
dataset_pca=dataset_pca[,colnames(dataset_pca)[!grepl("Width",colnames(dataset_pca))]]
dataset_pca=dataset_pca[,colnames(dataset_pca)[!grepl("Height",colnames(dataset_pca))]]
dataset_pca=dataset_pca[,colnames(dataset_pca)[!grepl("WN",colnames(dataset_pca))]]

# Remove the NAs in the dataset
data0_nd=na.omit(dataset_pca[,-ncol(dataset_pca)])
# Remove those columns with zero variance (based on the nearZeroVar() function from the caret package)
data1_nzv=data0_nd[,-nearZeroVar(data0_nd)]


dataset_class=dataset_pca[,c("Metadata_FeatureIdx","Class")]
dataset_pca_numerical=dataset_pca[,colnames(dataset_pca)!="Metadata_FeatureIdx"]

# Calculate the principal components with PCA
macr.pca <- PCA(dataset_pca_numerical[,-ncol(dataset_pca_numerical)], graph = FALSE)
# Plot the variance explained (in percentage) by each principal component
fviz_eig(macr.pca, addlabels = TRUE, ylim = c(0, 50))

# Plot the first two principal components and the elipses per category (high and low ranked surface)
var <- get_pca_var(macr.pca)
svg("PCA_plot.svg")
fviz_pca_ind(macr.pca, geom.ind = "point",
             col.ind = dataset_pca_numerical[,ncol(dataset_pca_numerical)], # color by groups
             palette = c("Orange", "Blue"),
             addEllipses = TRUE, ellipse.type = "t",
		 pointsize=2,
             legend.title = "Groups"
)+labs(title="",, x = "PC1 (34.5%)", y = "PC2 (17.8%)")+
  theme(
    legend.text = element_text(size = 14),
    axis.text.x = element_text(size=12),
    axis.text.y = element_text(size=12),
    axis.title.x = element_text(size = 14),
    axis.title.y = element_text(size = 14),
    legend.title=element_text(size=15)
  )
dev.off()

################ Figure 6B: random forest model ############################################################
set.seed(107)

# Divide the full feature dataset in training and testing data for the random forest model
inTrain <- createDataPartition(y = dataset_pca$Class,p = .75, list = FALSE)
training <- dataset_pca[ inTrain,]
testing <- dataset_pca[-inTrain,]
cvCtrl <- trainControl(method = "cv", number = 10, classProbs = TRUE)

cvCtrl <- trainControl(method = "cv", number = 10, classProbs = TRUE)
rfTune <- train(training[,-ncol(training)], training[,ncol(training)], method = "rf",
                tuneLength = 30,
                metric = "Accuracy",
                trControl = cvCtrl)  


cl.res<-rpart(Class~.,  method="class", data=dataset_pca[,-1])
plot(cl.res)
text(cl.res)
cl.res.party<-as.party(cl.res)
svg("Tree_plot.svg")
plot(cl.res.party)
dev.off()

################ Figure 6C: Correlation plot ###############################################################
# First, we will shorten the feature names since they all start with Pattern_AreaShape_Area_
feature_names=colnames(dataset_pca_numerical)[grepl("Pattern_AreaShape_Area",colnames(dataset_pca_numerical))]
shorten_feature_names=lapply(feature_names,function(x) strsplit(x,split="Pattern_AreaShape_Area_")[[1]][2])
shorten_feature_names_unlisted=unlist(shorten_feature_names)

# data for correlation
dataset_for_correlation=dataset_pca_numerical[,colnames(dataset_pca_numerical)[grepl("Pattern_AreaShape_Area",colnames(dataset_pca_numerical))]]
colnames(dataset_for_correlation)=shorten_feature_names

# Calculate the correlations
corrFeat=cor(dataset_for_correlation)
# Plot the correlation matrix
svg("correlation_plot.svg")
correlation_plot=corrplot(corrFeat, type="lower", 
         method="color", 
         tl.col="black", 
         diag=FALSE,
 	   tl.cex = 1.5,
	   cl.cex=1.2)
dev.off()

################ Figure 6D: scatterplot low and high ranked surfaces #######################################
numerical_data_for_scatter_plot=full_feature_data[,c("Metadata_FeatureIdx","Pattern_AreaShape_Area_percentile_0.1","Space_AreaShape_MeanRadius.BigSpace")]
classes_for_scatter_plot=featuredata
dataset_for_scatter_plot=merge(numerical_data_for_scatter_plot,classes_for_scatter_plot,by.x="Metadata_FeatureIdx", by.y="feature.idx")
# plot the scatterplot
svg("Scatter_plot_Pattern.svg")
ggplot(dataset_for_scatter_plot, aes(x=Pattern_AreaShape_Area_percentile_0.1, y=Space_AreaShape_MeanRadius.BigSpace,color=Class))+theme_classic(base_size = 14) + geom_point(size = 2)+ scale_x_continuous(trans='log2',name="Pattern_AreaShape_Area_percentile_0.1 (Log scale)") +
  scale_y_continuous(name="Space_AreaShape_MeanRadius.BigSpace")+scale_colour_manual(values=c('High'="Orange", 'Low'="Blue"))
dev.off()