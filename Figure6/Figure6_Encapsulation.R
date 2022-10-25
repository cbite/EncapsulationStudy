###### CODE##########
###Load all requered packages
if (!require("pacman")) install.packages("pacman")
pacman::p_load(ape, caret, ROCR,picante,plyr,doParallel, ggplot2, corrplot,randomForest,e1071, PMCMR, FSA)

##Load the data, correct path to the file should be specified
# feat.data<-read.csv(file="indexes_and_features.csv",header=T, stringsAsFactors=F)
# median.data<-read.csv(file="medianeria_data_red_sort_2.csv",header=T, stringsAsFactors=F)
# TopoScreen.data<-read.csv(file="TopoScreen.csv",header=T, stringsAsFactors=F)
# 
# feat.data["median"]<-median.data[,1]
# feat.data["NewName"]<-median.data[,3]
# feat.data$NewName<- sprintf("%04d", feat.data$NewName) #add preleading zeros so that it is always four digits
# 
# TopoScreen.data["NewName"]<-median.data[,3]
# TopoScreen.data$NewName<- sprintf("%04d", TopoScreen.data$NewName) #add preleading zeros so that it is always four digits


feat.data<-read.csv2(file="Ranking_for_paper.csv",header=T, stringsAsFactors=F)
feat.data_internalidx<-read.csv(file="ranking_list.csv_Cell count.csv",header=T, stringsAsFactors=F)

feat.data=merge(feat.data_internalidx[,c("internal.idx","feature.idx")], feat.data)
feat.data=feat.data[,-1]

##generate data for NN

NN.feat.data=feat.data

colnames(NN.feat.data)=c("FeatureIdx","Label")
NN.feat.data$Label=ifelse(NN.feat.data$Label=="Low",0,1)

table(NN.feat.data$Label)

write.csv2(NN.feat.data, file= "../Macrophage/Generating TopoGRaphies/Make representative topography image/Featureidx_with_labels.csv", row.names = F)

# full.feat.data= read.csv2(file="../../../../TopoChip/Features/TopoFeatureFull.csv", header=T, stringsAsFactors=F)
# 
# full.feat.data=full.feat.data[,-c(3683:3720)]



# 
# nrow(feat.data[feat.data$number.of.outliers<10,])
# 
# plot(feat.data$number.of.outliers,feat.data$ranksum)
# feat.data=feat.data[feat.data$number.of.outliers<10,]
# 
# hist(feat.data$number.of.outliers)
# 
# hist(feat.data$ranksum)
# 
# feat.data$median=feat.data$ranksum
# 
# feat.data=feat.data[,-c(1,3:12,14:19, 57)]
# 
# hist(feat.data$median)




feat.data=merge(full.feat.data,feat.data[,c("feature.idx","Class")], by.y="feature.idx", by.x="Metadata_FeatureIdx" )

feat.data=feat.data[,colnames(feat.data)[!grepl("feature.dx",colnames(feat.data))]]

feat.data=feat.data[,colnames(feat.data)[!grepl("Texture",colnames(feat.data))]]

feat.data=feat.data[,colnames(feat.data)[!grepl("Zernike",colnames(feat.data))]]

feat.data=feat.data[,colnames(feat.data)[!grepl("Center",colnames(feat.data))]]

feat.data=feat.data[,colnames(feat.data)[!grepl("FCPLOG",colnames(feat.data))]]

feat.data=feat.data[,colnames(feat.data)[!grepl("Group_Indexa",colnames(feat.data))]]

feat.data=feat.data[,colnames(feat.data)[!grepl("Width",colnames(feat.data))]]

feat.data=feat.data[,colnames(feat.data)[!grepl("Height",colnames(feat.data))]]

feat.data=feat.data[,colnames(feat.data)[!grepl("WN",colnames(feat.data))]]

data0_nd=na.omit(feat.data[,-ncol(feat.data)])

data1_nzv=data0_nd[,-nearZeroVar(data0_nd)]

# ##Ecslude highly corrlelated values
# data1_nzv_cor = cor(data1_nzv)
# 
# nrow(data1_nzv_cor)
# ncol(data1_nzv_cor)
# 

feat.data.keep.class=feat.data[,c("Metadata_FeatureIdx","Class")]

# 
# hc = findCorrelation(data1_nzv_cor, cutoff=0.999999999999999999 ) # put any value as a "cutoff" 
# hc = sort(hc)
feat.data = data1_nzv#[,-c(hc)]

feat.data=merge(feat.data,feat.data.keep.class)

# #chose 100 lowest and highest - manually in excel
# feat.data[feat.data$median<(-2),"Class"]<-"Low"
# feat.data[feat.data$median>2,"Class"]<-"High"

# feat.data.high<-feat.data[feat.data$Class%in%c("High"),]
# feat.data.low<-feat.data[feat.data$Class%in%c("Low"),]

table(feat.data$Class)
unique(feat.data$Class)

feat.data$Class<-as.factor(feat.data$Class)

feat.data2<-feat.data[,colnames(feat.data)!="Metadata_FeatureIdx"]

library("corrplot")

corrFeat=cor(feat.data2[,colnames(feat.data2)[grepl("Pattern_AreaShape_Area",colnames(feat.data2))]])

colnames(corrFeat)=gsub("Pattern_AreaShape_Area_", "", colnames(corrFeat))
rownames(corrFeat)=gsub("Pattern_AreaShape_Area_", "", rownames(corrFeat))

corrplot(corrFeat)


##############
# PCA analysis 

library("FactoMineR")
library("factoextra")

macr.pca <- PCA(feat.data2[,-ncol(feat.data2)], graph = F)
fviz_eig(macr.pca, addlabels = TRUE, ylim = c(0, 50))

var <- get_pca_var(macr.pca)
var
head(var$contrib)
library("corrplot")

corrplot(var$cos2[order(-rowSums(var$cos2)),][1:20,], is.corr=FALSE)

fviz_cos2(macr.pca, choice = "var", axes = 1:2,top=10)

fviz_contrib(macr.pca, choice = "var", axes = 2, top = 10)
fviz_contrib(macr.pca, choice = "var", axes = 1, top = 10)


#http://www.sthda.com/english/articles/31-principal-component-methods-in-r-practical-guide/112-pca-principal-component-analysis-essentials/

head(var$coord)

plot(var$coord[,c(1,2)])

grp_coord=var$coord[,1]

# grp_coord=ifelse(var$coord[,"Dim.1"]<0&var$coord[,"Dim.2"]>0, "Top_Left",
#                  ifelse(var$coord[,"Dim.1"]>0&var$coord[,"Dim.2"]>0, "Top_Right",
#                         (var$coord[,"Dim.1"]<0&var$coord[,"Dim.2"]<0,"Bottom_Right", "Bottom_Left")))


Group_SIZE=colnames(feat.data2)[(grepl("Diameter", colnames(feat.data2))|
                                   grepl("Radius", colnames(feat.data2))|
                                   grepl("Length", colnames(feat.data2))|
                                   grepl("Perimeter", colnames(feat.data2))|
                                   grepl("AreaShape_Area", colnames(feat.data2))|
                                   grepl("AreaOccupied", colnames(feat.data2))
)]



Group_SHAPE=colnames(feat.data2)[(grepl("Eccentricity", colnames(feat.data2))|
                                    grepl("EulerNumber", colnames(feat.data2))|
                                    grepl("Extent", colnames(feat.data2))|
                                    grepl("FormFactor", colnames(feat.data2))|
                                    grepl("Solidity", colnames(feat.data2))|
                                    grepl("Compactness", colnames(feat.data2))
)]

Group_SIZE_Pattern=Group_SIZE[grepl("Pattern",Group_SIZE)]
Group_SIZE_Space=Group_SIZE[grepl("Space",Group_SIZE)]
Group_SHAPE_Pattern=Group_SHAPE[grepl("Pattern",Group_SHAPE)]
Group_SHAPE_Space=Group_SHAPE[grepl("Space",Group_SHAPE)]

Group_OTHER=colnames(feat.data2)[!colnames(feat.data2)%in%c(Group_SHAPE,Group_SIZE,"Class")]

grp <- ifelse(rownames(var$coord)%in%Group_SIZE_Pattern,"Size_Pattern",
              ifelse(rownames(var$coord)%in%Group_SIZE_Space,"Size_Space",
                     ifelse(rownames(var$coord)%in%Group_SHAPE_Pattern,"Shape_Pattern",
                            ifelse(rownames(var$coord)%in%Group_SHAPE_Space,"Shape_Space","Other"))))



# grp <- ifelse(grepl("Pattern",rownames(var$coord)),"Pattern",
#               ifelse(grepl("Space",rownames(var$coord)),"Space","Other"))
# Color variables by groups
fviz_pca_var(macr.pca, col.var = grp, geom = "arrow",
             #palette = c("#0073C2FF", "#EFC000FF", "#868686FF"),
             legend.title = "Type of Feature")

res.desc <- dimdesc(macr.pca, axes = c(1,2), proba = 0.05)
res.desc$Dim.1
res.desc$Dim.2

fviz_pca_ind(macr.pca, geom.ind = "point",
             col.ind = feat.data2[,ncol(feat.data2)], # color by groups
             palette = c("#00AFBB", "#E7B800"),
             addEllipses = TRUE, ellipse.type = "t",
             legend.title = "Groups"
)
##Exclude other

feat.data3=feat.data2[,!colnames(feat.data2)%in%Group_OTHER]

macr.pca2 <- PCA(feat.data3[,-ncol(feat.data3)], graph = F)

fviz_pca_ind(macr.pca2, geom.ind = "point",
             col.ind = feat.data2[,ncol(feat.data2)], # color by groups
             palette = c("#00AFBB", "#E7B800"),
             addEllipses = TRUE, ellipse.type = "t",
             legend.title = "Groups"
)

# fviz_pca_biplot(macr.pca, 
#                 # Individuals
#                 geom.ind = "point",
#                 fill.ind = iris$Species, col.ind = "black",
#                 pointshape = 21, pointsize = 2,
#                 palette = "jco",
#                 addEllipses = TRUE,
#                 # Variables
#                 alpha.var ="contrib", col.var = "contrib",
#                 gradient.cols = "RdYlBu",
#                 
#                 legend.title = list(fill = "Species", color = "Contrib",
#                                     alpha = "Contrib")
# )

##

##Factor/Correspondanse analysis

feat.data2$Class=as.character(feat.data2$Class)

##make groups from the variables
colnames(feat.data2)




res.mfa <- MFA(feat.data2[,c(Group_SIZE_Pattern,
                             Group_SIZE_Space,
                             Group_SHAPE_Pattern,
                             Group_SHAPE_Space,
                             Group_OTHER, "Class")], 
               group = c(length(Group_SIZE_Pattern),
                         length(Group_SIZE_Space),
                         length(Group_SHAPE_Pattern),
                         length(Group_SHAPE_Space),
                         length(Group_OTHER),1), 
               type = c("c","c","c","c","c","n"),
               name.group = c("Size_Pattern","Size_Space",
                              "Shape_Pattern","Shape_Space","Other","Class"),
               #num.group.sup = ncol(feat.data2),
               graph = FALSE)
print(res.mfa)
group <- get_mfa_var(res.mfa, "group")
group

fviz_mfa_var(res.mfa, "group")

quanti.var <- get_mfa_var(res.mfa, "quanti.var")
quanti.var 

# fviz_mfa_var(res.mfa, "quanti.var", palette = "jco", top=10,
#              col.var.sup = "violet", repel = TRUE)

fviz_mfa_var(res.mfa, "quanti.var", palette = "jco",top=10, 
             col.var.sup = "violet", repel = TRUE,
             geom = c("point", "text"), legend = "bottom")

fviz_contrib(res.mfa, choice = "quanti.var", axes = 1, top = 20,
             palette = "jco")

fviz_contrib(res.mfa, choice = "quanti.var", axes = 2, top = 20,
             palette = "jco")


fviz_mfa_ind(res.mfa, partial = "all") 

fviz_mfa_ind(res.mfa, 
             habillage = "Class", # color by groups 
             palette = c("#00AFBB", "#E7B800"),
             addEllipses = TRUE, ellipse.type = "confidence", 
             repel = TRUE # Avoid text overlapping
) 

library(MASS)

feat.data2.transformed=feat.data2

feat.data2.transformed[,colnames(feat.data2.transformed)!="Class"]=scale(feat.data2.transformed[,colnames(feat.data2.transformed)!="Class"])

model <- lda(Class~., data = feat.data2.transformed)
plot(model)

model

lda.data <- cbind(feat.data2.transformed, predict(model)$x)
ggplot(lda.data, aes(Class,LD1)) +
  geom_boxplot()

library(klaR)
partimat(G~x1+x2+x3,data=feat.data2.transformed,method="lda")

# hierarchical tree
library(rpart)
library(partykit)
#feat.data2<-feat.data[feat.data$Class%in%c("High","Low"),]
cl.res<-rpart(Class~.,  method="class", data=feat.data2[,-1])
plot(cl.res)
text(cl.res)
cl.res.party<-as.party(cl.res)
plot(cl.res.party)

# cl.res=prune(cl.res, 0.1)
# 
# cl.res.party<-as.party(cl.res)
# plot(cl.res.party)

library(ggplot2)

ggplot(feat.data2, aes(Pattern_AreaShape_Area_percentile_0.1, Space_AreaShape_MeanRadius.BigSpace, colour=Class)) +
  geom_point(size=5) + scale_x_continuous(trans='log2') +
  scale_colour_manual(values=c('High'="black", 'Low'="orange"))+theme_bw()

library(klaR)

feat.data_viz=feat.data2
feat.data_viz$Pattern_AreaShape_Area_percentile_0.1=log(feat.data_viz$Pattern_AreaShape_Area_percentile_0.1)

partimat(Class~Space_AreaShape_MeanRadius.BigSpace+Pattern_AreaShape_Area_percentile_0.1,data=feat.data_viz,method="rpart")

# ggplot(feat.data2, aes(Pattern_AreaShape_Area_mean, Space_AreaShape_Area.BigSpace, colour=Class)) +
#   geom_point(size=5) + scale_x_continuous(trans='log2') +
#   scale_colour_manual(values=c('High'="black", 'Low'="orange"))+theme_bw()


# ###Regression with trees #whole data
# cl.res<-rpart(feat.data$median~.,  method="anova", data=feat.data[,-c(1,2,ncol(feat.data))])
# plot(cl.res)
# text(cl.res)
# 
# cl.res.party<-as.party(cl.res)
# plot(cl.res.party)
# 
# cl.res=prune(cl.res, 0.03)
# 
# cl.res.party<-as.party(cl.res)
# plot(cl.res.party)
# 
# 
# 
# plot(feat.data$median,feat.data$FCP)
# plot(feat.data$median,feat.data$FCP*feat.data$FeatSize)

# Train ElasticNet model, low versus high

##code taken from phenome project

#data_for_model=data_for_model[sub("\\_.*", "",row.names(data_for_model))!="2177",]

data_for_model=na.omit(feat.data)
df1 <- data_for_model
#head(df1)
df2 <- df1#[,-1]
#set.seed(2017)
n2 <- nrow(df2)
#sample <- sample(seq(n2), size = n * 0.5, replace = FALSE)
mdlY <- as.matrix(df2["Class"])
mdlX <- as.matrix(df2[setdiff(colnames(df1),  c("Metadata_FeatureIdx","Class"))])


#  Find best alpha": ELASTIC NET WITH 0 < ALPHA < 1
a <- seq(0.001, 1, 0.05)
library(foreach)
search <- foreach(
  i = a, .combine = rbind, .packages='glmnet') %dopar%
  {
    cv <- cv.glmnet(mdlX, mdlY, family = "gaussian", nfold = n2, type.measure = "deviance", paralle = TRUE, alpha = i,type.gaussian = "naive")
    # plot(cv)
    data.frame(cvm = cv$cvm[cv$lambda == cv$lambda.1se], lambda.1se = cv$lambda.1se, alpha = i)
  }
search
cv3 <- search[search$cvm == min(search$cvm), ]
md3 <- cv.glmnet(mdlX, mdlY, family = "gaussian",type.gaussian = "naive",type.measure = "deviance", alpha = cv3$alpha, nfold = n2)

#summary(md3)

# coef(md3,s = "lambda.min")
# 
# plot(md3)

## show goodness of fit of the model

gr_stat=postResample(pred = as.numeric(predict(md3, mdlX, type = "response",s = "lambda.min")),
                     obs = as.numeric(mdlY))

ActY=as.numeric(mdlY)
PredY=as.numeric(predict(md3, mdlX,s = "lambda.min"))

gr_stat_c=cor(ActY, PredY)
##make nice graph
prtrda=ggplot(as.data.frame(cbind(PredY, ActY)),aes(x=PredY, y=ActY)) + geom_point(shape=1) +
  geom_smooth(method=lm, se=FALSE) + ggtitle(paste("Prediction on training data, ","", SelVar)) +
  scale_x_continuous(name = "Predicted") +
  scale_y_continuous(name = "Observed") +
  annotate("text", x=max(PredY)*0.95,y=min(ActY)*1.1, label = paste("R^2 ==","",round(as.numeric(gr_stat[2]),2) ), parse=T,colour = "blue",size = 5) +
  annotate("text", x=max(PredY)*0.95,y=min(ActY)*1.3, label = paste("r ==","",round(as.numeric(gr_stat_c),2) ), parse=T,colour = "blue",size = 5) +
  #annotate("text", x=max(PredY)*0.95,y=min(ActY)*1.7, label = paste("RMSE ==","",round(as.numeric(gr_stat[1]),2) ), parse=T,colour = "blue",size = 5)+
  annotate("text", x=max(PredY)*0.95,y=min(ActY)*1.5, label = paste("alpha ==","",round(as.numeric(cv3$alpha),4) ), parse=T,colour = "blue",size = 5)+
  theme_minimal()



### show importance of the features
##original coeficients shouls be resaled: see agresti method
#https://stats.stackexchange.com/questions/14853/variable-importance-from-glmnet/211396#211396

sds <- apply(mdlX, 2, sd)
coefs <- as.matrix(coef(md3, s = "lambda.min"))
std_coefs <- coefs[-1, 1] * sds
##plot feature importance
results <- as.data.frame(std_coefs)
results$VariableName <- rownames(results)
colnames(results) <- c('Weight','VariableName')
results <- results[order(-abs(results$Weight)),]
results <- results[(results$Weight != 0),]
#hist(results$Weight)
results=results[c(1:20),]
results$Coefficient=ifelse((results$Weight > 0), 'Positive', 'Negative')
results$Weight=abs(results$Weight)/max(abs(results$Weight))
results$VariableName=factor(results$VariableName,
                            levels = results$VariableName[order(results$Weight)])

featimpplot=ggplot(results,aes(x=VariableName, y=Weight, fill=Coefficient))+geom_bar(stat = "identity")+
  coord_flip()+scale_fill_manual(values=c("#E69F00", "#56B4E9"))+
  theme_minimal()+labs(x="Features",y="Scaled Importance")

#Show together
print(ggarrange(prtrda, featimpplot + rremove("x.text"), 
                labels = c("A", "B"), ncol = 2, nrow = 1))

# par(mar=c(5,25,1,0)) # increase y-axis margin. 
# xx <- barplot(results$Weight, width = 0.85, 
#               main = paste("Variable Importance -",SelVar), horiz = T, 
#               xlab = "< (-) importance >  < neutral >  < importance (+) >", axes = FALSE, 
#               col = ifelse((results$Weight > 0), 'blue', 'red')) 
# axis(2, at=xx, labels=results$VariableName, tick=FALSE, las=2, line=-0.3, cex.axis=1.5) 
# par(mar=c(5,5,5,5))


library(caret)
set.seed(107)
# feat.data2<-feat.data[feat.data$Class%in%c("High","Low"),]
# 
# plot(feat.data2$Class, feat.data2$Space_AreaShape_Extent.VertSpace)

feat.data2$Class<-as.factor(feat.data2$Class)
feat.data2=feat.data2[,-1]
inTrain <- createDataPartition(y = feat.data2$Class,p = .75, list = FALSE)
training <- feat.data2[ inTrain,]
testing <- feat.data2[-inTrain,]
nrow(training)
nrow(testing)

cvCtrl <- trainControl(method = "cv", number = 10, classProbs = TRUE)

library(doParallel)
cl <- makeCluster(detectCores(), type='PSOCK') #specify number of cores
registerDoParallel(cl)

rfTune <- train(training[,-ncol(training)], training[,ncol(training)], method = "rf",
                tuneLength = 30,
                metric = "Accuracy",
                trControl = cvCtrl)  # you determine the best model fit

stopCluster(cl)
registerDoSEQ()

plot(varImp(rfTune), top=10)

rfProbs <- predict(rfTune, testing[,-ncol(training)])#, type = "prob")  # you test the model fit on the testing set but frTune has so many properties, how to know which one to use???
confusionMatrix(rfProbs, testing$Class) #to determine true pos/neg pos etc.

rfProbs <- predict(rfTune, testing, type = "prob")[,2]
pred <- prediction(rfProbs, testing$Class)  #what is the difference with the confusion matrix or the lines of code above?
perf <- performance(pred,"tpr","fpr")
plot(perf,col="black",lty=3, lwd=3)
abline(a=0,b=1,col="grey",lwd=1,lty=3)

#tiff("Fig1.tiff", height = 5, width = 5, units = 'in', res=600)

perf_AUC=performance(pred,"auc") #Calculate the AUC value
AUC=perf_AUC@y.values[[1]]
perf_ROC=performance(pred,"tpr","fpr") #plot the actual ROC curve

plot(perf_ROC, main="ROC plot")
text(0.5,0.5,paste("AUC = ",format(AUC, digits=5, scientific=FALSE)))
abline(a=0,b=1,col="grey",lwd=1,lty=3)


# Train RF model, low versus high
library(caret)
set.seed(107)
# feat.data2<-feat.data[feat.data$Class%in%c("High","Low"),]
# 
# plot(feat.data2$Class, feat.data2$Space_AreaShape_Extent.VertSpace)

feat.data2$Class<-as.factor(feat.data2$Class)
feat.data2=feat.data2[,-1]
inTrain <- createDataPartition(y = feat.data2$Class,p = .75, list = FALSE)
training <- feat.data2[ inTrain,]
testing <- feat.data2[-inTrain,]
nrow(training)
nrow(testing)

cvCtrl <- trainControl(method = "cv", number = 10, classProbs = TRUE)

library(doParallel)
cl <- makeCluster(detectCores(), type='PSOCK') #specify number of cores
registerDoParallel(cl)

rfTune <- train(training[,-ncol(training)], training[,ncol(training)], method = "rf",
                tuneLength = 30,
                metric = "Accuracy",
                trControl = cvCtrl)  # you determine the best model fit

stopCluster(cl)
registerDoSEQ()

plot(varImp(rfTune), top=10)

rfProbs <- predict(rfTune, testing[,-ncol(training)])#, type = "prob")  # you test the model fit on the testing set but frTune has so many properties, how to know which one to use???
confusionMatrix(rfProbs, testing$Class) #to determine true pos/neg pos etc.

rfProbs <- predict(rfTune, testing, type = "prob")[,2]
pred <- prediction(rfProbs, testing$Class)  #what is the difference with the confusion matrix or the lines of code above?
perf <- performance(pred,"tpr","fpr")
plot(perf,col="black",lty=3, lwd=3)
abline(a=0,b=1,col="grey",lwd=1,lty=3)

#tiff("Fig1.tiff", height = 5, width = 5, units = 'in', res=600)

perf_AUC=performance(pred,"auc") #Calculate the AUC value
AUC=perf_AUC@y.values[[1]]
perf_ROC=performance(pred,"tpr","fpr") #plot the actual ROC curve

plot(perf_ROC, main="ROC plot")
text(0.5,0.5,paste("AUC = ",format(AUC, digits=5, scientific=FALSE)))
abline(a=0,b=1,col="grey",lwd=1,lty=3)

#plot(feat.data2$Class, feat.data2$Space_AreaShape_Compactness.BigSpace)

# 
# # train multiple times
# feat.data2<-feat.data[feat.data$Class%in%c("High","Low"),]
# feat.data2$Class<-as.factor(feat.data2$Class)
# feat.data2=feat.data2[,-1]
# class_data<-feat.data2[,"Class"]
# model_accuracy<-data.frame(Run=c(),Accuracy=c(),BalAccuracy=c(),ROC=c())
# bach.predictors<-data.frame(t(feat.data2[1,-ncol(feat.data2)])) 
# bach.predictors$X1<-0
# 
# for(i in 1:100)
# {
#   print(i)
#   
#   inTrain <- createDataPartition(y = feat.data2$Class,p = .75, list = FALSE)
#   training <- feat.data2[ inTrain,]
#   testing <- feat.data2[-inTrain,]
#   nrow(training)
#   nrow(testing)
# 
#   cvCtrl <- trainControl(method = "cv", number = 10, classProbs = TRUE)
# 
#   library(doParallel)
#   cl <- makeCluster(detectCores(), type='PSOCK') #specify number of cores
#   registerDoParallel(cl)
# 
#   rpartTune <- train(training[,-ncol(training)], training[,ncol(training)], method = "rf",
#                      tuneLength = 30,
#                      metric = "ROC",
#                      allowParallel=TRUE,
#                      trControl = cvCtrl)
# 
#   stopCluster(cl)
#   registerDoSEQ()
# 
#   bach.predictors.temp<-data.frame(varImp(rpartTune)$importance)  #importance of each feature in that model
#   bach.predictors<-cbind(bach.predictors,bach.predictors.temp[,][match(rownames(bach.predictors), rownames(bach.predictors.temp))])
#   colnames(bach.predictors)[ncol(bach.predictors)]<-paste("trial",i)  #add the model to the bach.predictors
# 
#   rpartPred <- predict(rpartTune, testing[,-ncol(testing)])#, type = "prob")
#   confusionMatrix(rpartPred, testing$Class)
# 
#   rpartProbs <- predict(rpartTune, testing, type = "prob")[,2]
# 
#   library(ROCR)
#   pred <- prediction(rpartProbs, testing$Class)
#   perf <- performance(pred,"tpr","fpr")
# 
#   accuracy<-confusionMatrix(rpartPred, testing$Class)
#   accuracy[[3]][1]
#   confmat<-table(rpartPred, testing$Class)
# 
#   auc<-performance(pred,"auc")
#   auc <- unlist(slot(auc, "y.values"))
# 
#   balanced_accuracy<-((confmat[2,2]/sum(confmat[,2]))+confmat[1,1]/sum(confmat[,1]))/2
#   balanced_accuracy
#   auc
#   results<-c(i,as.numeric(accuracy[[3]][1]),balanced_accuracy,auc)
#   model_accuracy<-rbind(model_accuracy,results)
# }
# 
# colnames(model_accuracy)<-c("Trial","Accuracy","Balanced accuracy","ROC")
# model_accuracy
# boxplot(model_accuracy[,-1])
# accu_mean<-mean(model_accuracy[,2])
# accu_sd<-sd(model_accuracy[,2])
# 
# library(ggplot2)
# ggplot(model_accuracy, aes(Accuracy,x="Accuracy")) + geom_jitter(width = 0.2, cex=5)+
#   ylim(0.4,1)+theme_bw()+
#   geom_errorbar(aes(ymin =accu_mean-accu_sd, ymax = accu_mean+accu_sd),
#                 colour = "red", width = 0.2, cex=2)+
#   geom_point(aes(accu_mean,x="Accuracy"), size=5, shape=21, fill="white")+
#   theme_minimal()+
#   ylab("Accuracy")+theme(#text = element_text(size=18),
#     axis.text.x = element_blank(),
#     axis.text.y = element_text(colour="grey20",size=32,angle=0,hjust=1,vjust=0,face="plain"),
#     axis.title.x = element_blank(),
#     axis.title.y = element_text(colour="grey20",size=32,angle=90,hjust=.5,vjust=.5,face="plain"))
# ##show combined feature importance
# bach.predictors.f<-bach.predictors[-47,-1]
# ##calculate statistics
# library(plyr)
# bach.pred.stat<-transform(bach.predictors.f, SD=apply(bach.predictors.f,1, sd, na.rm = TRUE),
#                           Mean=apply(bach.predictors.f,1, mean, na.rm = TRUE))
# bach.pred.stat.s<-bach.pred.stat[order(-bach.pred.stat$Mean),]
# #bach.pred.stat.s<-bach.pred.stat.s[(nrow(bach.pred.stat.s)-20):nrow(bach.pred.stat.s),]
# bach.pred.stat.s<-bach.pred.stat.s[1:8,] #number of rows
# 
# bach.pred.stat.s$Cond<-row.names(bach.pred.stat.s)
# bach.pred.stat.s$Order<-c(nrow(bach.pred.stat.s):1)
# bach.pred.stat.s$Order<-as.factor(bach.pred.stat.s$Order)
# #levels(bach.pred.stat.s$Cond)<-as.character(bach.pred.stat.s$Cond)
# 
# #levels(bach.pred.stat.s$Cond)<-row.names(bach.pred.stat.s)
# 
# ggplot(bach.pred.stat.s, aes(y=Mean, x=Order)) +
#   geom_bar(stat="identity", position=position_dodge()) +
#   geom_errorbar(aes(ymin=Mean-SD, ymax=Mean+SD), width=.2, cex=0.4,
#                 position=position_dodge(.9),colour="red")+
#   scale_x_discrete(breaks=c(1:nrow(bach.pred.stat.s)),
#                    labels=rev(as.character(bach.pred.stat.s$Cond)))+
#   coord_flip() +  theme_classic() +  xlab("Parameter")+
#   ylab("Importance, a.u.")+theme(#text = element_text(size=18),
#     axis.text.x = element_text(colour="grey20",size=20,angle=0,hjust=.5,vjust=.5,face="plain"),
#     axis.text.y = element_text(colour="grey20",size=20,angle=0,hjust=1,vjust=0,face="plain"),
#     axis.title.x = element_text(colour="grey20",size=32,angle=0,hjust=.5,vjust=0,face="plain"),
#     axis.title.y = element_text(colour="grey20",size=32,angle=90,hjust=.5,vjust=.5,face="plain"))

# 
# ##########################################
# # Train RF model, whole dataset regression
# # library(caret)
# # set.seed(107)
# # 
# # ##remove redundant features
# # 
# # data0_nd=na.omit(feat.data[,-3722])
# # 
# # data1_nzv=data0_nd[,-nearZeroVar(data0_nd)]
# # 
# # data1_nzv_cor = cor(data1_nzv)
# # 
# # nrow(data1_nzv_cor)
# # ncol(data1_nzv_cor)
# # 
# # hc = findCorrelation(data1_nzv_cor, cutoff=0.75 ) # putt any value as a "cutoff" 
# # hc = sort(hc)
# # feat.data = data1_nzv[,-c(hc)]
# #ncol(feat.data2)
# 
# 
# 
# inTrain <- createDataPartition(y = feat.data$median,p = .75, list = FALSE)
# training <- feat.data[ inTrain,]
# testing <- feat.data[-inTrain,]
# nrow(training)
# nrow(testing)
# 
# cvCtrl <- trainControl(method = "cv", number = 10, classProbs = TRUE)
# 
# library(doParallel)
# cl <- makeCluster(detectCores(), type='PSOCK')
# registerDoParallel(cl)
# 
# rfTune <- train(training[,-c(1,2,ncol(training))], training[,1], method = "rf",
#                 tuneLength = 10,
#                 metric = "RMSE",
#                 allowParallel=TRUE,
#                 importance=TRUE,
#                 trControl = cvCtrl)  # you determine the best model 
# stopCluster(cl)
# registerDoSEQ()
# 
# summary(rfTune)
# 
# #predictors(rfTune)
# plot(varImp(rfTune), top=10)
# rfTune$finalModel
# ##plot prediction vs observed  #you need correlations; you cannot do confusion matrix as you do not have any classes
# #in caret package 
# predTargets <- extractPrediction(list(rfTune), testX=training)
# plotObsVsPred(predTargets)
# #in latice extra package
# Observed = training$median
# Predicted = predict(rfTune$finalModel, training[,-c(1,2,ncol(training))])
# cor(Observed,Predicted)
# plot(Observed,Predicted)
# library(latticeExtra)
# xyplot(Observed ~ Predicted, panel = function(x, y, ...) {
#   panel.xyplot(x, y, ...)
#   panel.lmlineq(x, y, adj = c(1,0), lty = 1,xol.text='red',
#                 col.line = "blue", digits = 1,r.squared =TRUE)
# })
# 
# ##compare 
# predTargets.c <- extractPrediction(list(rfTune), testing)
# plotObsVsPred(predTargets.c)
# 
# 
# ##########################################
# # Train RF model, whole dataset regression; multiple times
# model_accuracy<-data.frame(Run=c(),RMSE=c(),Rsquared=c(), Cor=c())
# bach.predictors<-data.frame(t(feat.data[1,-ncol(feat.data)])) 
# bach.predictors$X1<-0
# for(i in 1:100)
# {
# library(caret)
# set.seed(107)
# inTrain <- createDataPartition(y = feat.data$median,p = .75, list = FALSE)
# training <- feat.data[ inTrain,]
# testing <- feat.data[-inTrain,]
# nrow(training)
# nrow(testing)
# 
# cvCtrl <- trainControl(method = "cv", number = 10, classProbs = TRUE)
# 
# library(doParallel)
# cl <- makeCluster(detectCores(), type='PSOCK')
# registerDoParallel(cl)
# 
# rfTune <- train(training[,-c(1,2,ncol(training))], training[,1], method = "rf",
#                 tuneLength = 10,
#                 metric = "RMSE",
#                 allowParallel=TRUE,
#                 importance=TRUE,
#                 trControl = cvCtrl)  # you determine the best model 
# 
# stopCluster(cl)
# registerDoSEQ()
# 
# summary(rfTune)
#   
# bach.predictors.temp<-data.frame(varImp(rfTune)$importance)  #importance of each feature in that model
# bach.predictors<-cbind(bach.predictors,bach.predictors.temp[,][match(rownames(bach.predictors), rownames(bach.predictors.temp))])
# colnames(bach.predictors)[ncol(bach.predictors)]<-paste("trial",i)  #add the model to the bach.predictors
# 
# predTargets <- extractPrediction(list(rfTune), testX=testing)
# plotObsVsPred(predTargets)
# #in latice extra package
# Observed = testing$median
# Predicted = predict(rfTune, testing)
# correl<-cor(Observed,Predicted)
# 
# results<-c(i, mean(rfTune$finalModel$mse), mean(rfTune$finalModel$rsq),correl) 
# model_accuracy<-rbind(model_accuracy,results)
# }
# 
# 
# colnames(model_accuracy)<-c("Trial","RMSE","Rsquared","cor")
# model_accuracy
# boxplot(model_accuracy[,-1])
# accu_mean<-mean(model_accuracy[,2])
# accu_sd<-sd(model_accuracy[,2])
# 
# library(ggplot2)
# ggplot(model_accuracy, aes(Accuracy,x="Accuracy")) + geom_jitter(width = 0.2, cex=5)+
#   ylim(0.4,1)+theme_bw()+
#   geom_errorbar(aes(ymin =accu_mean-accu_sd, ymax = accu_mean+accu_sd),
#                 colour = "red", width = 0.2, cex=2)+ 
#   geom_point(aes(accu_mean,x="Accuracy"), size=5, shape=21, fill="white")+
#   theme_minimal()+
#   ylab("Accuracy")+theme(#text = element_text(size=18),
#     axis.text.x = element_blank(),
#     axis.text.y = element_text(colour="grey20",size=32,angle=0,hjust=1,vjust=0,face="plain"),  
#     axis.title.x = element_blank(),
#     axis.title.y = element_text(colour="grey20",size=32,angle=90,hjust=.5,vjust=.5,face="plain"))
# ##show combined feature importance
# bach.predictors.f<-bach.predictors[-47,-1]
# ##calculate statistics
# library(plyr)
# bach.pred.stat<-transform(bach.predictors.f, SD=apply(bach.predictors.f,1, sd, na.rm = TRUE),
#                           Mean=apply(bach.predictors.f,1, mean, na.rm = TRUE))
# bach.pred.stat.s<-bach.pred.stat[order(-bach.pred.stat$Mean),]
# #bach.pred.stat.s<-bach.pred.stat.s[(nrow(bach.pred.stat.s)-20):nrow(bach.pred.stat.s),]
# bach.pred.stat.s<-bach.pred.stat.s[1:8,] #number of rows
# 
# bach.pred.stat.s$Cond<-row.names(bach.pred.stat.s)
# bach.pred.stat.s$Order<-c(nrow(bach.pred.stat.s):1)
# bach.pred.stat.s$Order<-as.factor(bach.pred.stat.s$Order)
# #levels(bach.pred.stat.s$Cond)<-as.character(bach.pred.stat.s$Cond)
# 
# #levels(bach.pred.stat.s$Cond)<-row.names(bach.pred.stat.s)
# 
# ggplot(bach.pred.stat.s, aes(y=Mean, x=Order)) + 
#   geom_bar(stat="identity", position=position_dodge()) +
#   geom_errorbar(aes(ymin=Mean-SD, ymax=Mean+SD), width=.2, cex=0.4,
#                 position=position_dodge(.9),colour="red")+
#   scale_x_discrete(breaks=c(1:nrow(bach.pred.stat.s)),
#                    labels=rev(as.character(bach.pred.stat.s$Cond)))+
#   coord_flip() +  theme_classic() +  xlab("Parameter")+
#   ylab("Importance, a.u.")+theme(#text = element_text(size=18),
#     axis.text.x = element_text(colour="grey20",size=20,angle=0,hjust=.5,vjust=.5,face="plain"),
#     axis.text.y = element_text(colour="grey20",size=20,angle=0,hjust=1,vjust=0,face="plain"),  
#     axis.title.x = element_text(colour="grey20",size=32,angle=0,hjust=.5,vjust=0,face="plain"),
#     axis.title.y = element_text(colour="grey20",size=32,angle=90,hjust=.5,vjust=.5,face="plain"))
# 
# 
# #plot most important features; scatter
# plot(feat.data2[,c(22)], feat.data2[,c(25)], col=c("blue","red")[feat.data2$Class],  xlab="FCP", ylab="WN0.2")
# legend("topright", levels(feat.data2$Class), col=c("blue","red"), lwd=, lty=c(1,2))
# text(feat.data2[,c(22)], feat.data2[,c(25)],labels=feat.data2$NewName, cex= 0.7, pos=3, col=c("blue","red")[feat.data2$Class])
# 
# 
# #ward hierarchical clustering
# #package ape required
# library(ape)
# feat.data3 <- feat.data2
# row.names(feat.data3) <- feat.data2$NewName
# d <- dist(feat.data3[,c(14:50)], method = "euclidean") # distance matrix
# fit <- hclust(d, method="complete") 
# colors = c("blue", "red")
# groups <- feat.data3[,c(54)]
# #tiplabels(text=toString(feat.data2$feature.idx))
# plot(as.phylo(fit), type = "fan", tip.color = colors[groups],  no.margin = TRUE,
#      label.offset = 0, cex = 0.7)
# legend("topleft", legend = c("Low","High"), col=c("red", "blue"), lwd=, lty=c(1,2))
# 
