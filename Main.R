
library(readr)
library(randomForest)

titanic.test <- read_csv("~/R/Titanic/test.csv")
titanic.train <- read_csv("~/R/Titanic/train.csv")

titanic.train$IsTrainSet <- TRUE
titanic.test$IsTrainSet <- FALSE

titanic.test$Survived <- NA

titanic.consolidated <- rbind(titanic.train, titanic.test)

#Cast categorical value as factors.
titanic.consolidated$Pclass <- as.factor(titanic.consolidated$Pclass) #ordinal
titanic.consolidated$Sex <- as.factor(titanic.consolidated$Sex)
titanic.consolidated$Embarked <- as.factor(titanic.consolidated$Embarked)

#Imputation

#Get column names that have null values
names(which(colSums(is.na(titanic.consolidated))>0))

#What is the mode for the embarked table?
table(titanic.consolidated$Embarked)

#Cheap imputation:  Assign the mode to the missing values 
titanic.consolidated[is.na(titanic.consolidated$Embarked), "Embarked"] <- 'S'

#Just use the median age for now.
#age.median <- median(titanic.consolidated$Age, na.rm = TRUE)
#titanic.consolidated[is.na(titanic.consolidated$Age), "Age"] <- age.median

boxplot.stats(titanic.consolidated$Age)

upper.whisker.age <- boxplot.stats(titanic.consolidated$Age)$stats[5]
outlier.filter.age <- titanic.consolidated$Age < upper.whisker.age
titanic.filtered.age <- titanic.consolidated[outlier.filter.age,]

age.equation = "Age~ Pclass + Sex + Fare + SibSp + Parch + Embarked"
age.model <- lm(
  formula = age.equation,
  data = titanic.filtered.age
)

age.row <- titanic.consolidated[
  is.na(titanic.consolidated$Age),
  c("Pclass", "Sex", "Fare", "SibSp", "Parch", "Embarked")
  ]
age.prediction <- predict(age.model, newdata = age.row)
titanic.consolidated[is.na(titanic.consolidated$Age), "Age"] <- age.prediction 


#Just use the median for fare.
#fare.median <- median(titanic.consolidated$Fare, na.rm = TRUE)
#titanic.consolidated[is.na(titanic.consolidated$Fare), "Fare"] <- fare.median

#Last whisker should be sufficent LM outlier filter
boxplot.stats(titanic.consolidated$Fare)

upper.whisker.fare <- boxplot.stats(titanic.consolidated$Fare)$stats[5]
outlier.filter.fare <- titanic.consolidated$Fare < upper.whisker.fare
titanic.filtered.fare <- titanic.consolidated[outlier.filter.fare,]

fare.equation = "Fare ~ Pclass + Sex + Age + SibSp + Parch + Embarked"
fare.model <- lm(
  formula = fare.equation,
  data = titanic.filtered.fare
)

fare.row <- titanic.consolidated[
  is.na(titanic.consolidated$Fare),
  c("Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked")
]
fare.prediction <- predict(fare.model, newdata = fare.row)
titanic.consolidated[is.na(titanic.consolidated$Fare), "Fare"] <- fare.prediction 


#Reassignment
titanic.train <- titanic.consolidated[titanic.consolidated$IsTrainSet==TRUE,]
titanic.test <- titanic.consolidated[!titanic.consolidated$IsTrainSet==TRUE,]

titanic.train$Survived <- as.factor(titanic.train$Survived)


#Declare predictors for RandomForest 

survived.equation <- "Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked"
survived.formula <- as.formula(survived.equation)
titanic.model <- randomForest(formula = survived.formula, data = titanic.train, 
                              ntree = 500, mtry = 3, nodesize = 0.01 * nrow(titanic.test))


titanic.prediction <- predict(titanic.model, newdata = titanic.test)

PassengerId <- titanic.test$PassengerId
output.df <- as.data.frame(PassengerId)
output.df$Survived <- titanic.prediction


write.csv(output.df, file="kaggle_submission.csv", row.names = FALSE)
