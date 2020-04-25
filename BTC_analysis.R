#Loading libraries

library(dslabs)
library(tidyverse)
library(caret)
library(lubridate)
library(readxl)
library(psych)
library(MASS)
library(rpart)

#Reading the database
#I'm attaching the original excel file so, if you are going to run the code,
#don't forget to change the root directory.

exchange_mxn <- read_excel("~/Cesar/edX/Data Science/9. Capstone Project/My ML Algorithm/exchange_mxn.xlsx")
View(exchange_mxn)

#Searching for NA's

sum(is.na(exchange_mxn))

#Adjusting Date

exchange_mxn <- exchange_mxn %>% mutate(Date = Date - 25569)
as_date(exchange_mxn$Date)

#Exploring the data

head(exchange_mxn)

str(exchange_mxn)

# Exploring the summary

summary(exchange_mxn)

# Checking the Bitcoin variability

#Adding variability column

exchange_mxn <- exchange_mxn %>% mutate(var_rate = (BTC_close/BTC_open)-1)

summary(exchange_mxn$var_rate)

exchange_mxn %>% ggplot(aes(x = as_date(Date), y = var_rate)) +
  theme_bw() +
  geom_line() +
  labs(y = "Daily Variation",
       x = "Date",
       title = "Daily Variation in Bitcoin Price")

#Comparing behaviours

par(mfcol = c(2,2))
plot(as_date(exchange_mxn$Date), exchange_mxn$BTC_close)
plot(as_date(exchange_mxn$Date), exchange_mxn$USD)
plot(as_date(exchange_mxn$Date), exchange_mxn$GOLD)
plot(as_date(exchange_mxn$Date), exchange_mxn$AMZN)

#Reseting layout parameter

par(mfcol = c(1,1))

#Looking for correlations

correlations <- cor(exchange_mxn[,-c(24)])

data.frame(Correlations = correlations[,5])

btc_cor <- data.frame(Feature = row.names(correlations), corBTC = correlations[ ,5])
btc_cor

plot(cor(exchange_mxn[,-c(24)])[,5],
     main = "Correlation BTC_close ~ All Features",
     sub = "Excluding var_rate",
     xlab = "Feature",
     ylab = "Correlation")

#Keeping features with a correlation between +/- 0.1

btc_cor %>% filter(corBTC <= 0.1 & corBTC >= -0.1)

cor.test(exchange_mxn$BTC_close, exchange_mxn$Day)[3]
cor.test(exchange_mxn$BTC_close, exchange_mxn$Quarter)[3]
cor.test(exchange_mxn$BTC_close, exchange_mxn$USD)[3]
cor.test(exchange_mxn$BTC_close, exchange_mxn$EUR)[3]
cor.test(exchange_mxn$BTC_close, exchange_mxn$SAR)[3]
cor.test(exchange_mxn$BTC_close, exchange_mxn$AED)[3]
cor.test(exchange_mxn$BTC_close, exchange_mxn$INR)[3]
cor.test(exchange_mxn$BTC_close, exchange_mxn$OIL_WTI)[3]
cor.test(exchange_mxn$BTC_close, exchange_mxn$IPC)[3]

#Adjusting our new dataset

exchange_mxn <- exchange_mxn[, -c(1, 2, 8, 9, 12, 13, 17, 21, 24)]
head(exchange_mxn)

#Plotting by category

#Cryptocurrencies
pairs.panels(exchange_mxn[,2:5], gap=0)

#Exchanges
pairs.panels(exchange_mxn[,6:10], gap=0)

#Indexes
pairs.panels(exchange_mxn[,12:13], gap=0)

#Stocks
pairs.panels(exchange_mxn[,14:15], gap=0)


#Creating Train and Test sets and omitting columns Date and var_rate

set.seed(2049, sample.kind = "Rounding") #Calculated with the date 23/04/2020

test_index <- createDataPartition(exchange_mxn$BTC_close, times = 1, p = 0.2, list = FALSE)

train_set <- exchange_mxn[-test_index,]

test_set <- exchange_mxn[test_index,]

#Training a model with knn

knn_model <- train(BTC_close ~ ., data = train_set, method = "knn",
                   tuneGrid = data.frame(k = seq(3, 51, 2)))

knn_model

#Predicting on Test set with knn

y_hat_knn <- predict(knn_model, test_set)

qplot(y_hat_knn, test_set$BTC_close,
      main = "Prediction vs Real Data",
      xlab = "Predicted Price",
      ylab = "Real Price")

#Computing RMSE

knn_RMSE <- RMSE(y_hat_knn, test_set$BTC_close)

RMSE_comp <- data.frame(Method = "knn", RMSE = knn_RMSE)

RMSE_comp %>% knitr::kable()

#Computing the prediction variability

knn_var <- (y_hat_knn / test_set$BTC_close) - 1

summary(knn_var)

boxplot(knn_var)



#Training a model with glm

control <- trainControl(method = "cv", number = 10, p = .9)

glm_model <- train(BTC_close ~ ., method = "bayesglm", data = train_set,
                   trControl = control)

glm_model

#Predicting on Test set with glm

y_hat_glm <- predict(glm_model, test_set)

qplot(y_hat_glm, test_set$BTC_close,
      main = "Prediction vs Real Data",
      xlab = "Predicted Price",
      ylab = "Real Price")

#Computing RMSE

glm_RMSE <- RMSE(y_hat_glm, test_set$BTC_close)

RMSE_comp <- bind_rows(RMSE_comp, data.frame(Method = "glm", RMSE = glm_RMSE))

RMSE_comp %>% knitr::kable()

#Computing the prediction variability

glm_var <- (y_hat_glm / test_set$BTC_close) - 1

summary(glm_var)

boxplot(knn_var, glm_var)



#Training a model with method rf

mtry <- data.frame(mtry = c(2:10))

rf_model <- train(BTC_close ~ ., data = train_set, method = "rf", tuneGrid = mtry, importance = TRUE)

varImp(rf_model)

rf_model

#Predicting on Test set with rf

y_hat_rf <- predict(rf_model, test_set)

qplot(y_hat_rf, test_set$BTC_close,
      main = "Prediction vs Real Data",
      xlab = "Predicted Price",
      ylab = "Real Price")

#Computing RMSE

rf_RMSE <- RMSE(y_hat_rf, test_set$BTC_close)

RMSE_comp <- bind_rows(RMSE_comp, data.frame(Method = "rf", RMSE = rf_RMSE))

RMSE_comp %>% knitr::kable()

#Computing the prediction variability

rf_var <- (y_hat_rf / test_set$BTC_close) - 1

summary(rf_var)

boxplot(knn_var, glm_var, rf_var)



#Training a model with blassoAveraged

br_model <- train(BTC_close ~ ., method = "blassoAveraged", data = train_set)

br_model

#Predicting on Test set with blassoAveraged

y_hat_br <- predict(br_model, test_set)

qplot(y_hat_br, test_set$BTC_close,
      main = "Prediction vs Real Data",
      xlab = "Predicted Price",
      ylab = "Real Price")

#Computing RMSE

br_RMSE <- RMSE(y_hat_br, test_set$BTC_close)

RMSE_comp <- bind_rows(RMSE_comp, data.frame(Method = "br", RMSE = br_RMSE))

RMSE_comp %>% knitr::kable()

#Computing the prediction variability

br_var <- (y_hat_br / test_set$BTC_close) - 1

summary(br_var)

boxplot(knn_var, glm_var, rf_var, br_var,
        xlab = "Methods",
        ylab = "Variability")



#Creating an Ensemble

y_hat_ens <- (y_hat_knn + y_hat_glm + y_hat_rf + y_hat_br) / 4

#Computing Ensemble RMSE

ens_RMSE <- RMSE(y_hat_ens, test_set$BTC_close)

RMSE_comp <- bind_rows(RMSE_comp, data.frame(Method = "Ensemble", RMSE = ens_RMSE))

RMSE_comp %>% knitr::kable()

qplot(y_hat_ens, test_set$BTC_close,
      main = "Prediction vs Real Data",
      xlab = "Predicted Price",
      ylab = "Real Price")

#Computing the prediction variability

ens_var <- (y_hat_ens / test_set$BTC_close) - 1

summary(ens_var)

boxplot(knn_var, glm_var, rf_var, br_var, ens_var)







