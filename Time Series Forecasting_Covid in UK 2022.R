library(forecast)
library(imputeTS)
library(xts)

covid = read.csv("/Users/sabrina/Documents/2021/Data Science/Masters_APU/Sem 3/TSF/owid-covid-data.csv")
#country: United Kingdom
uk_raw = covid[covid$location == 'United Kingdom',]
uk_raw = uk_raw[(c("date","new_cases_per_million"))]
uk_raw$date = as.Date(as.character(uk_raw$date), format = '%Y-%m-%d')

#extract rows from 1st Jan 2021 onwards
uk = subset(uk_raw, date>='2021-01-01')

#time series plot
uk_ts = xts(uk$new_cases_per_million, uk$date)
plot(uk_ts,ylab='New Confirmed Covid Cases per Million', xlab='Date', 
     main = 'New Confirmed Cases per Million Time Series in United Kingdom')

#perform imputation for missing values
colSums(sapply(uk_ts,is.na))
#spline imputation
uk_ts= na_interpolation(uk_ts, "spline")

uk_ts = ts(uk$new_cases_per_million, frequency=7)
plot(uk_ts)

#verifying trend
library(pastecs)
trend.test(uk_ts)
#p-value = 2.2x10^-16, Ho is rejected, trend component is present in the series
library(Kendall)
MannKendall(uk_ts)

#verifying seasonality
library(seastests)
isSeasonal(uk_ts)
#Kruskall Wallis
kw(uk_ts) #p-value = 0 < 0.05, Ho is rejected. There is seasonal variation in the series
par(mfrow = c(1,1))
seasonplot(uk_ts)

#decomposing time series
uk_ts_components = decompose(uk_ts)
plot(uk_ts_components)

#Data partitioning - 80% and 20%
k = round(length(uk_ts)*0.8,0) ;k
train = subset(uk_ts, end=k)
test = subset(uk_ts, start=k+1)

#holt winters method
#holt winters model A using HoltWinters function
hw_modelA = HoltWinters(train, seasonal = 'additive')

#hw model details
summary(hw_modelA)
hw_modelA$alpha
hw_modelA$beta
hw_modelA$gamma
#alpha = 0.5055, beta = 0.1151, gamma = 0.7320

hw_forecastA = forecast(hw_modelA, h=length(test))
accuracy(hw_forecastA, test)

#plotting holts winter trained model A and forecasts against historical data
par(mfrow = c(2,1))
plot(uk_ts, main = 'Trained Holts Winter Model on UK Covid Time Series', ylab = 'New cases per million')
legend('topleft', c("Historical data", "Train Data", "Forecasted Data"), col=c("black",'red','blue'), pch=15, y.intersp=0.1,x.intersp=0.5, bty='n')
lines(fitted(hw_modelA)[,1], col='red',lwd=2)
plot(hw_forecastA, main = 'Holts Winter Forecast',ylab = 'New cases per million')
mtext(side=3, line=0.5, at=33, 'Model A: 0.5055, beta = 0.1151, gamma = 0.7320')

#holt winters model B using hw function
hw_modelB = hw(train, initial = 'optimal')
summary(hw_modelB) 
#alpha = 0.5041, beta = 0.0581, gamma = 0.2851

hw_forecastB = forecast(fitted(hw_modelB), h=length(test))
accuracy(hw_forecastB,test)

#plotting holts winter trained model A and forecasts against historical data
plot(hw_forecastB, main = 'Holts Winter Forecast',ylab = 'New cases per million', xlab = 'Time')
lines(uk_ts, col='black', lwd=1)
lines(fitted(hw_modelB), col='red',lwd=2)
legend('topleft', c("Historical data", "Train Data", "Forecasted Data"), col=c("black",'red','blue'), pch=15, y.intersp=0.3,x.intersp=0.5, bty='n')
mtext(side=3, line=0.5, at=33, 'alpha = 0.5041, beta = 0.0581, gamma = 0.2851')


#decomposition model
#additive model
add_fit = decompose(uk_ts)
add_fit
plot(add_fit)
seadj = uk_ts - add_fit$seasonal
plot(uk_ts, main = 'Seasonal adjustment of UK Covid Cases', sub = '2021-01-01 / 2022-03-29', ylab = 'New cases per million')
legend('topleft', c("Historical data", "Seasonally Adjusted Data"), col=c("black",'red'), pch=15, y.intersp=0.5,x.intersp=0.5, bty='n')
lines(seadj, col='red', lwd=2)

add_modelA = stl(train, s.window='periodic')
summary(add_modelA)

add_forecastA = stlf(train, s.window='period', h=length(test))
accuracy(add_forecastA, test)

plot(add_forecastA, main = 'Additive Decomposition Model Forecast', ylab= 'New cases per million', xlab = 'Time')
lines(uk_ts, col='black', lwd=1)
fitted = trendcycle(add_modelA) + seasonal(add_modelA)
lines(fitted, col='red', lwd = 2)
legend('topleft', c("Historical data", "Train Data", "Forecasted Data"), col=c("black",'red','blue'), pch=15, y.intersp=0.3, x.intersp=0.5, bty='n')

#regression with dummy variables
season = seasonaldummy(train)
#creating time variable
time = 1:length(train)

reg_modelA = tslm(train ~ season+time)
summary(reg_modelA)

reg_forecastA = forecast(fitted(reg_modelA), h=length(test))
accuracy(reg_forecastA, test)
rbind(accuracy(reg_modelA), accuracy(reg_forecastA$mean,test))

plot(reg_forecastA, main = 'Regression Model Forecast', ylab= 'New cases per million', xlab = 'Time', ylim = c(0,4000))
lines(uk_ts, col='black', lwd=1)
lines(fitted(reg_modelA), col='red', lwd=2)
legend('topleft', c("Historical data", "Train Data", "Forecasted Data"), col=c("black",'red','blue'), pch=15, y.intersp=0.3, x.intersp=0.5, bty='n')

#self experiment: holts method
holt_modelA = holt(train, initial = 'optimal')
summary(holt_modelA) #alpha = 0.8672, beta = 0.0295

holt_forecastA = forecast(fitted(holt_modelA), h = length(test))
accuracy(holt_forecastA, test)

plot(uk_ts, main = 'Trained Holts Model on UK Covid Time Series', ylab = 'New cases per million' )
legend('topleft', c("Historical data", "Train Data", "Forecasted Data"), col=c("black",'red','blue'), pch=15, y.intersp=0.3, x.intersp=0.5, bty='n')
lines(fitted(holt_modelA), col='red', lwd=2)
plot(holt_forecastA, main = 'Regression Model Forecast', ylab= 'New cases per million')

#check for stationary 
library(uroot)
ch.test(uk_ts) 

#check for stationary in seasonal
library(tseries)
adf.test(uk_ts)

#check for stationary in trend
kpss.test(uk_ts, null=c("Trend")) #p-value < 0.05

#Ho is rejected

#p-values of all variables are less than 0.05. Series is not stationary

#acf and pacf
par(mfrow = c(1,1))
acf(uk_ts, ylim = c(-1,1))
pacf(uk_ts)

#seasonal differencing is required
#first degree of seasonal differencing
nsdiffs(uk_ts)
par(mfrow = c(1,2))
acf(diff(uk_ts,7), lag.max=35)
pacf(diff(uk_ts,7), lag.max = 35)

#stationarity testing
ch.test(diff(uk_ts,7)) #all variables significant, Series is stationary
adf.test(diff(uk_ts,7)) #p-value = 0.01 < 0.05. Ho is rejected. Series is stationary
kpss.test(diff(uk_ts,7), null=c("Trend")) #p-value = 0.1 > 0.05. Ho is accepted. Series is stationary in trend

#experiment: Applying first degree non seasonal differencing
acf(diff(diff(uk_ts,7),1), lag.max=35)
pacf(diff(diff(uk_ts,7),1), lag.max = 35)

nsdiffs(diff(diff(uk_ts,7),1))
ndiffs(diff(diff(uk_ts,7),1))
ch.test(diff(diff(uk_ts,7),1)) #one season has p value < 0.05, suggesting non-stationarity
adf.test(diff(diff(uk_ts,7),1)) # p = 0.01
kpss.test(diff(diff(uk_ts,7),1), null=c("Trend")) # p = 0.1

#proposing arima models
arima_modelA = Arima(uk_ts, order = c(3,0,0), seasonal = c(1,1,0))#arima(3,0,0)(1,1,0)
summary(arima_modelA)
accuracy(arima_modelA)#arima(3,0,0)(1,1,0)[7]

arima_modelB = auto.arima(uk_ts, trace=TRUE, d=1) #arima(2,1,0)(0,1,2)[7]
summary(arima_modelB)
accuracy(arima_modelB)#arima(2,1,0)(0,1,2)[7]

arima_modelC = auto.arima(uk_ts, trace=TRUE) #arima(5,0,2)(1,1,0)[7]
summary(arima_modelC)
accuracy(arima_modelC)#arima(5,0,2)(1,1,0)[7]

#check adequacy of each model
library(lmtest)
checkresiduals(arima_modelA) #ljung Box p value < 0.05, residuals mean = 0, 5 significant spike in ACF, residuals follow norm distribution
checkresiduals(arima_modelB) #ljung box p value < 0.05, residuals mean = 0, 2 sgnificant spikes in ACF, residuals follow norm distribution
checkresiduals(arima_modelC)
#check significant of coefficients
coeftest(arima_modelA) #only three variables are significant, ar2 is not significant
coeftest(arima_modelB) #all variables are significant
coeftest(arima_modelC)


#perform forecasting

par(mfrow=c(2,1))

arima_forecastA = forecast(arima_modelA, h = 5)
plot(arima_forecastA)
lines(fitted(arima_modelA), col='red', lwd=2)
legend('topleft', c("Historical data", "Trained Data", "Forecasted Data"), col=c("black",'red','blue'), pch=15, y.intersp=0.3, x.intersp=0.5, bty='n')


arima_forecastB = forecast(arima_modelB, h = 5)
plot(arima_forecastB)
lines(fitted(arima_modelB), col='red', lwd=2)
legend('topleft', c("Historical data", "Trained Data", "Forecasted Data"), col=c("black",'red','blue'), pch=15, y.intersp=0.3, x.intersp=0.5, bty='n')



