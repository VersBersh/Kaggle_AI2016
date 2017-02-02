#============================================================
#
# Institute of Actuaries Kaggle Competition 2016 - VicRoads
# Author: Oliver Chambers
# Date: 2016/12/15
# Description: XGBoost fit
#
#============================================================

rm(list=ls()); gc()

if( !require(data.table) ) {
    install.packages('data.table')
    library(data.table)
}

if( !require(caret) ) {
    install.packages('caret')
    library(caret)
}

if( !require(xgboost) ) {
    install.packages('xgboost')
    library(xgboost)
}

if( !require(arm) ) {
    install.packages('arm')
    library(arm)
}

#===========================================================
# useful functions
#===========================================================

cb <- function(x) write.table(x, "clipboard", sep="\t", row.names=FALSE)
rmse <- function(x, y){ sqrt(mean((x - y)^2)) }

#===========================================================
# partition data into fit/val/test sets
#===========================================================

# read in data
setwd('C:/Projects/Kaggle/AI2016')
train <- fread('./data/training_data.csv')
test  <- fread('./data/testing_data.csv')
roads <- fread('./data/roads.csv')

# combine datasets into one
train[, train := T]
test[, ':=' (train = F, COST = NA)]
all <- rbind(train, test)
rm(list=c('train', 'test'))

# define out of time sample
all[, year := as.numeric(substr(QUARTER,1,4))]
all[, quarter := substr(QUARTER,5,6)]
all[, oot := (year >= 2014)]
all <- merge(all, roads[, c('ROAD_ID',"BLOCK")], by='ROAD_ID', all.x=TRUE)

# encode quarter as four variables
q <- dummyVars('~ quarter', data = all, fullRank = F)
q <- data.frame(predict(q, newdata = all))
all[,names(q)] <- q
rm(q)

# define blocks for validation
set.seed(123)
train.blocks <- all[train == T, .(count = .N), by = BLOCK]
validation.blocks <- sample(train.blocks$BLOCK, 150, prob = train.blocks$count)

# partition data into fitting / validation / test
fit.data        <- all[(train == T) & (oot == F) & !(BLOCK %in% validation.blocks)]
validation.data <- all[(train == T) & ((oot == T) | (BLOCK %in% validation.blocks))]
test.data       <- all[train == F]
rm(list = c('all', 'train.blocks'))

#===========================================================
# encode categorical variables into dummies 
#===========================================================

# create indicators for landmark and intersection
roads[, HAS.LANDMARK    := as.numeric(!(LANDMARK == ''))]
roads[, IS.INTERSECTION := as.numeric(!(INTERSECTION_TYPE == ''))]
roads[, CARRIAGEWAY     := as.factor(CARRIAGEWAY)]

# the following variables are factors with an implicit order
roads[, SPEED_LIMIT := factor(SPEED_LIMIT, 
           levels = c( '<20mph',
                       '<30km/h',
                       '40km/h',
                       '50km/h',
                       '60km/h',
                       '70km/h',
                       '80km/h',
                       '90km/h',
                       '100km/h',
                       '110km/h' ))]


roads[, OPERATING_SPEED_85TH_PERCENTILE := factor(OPERATING_SPEED_85TH_PERCENTILE,
           levels = c( '<30km/h',
                       '35km/h',
                       '40km/h',
                       '45km/h',
                       '50km/h',
                       '55km/h',
                       '60km/h',
                       '65km/h',
                       '70km/h',
                       '75km/h',
                       '80km/h',
                       '85km/h',
                       '90km/h',
                       '95km/h',
                       '100km/h',
                       '105km/h',
                       '110km/h' ))] 


roads[, OPERATING_SPEED_MEAN := factor(OPERATING_SPEED_MEAN,
           levels = c( '<30km/h',
                       '35km/h',
                       '40km/h',
                       '45km/h',
                       '50km/h',
                       '55km/h',
                       '60km/h',
                       '65km/h',
                       '70km/h',
                       '75km/h',
                       '80km/h',
                       '85km/h',
                       '90km/h',
                       '95km/h',
                       '100km/h',
                       '110km/h' ))]


roads[, LANE_WIDTH := factor(LANE_WIDTH, 
           levels = c( 'Wide (? 3.25m)',
                       'Medium (? 2.75m to < 3.25m)',
                       'Narrow (? 0m to < 2.75m)'))]

roads[, PAVED_SHOULDER_DRIVERS_SIDE := factor(PAVED_SHOULDER_DRIVERS_SIDE,
           levels = c( 'None',
                       'Paved 0< Width<=1m',
                       'Paved 1< Width < 2.4m',
                       'Paved >= 2.4m' ))]
              
roads[, PAVED_SHOULDER_PASSENGER_SIDE := factor(PAVED_SHOULDER_PASSENGER_SIDE,
           levels = c( 'None',
                       'Paved 0< Width<=1m',
                       'Paved 1< Width < 2.4m',
                       'Paved >= 2.4m' ))]

roads[, CURVATURE := factor(CURVATURE,
           levels = c( 'Straight or gently curving',
                       'Moderate curvature',
                       'Sharp curve',
                       'Very sharp'))]

roads[, ROADSIDE_SEVERITY_DRIVERS_SIDE_DISTANCE := factor(ROADSIDE_SEVERITY_DRIVERS_SIDE_DISTANCE,
           levels = c( '0 to <1m',
                       '1 to <5m',
                       '5 to <10m',
                       '>=10m' ))]

roads[, ROADSIDE_SEVERITY_PASSENGER_SIDE_DISTANCE := factor(ROADSIDE_SEVERITY_PASSENGER_SIDE_DISTANCE,
           levels = c( '0 to <1m',
                       '1 to <5m',
                       '5 to <10m',
                       '>=10m' ))]
      
roads[, INTERSECTING_ROAD_VOLUME := factor(INTERSECTING_ROAD_VOLUME,
           levels = c( 'Not applicable',
                       'Very low',
                       'Low',
                       'Medium',
                       'High',
                       'Very high' ))]


ordered.factors  <- c( 'SPEED_LIMIT', 
                       'OPERATING_SPEED_85TH_PERCENTILE',
                       'OPERATING_SPEED_MEAN',
                        'LANE_WIDTH',
                       'PAVED_SHOULDER_DRIVERS_SIDE',
                       'PAVED_SHOULDER_PASSENGER_SIDE',
                       'CURVATURE',
                       'ROADSIDE_SEVERITY_DRIVERS_SIDE_DISTANCE',
                       'ROADSIDE_SEVERITY_PASSENGER_SIDE_DISTANCE',
                       'INTERSECTING_ROAD_VOLUME')
                 
unordered.factors<- c( 'CARRIAGEWAY',
                       'SHOULDER_RUMBLE_STRIPS',
                       'QUALITY_OF_CURVE',
                       'DELINEATION',
                       'GRADE',
                       'ROAD_CONDITION',
                       'INTERSECTION_TYPE',
                       'INTERSECTION_QUALITY',
                       'INTERSECTING_ROAD_VOLUME',
                       'MEDIAN_TYPE',
                       'SKID_RESISTANCE_GRIP',
                       'STREET_LIGHTING',
                       'ROADSIDE_SEVERITY_PASSENGER_SIDE_OBJECT',
                       'ROADSIDE_SEVERITY_DRIVERS_SIDE_OBJECT',
                       'ACCESS_POINTS',
                       'PAVED_SHOULDER_DRIVERS_SIDE',
                       'PAVED_SHOULDER_PASSENGER_SIDE',
                       'CENTRELINE_RUMBLE_STRIPS')

useless.factors  <- c( 'ROAD_NAME', 
                       'LANDMARK', 
                       'LENGTH', 
                       'Latitude', 
                       'Longitude', 
                       'TRAVEL_DIRECTION',
                       'BLOCK')



# encode factors as dummy variables	
# for ordered factors perform one-hot encoding and naive encoding
for( name in ordered.factors ) {
    print(paste('dummying ', name, '...', sep=''))
    rank = ifelse(length(unique(roads[[name]]))==2, T, F)
    c <- dummyVars(paste('~',name), data = roads, fullRank = rank)
    c <- data.frame(predict(c, newdata = roads))
    roads[,names(c)] <- c
    roads[,(name):=as.numeric(roads[[name]])]
}

# for ordered factors perform one-hot encoding and remove existing variable
for( name in unordered.factors ) {
    print(paste('dummying ', name, '...', sep=''))
    rank = ifelse(length(unique(roads[[name]]))==2, T, F)
    c <- dummyVars(paste('~',name), data = roads, fullRank = rank)
    c <- data.frame(predict(c, newdata = roads))
    roads[,names(c)] <- c
    roads[,(name):=NULL]
}

# remove factors that are useless predictos
roads[, (useless.factors) := NULL]


#===========================================================
# prepare data for xgboost 
#===========================================================

prep.data <- function(data, col.range) {
    
    # collapse the table accross the time dimension
    data <- data[, .(cost = mean(COST)), by = .(ROAD_ID, quarter)]
    data <- merge(data, roads[,col.range,with=F], by = 'ROAD_ID', all.x=TRUE)
    
    # delete columns that xgboost doesn't need
    ID <- data[, .(ROAD_ID, quarter)]
    data[, c('ROAD_ID', 'quarter') := NULL]
    
    # convert all columns to numeric
    data[, names(data) := lapply(.SD, as.numeric)]
    
    # convert data to xgboost matrix
    cost <- data$cost
    data[, cost := NULL]
    X <- list(ID = ID, names = names(data), M = xgb.DMatrix(data=as.matrix(data), label=cost), data=as.matrix(data))
    
    return(X)
}

source('./topVars.r')
x.fit <- prep.data(fit.data, top.vars)
x.val <- prep.data(validation.data, top.vars)

# note that test comes first in the watch list so that xgboost uses that for early stoping
watchlist <- list(test=x.val$M, train = x.fit$M) 



#===========================================================
# fit xgboost
#===========================================================

params <- list( eta = 0.05, 
                max_depth = 7, 
                sub_sample = 0.7, 
                alpha=0.8, 
                min_child_weight = 1000) 

xgb.model <- xgb.train( data=x.fit$M, 
                        params = params, 
                        objective='reg:linear', 
                        nrounds=1000, 
                        watchlist = watchlist, 
                        early.stop.round = 30, 
                        maximize = F)

# merge prediction back onto uncollapsed dataset and test RMSE
pred <- cbind(x.val$ID, pmax(0, predict(xgb.model, x.val$M)))
pred <- merge(validation.data[,c('ROAD_ID','quarter','COST'),with=F], pred, by = c('ROAD_ID', 'quarter'), all.x = T)
rmse(pred$COST, pred$V2)

#===========================================================
# fit bayesglm
#===========================================================

# convert fit and val data to a dataframe that can be used by Caret API
# also add the xgboost prediction as an 
t <- as.data.frame(x.fit$data, col.names = x.fit$names)
t$COST <- getinfo(x.fit$M, 'label')
t$xgb  <- predict(xgb.model, x.fit$M)

v <- as.data.frame(x.val$data, col.names = x.val$names)
v$COST <- getinfo(x.val$M, 'label')
v$xgb  <- predict(xgb.model, x.val$M)

# fit the bayesGLM

bg.model <- train(COST ~ ., 
                  data = t,
                  method = "bayesglm")


pred.bg <- cbind(x.val$ID, pmax(0, predict(bg.model, newdata=v)))
pred.bg <- merge(validation.data[,c('ROAD_ID','quarter','COST'),with=F], pred.bg, by = c('ROAD_ID', 'quarter'), all.x = T)
rmse(pred.bg$COST, pred.bg$V2)

combo <- (0.4*pred.bg$V2 + 0.6*pred.xgb$V2)
rmse(pred.bg$COST, combo)



#===========================================================
# submission
#===========================================================

tx <- prep.data(test.data, top.vars)
tg <- as.data.frame(tx$data, col.names = tx$names) 
tg$xgb <- predict(xgb.model, tx$M)

pred.x <- cbind(tx$ID, pmax(0, predict(xgb.model, tx$M)))
pred.x <- merge(test.data[,c('ROAD_ID','quarter','COST'),with=F], pred.x, by = c('ROAD_ID', 'quarter'), all.x = T)

pred.g <- cbind(tx$ID, pmax(0, predict(bg.model, newdata=tg)))
pred.g <- merge(test.data[,c('ROAD_ID','quarter','COST'),with=F], pred.g, by = c('ROAD_ID', 'quarter'), all.x = T) 

test.data[, COST := 0.6*pred.x$V2 + 0.4*pred.g$V2]
write.csv(test.data[, c('ID', 'COST'), with=FALSE], file='./submission3.csv', row.names = FALSE)


