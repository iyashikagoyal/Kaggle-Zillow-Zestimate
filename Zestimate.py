
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import scipy.stats as stats


# In[2]:


import lightgbm as lgb


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
pd.options.display.max_columns = 999


# In[4]:


trainData = pd.read_csv("train_2016_v2.csv", parse_dates = ["transactiondate"])


# In[5]:


property_2016 = pd.read_csv("properties_2016.csv")


# In[6]:


property_2016.pooltypeid2.fillna(0,inplace = True)


# In[7]:


property_2016.pooltypeid7.fillna(0,inplace = True)


# In[8]:


property_2016['poolcnt'].fillna(0, inplace = True)


# In[9]:


property_2016.hashottuborspa.fillna(0,inplace = True)


# In[10]:


property_2016.hashottuborspa.replace(to_replace = True, value = 1,inplace = True)


# In[11]:


pss = property_2016.loc[property_2016.poolcnt==1, 'poolsizesum'].fillna(property_2016[property_2016.poolcnt==1].poolsizesum.median())
property_2016.loc[property_2016.poolcnt==1, 'poolsizesum'] = pss


# In[12]:


property_2016.loc[property_2016.poolcnt==0, 'poolsizesum']=0


# In[13]:


def analysis():
    continuous = ['basementsqft', 'finishedfloor1squarefeet', 'calculatedfinishedsquarefeet', 
              'finishedsquarefeet12', 'finishedsquarefeet13', 'finishedsquarefeet15',
              'finishedsquarefeet50', 'finishedsquarefeet6', 'garagetotalsqft', 'latitude',
              'longitude', 'lotsizesquarefeet', 'poolsizesum',  'yardbuildingsqft17',
              'yardbuildingsqft26', 'yearbuilt', 'structuretaxvaluedollarcnt', 'taxvaluedollarcnt',
              'landtaxvaluedollarcnt', 'taxamount']

    discrete = ['bathroomcnt', 'bedroomcnt', 'calculatedbathnbr', 'fireplacecnt', 'fullbathcnt',
            'garagecarcnt', 'poolcnt', 'roomcnt', 'threequarterbathnbr', 'unitcnt',
            'numberofstories', 'assessmentyear', 'taxdelinquencyyear']
    
    for col in continuous:
        values = data[col].dropna()
        lower = np.percentile(values, 1)
        upper = np.percentile(values, 99)
        fig = plt.figure(figsize=(18,9));
        sns.distplot(values[(values>lower) & (values<upper)], color='Sienna', ax = plt.subplot(121));
        sns.boxplot(y=values, color='Sienna', ax = plt.subplot(122));
        plt.suptitle(col, fontsize=16)       
    
    NanAsZero = ['fireplacecnt', 'poolcnt', 'threequarterbathnbr']
    for col in discrete:
        if col in NanAsZero:
            data[col].fillna(0, inplace=True)
        values = data[col].dropna()   
        fig = plt.figure(figsize=(18,9));
        sns.countplot(x=values, color='Sienna', ax = plt.subplot(121));
        sns.boxplot(y=values, color='Sienna', ax = plt.subplot(122));
        plt.suptitle(col, fontsize=16)
        
    for col in objects:
        values = data[col].astype('str').value_counts(dropna=False).to_frame().reset_index()
        
        

    ### Adding percents over bars
    height = [p.get_height() for p in ax.patches]    
    total = sum(height)
    for i, p in enumerate(ax.patches):    
        ax.text(p.get_x()+p.get_width()/2,
                height[i]+total*0.01,
                '{:1.0%}'.format(height[i]/total),
                ha="center")  


# In[14]:


property_2016.drop('pooltypeid10', axis=1, inplace=True)


# In[15]:


property_2016.loc[(property_2016['fireplaceflag'] == True) & (property_2016['fireplacecnt'].isnull()), ['fireplacecnt']] = 1


# In[16]:


property_2016.fireplacecnt.fillna(0,inplace = True)


# In[17]:


def checklogerror():
    col = 'logerror'

    values = data_sold[col].dropna()
    lower = np.percentile(values, 1)
    upper = np.percentile(values, 99)
    fig = plt.figure(figsize=(18,9));
    sns.distplot(values[(values>lower) & (values<upper)], color='Sienna', ax = plt.subplot(121));
    sns.boxplot(y=values, color='Sienna', ax = plt.subplot(122));
    plt.suptitle(col, fontsize=16);
    for col in continuous:     
        fig = plt.figure(figsize=(18,9));
        sns.barplot(x='logerror_bin', y=col, data=data_sold, ax = plt.subplot(121),
                    order=['Large Negative Error', 'Medium Negative Error','Small Error',
                           'Medium Positive Error', 'Large Positive Error']);
        plt.xlabel('LogError Bin');
        plt.ylabel('Average {}'.format(col));
        sns.regplot(x='logerror', y=col, data=data_sold, color='Sienna', ax = plt.subplot(122));
        plt.suptitle('LogError vs {}'.format(col), fontsize=16)   


# In[18]:


property_2016.loc[(property_2016['fireplacecnt'] >= 1.0) & (property_2016['fireplaceflag'].isnull()), ['fireplaceflag']] = True
property_2016.fireplaceflag.fillna(0,inplace = True)


# In[19]:


property_2016.garagecarcnt.fillna(0,inplace = True)
property_2016.garagetotalsqft.fillna(0,inplace = True)


# In[20]:


property_2016.taxdelinquencyflag.fillna(0,inplace = True)


# In[21]:


property_2016.taxdelinquencyflag.replace(to_replace = 'Y', value = 1,inplace = True)


# In[22]:


property_2016.drop('taxdelinquencyyear', axis=1, inplace=True)


# In[23]:


property_2016.drop('storytypeid', axis=1, inplace=True)


# In[24]:


property_2016.basementsqft.fillna(0,inplace = True)


# In[25]:


property_2016.yardbuildingsqft26.fillna(0,inplace = True)


# In[26]:


property_2016.drop('architecturalstyletypeid', axis=1, inplace=True)


# In[27]:


property_2016.drop('typeconstructiontypeid', axis=1, inplace=True)
property_2016.drop('finishedsquarefeet13', axis=1, inplace=True)


# In[28]:


property_2016.drop('buildingclasstypeid', axis=1, inplace=True)


# In[29]:


property_2016.decktypeid.fillna(0,inplace = True)


# In[30]:


property_2016.decktypeid.replace(to_replace = 66.0, value = 1,inplace = True)


# In[31]:


property_2016.drop('finishedsquarefeet6', axis=1, inplace=True)


# In[32]:


property_2016.drop('finishedsquarefeet12', axis=1, inplace=True)


# In[33]:


property_2016.drop('finishedfloor1squarefeet', axis=1, inplace=True)


# In[34]:


property_2016['calculatedfinishedsquarefeet'].fillna((property_2016['calculatedfinishedsquarefeet'].mean()), inplace=True)


# In[35]:


property_2016.loc[property_2016['finishedsquarefeet15'].isnull(),'finishedsquarefeet15'] = property_2016['calculatedfinishedsquarefeet']


# In[36]:


property_2016.numberofstories.fillna(1,inplace = True)


# In[37]:


def property():
    newDataSet1 = newDataSet.withColumn("transac", lit(1))
    ## impute missing values and transfer categorical features into numeric features
    string_title = map(str.strip, ['aircon',  'quality',  'heating',  'zoning_landuse_county', 'zoning_landuse', 'zoning_property', 'region_city', 'region_neighbor', 'region_zip', 'build_year', 'no_fip'])

    none_fill = dict(zip(string_title, ['unknown']*len(string_title)))

    newDataSet1 = newDataSet1.na.fill(none_fill)

    att = list()
    for i,title in enumerate(string_title):
      att = att + newDataSet1.select(title).distinct().rdd.flatMap(lambda x: x).collect()

    atts = list(set(att))
    n_atts = map(str,range(len(atts)))

    for title in string_title:
      newDataSet1 = newDataSet1.na.replace(atts, n_atts, title)

    num_title = ['id_parcel', 'num_bathroom', 'num_bedroom', 'area_total_calc', 'num_garage', 'area_garage', 'latitude', 'longitude', 'area_lot', 'num_room', 'num_unit', 'num_story', 'tax_building', 'tax_year', 'tax_land', 'tax_property']

    for title in num_title:
      newDataSet1 = newDataSet1.withColumn(title, newDataSet1[title].cast(DoubleType()))

    num_fill = dict()

    for title in num_title:
      average = newDataSet1.groupBy().avg(title).collect()[0][0]
      num_fill[title] = average

    newDataSet1 = newDataSet1.na.fill(num_fill)

    for title in string_title:
      newDataSet1 = newDataSet1.withColumn(title, newDataSet1[title].cast(DoubleType()))  
    newDataSet1.rdd.map(list)
    new_labeledPoints = newDataSet_rdd.map(lambda x : LabeledPoint(0, Vectors.dense(x[0:])))
    predictions = model.predict(new_labeledPoints.map(lambda x: x.features))

    ## product an output
    id_predictions = newDataSet_rdd.map(lambda l: l[0]).zip(predictions)


# In[38]:


property_2016.loc[property_2016['numberofstories'] == 1.0,'finishedsquarefeet50'] = property_2016['calculatedfinishedsquarefeet']
property_2016['finishedsquarefeet50'].fillna((property_2016['finishedsquarefeet50'].mean()), inplace=True)


# In[39]:


property_2016.yardbuildingsqft17.fillna(0,inplace = True)


# In[40]:


bathrooms = property_2016[property_2016['fullbathcnt'].notnull() & property_2016['threequarterbathnbr'].notnull() & property_2016['calculatedbathnbr'].notnull()]


# In[41]:


property_2016.drop('threequarterbathnbr', axis=1, inplace=True)


# In[42]:


property_2016.drop('fullbathcnt', axis=1, inplace=True)


# In[43]:


bathroommode = property_2016['calculatedbathnbr'].value_counts().argmax()
property_2016['calculatedbathnbr'] = property_2016['calculatedbathnbr'].fillna(bathroommode)


# In[44]:


property_2016.airconditioningtypeid.fillna(5,inplace = True)


# In[45]:


property_2016.drop('regionidneighborhood', axis=1, inplace=True)


# In[46]:


property_2016.heatingorsystemtypeid.fillna(13,inplace = True)


# In[47]:


def xgb():
    clf = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test.toarray())
    accScore = metrics.accuracy_score(y_test,preds)

    precision = metrics.precision_score(y_test,preds,average=None,labels=labels)
    recall = metrics.recall_score(y_test,preds,average=None,labels=labels)
    f1Score = metrics.f1_score(y_test,preds,average=None,labels=labels)

    print(clf)
    print(accScore,"\n")
    
    dtrain = xgb.DMatrix(x_train, label=labels)
    watchlist = [(dtest, 'eval'), (dtrain, 'train')]
    bst = xgb.train(param, dtrain, num_round, watchlist)
    csc = scipy.sparse.csc_matrix((dat, (row, col)))
    dtrain = xgb.DMatrix(csc, label=labels)
    watchlist = [(dtest, 'eval'), (dtrain, 'train')]
    bst = xgb.train(param, dtrain, num_round, watchlist)
    npymat = csr.todense()
    dtrain = xgb.DMatrix(npymat, label=labels)
    watchlist = [(dtest, 'eval'), (dtrain, 'train')]
    bst = xgb.train(param, dtrain, num_round, watchlist)

  


# In[48]:


buildingqual = property_2016['buildingqualitytypeid'].value_counts().argmax()

property_2016['buildingqualitytypeid'] = property_2016['buildingqualitytypeid'].fillna(buildingqual)


# In[49]:


property_2016.unitcnt.fillna(1,inplace = True)


# In[50]:


propertyzoningdesc = property_2016['propertyzoningdesc'].value_counts().argmax()
property_2016['propertyzoningdesc'] = property_2016['propertyzoningdesc'].fillna(propertyzoningdesc)


# In[51]:


property_2016['lotsizesquarefeet'].fillna((property_2016['lotsizesquarefeet'].mean()), inplace=True)


# In[52]:


property_2016.drop('censustractandblock', axis=1, inplace=True)


# In[53]:


taxdata = property_2016[property_2016['landtaxvaluedollarcnt'].notnull() & property_2016['structuretaxvaluedollarcnt'].notnull() & property_2016['taxvaluedollarcnt'].notnull() & property_2016['taxamount'].notnull()]


# In[54]:


def dt():
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    accScore = metrics.accuracy_score(y_test,preds)
    labels = range(32)

    precision = metrics.precision_score(y_test,preds,average=None,labels=labels)
    recall = metrics.recall_score(y_test,preds,average=None,labels=labels)
    f1Score = metrics.f1_score(y_test,preds,average=None,labels=labels)

    print(clf)
    print("\nOverall Acurracy: ",accScore,"\n")

    preds = clf.predict_proba(X_test)


# In[55]:


property_2016.landtaxvaluedollarcnt.fillna(0,inplace = True)

property_2016.structuretaxvaluedollarcnt.fillna(0,inplace = True)

property_2016['taxvaluedollarcnt'].fillna((property_2016['taxvaluedollarcnt'].mean()), inplace=True)


# In[56]:


from sklearn.metrics import log_loss
def randomForest():
    X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, test_size=0.20)

    clf1 = RandomForestClassifier(n_estimators=1000)
    clf1.fit(X_train, y_train)
    y_val_pred = clf1.predict_proba(X_val)
   


# In[57]:


def visulaize():
    ulimit = np.percentile(train.price.values, 99)
    train['logerror'].ix[train['logerror']>ulimit] = ulimit


    plt.figure(figsize=(8, 10))
    plt.scatter(range(train.shape[0]), train["logerror"].values,color='purple')
    
    


# In[58]:


property_2016['taxpercentage'] = property_2016['taxamount'] / property_2016['taxvaluedollarcnt']


# In[59]:


property_2016['taxpercentage'].fillna((property_2016['taxpercentage'].mean()), inplace=True)


# In[60]:


property_2016.drop('taxamount', axis=1, inplace=True)


# In[61]:


def catboost():
    
    model = CatBoostClassifier(
    custom_loss=['Accuracy'],
    random_seed=42,
    logging_level='Silent'
    )
    model.fit(
    X_train, y_train,
    cat_features=categorical_features_indices,
    eval_set=(X_validation, y_validation),
    logging_level='Verbose',  # you can uncomment this for text output
    plot=True);
    cv_data = cv(
    Pool(X, y, cat_features=categorical_features_indices),
    model.get_params(),
    plot=True)
    
    predictions = model.predict(X_test)
    predictions_probs = model.predict_proba(X_test)
    print(predictions[:10])
    print(predictions_probs[:10])
    model_without_seed = CatBoostClassifier(iterations=10, logging_level='Silent')
    model_without_seed.fit(X, y, cat_features=categorical_features_indices)
    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=validate_pool)

    best_model_params = params.copy()
    best_model_params.update({
        'use_best_model': True
    })
    best_model = CatBoostClassifier(**best_model_params)
    best_model.fit(train_pool, eval_set=validate_pool);


# In[62]:


property_2016.drop('regionidcity', axis=1, inplace=True)


# In[63]:


yearbuilt = property_2016['yearbuilt'].value_counts().argmax()
property_2016['yearbuilt'] = property_2016['yearbuilt'].fillna(yearbuilt)


# In[64]:


fips = property_2016['fips'].value_counts().argmax()
property_2016['fips'] = property_2016['fips'].fillna(fips)

propertylandusetypeid = property_2016['propertylandusetypeid'].value_counts().argmax()
property_2016['propertylandusetypeid'] = property_2016['propertylandusetypeid'].fillna(propertylandusetypeid)

property_2016.drop('regionidcounty', axis=1, inplace=True)

latitude = property_2016['latitude'].value_counts().argmax()
property_2016['latitude'] = property_2016['latitude'].fillna(latitude)

longitude = property_2016['longitude'].value_counts().argmax()
property_2016['longitude'] = property_2016['longitude'].fillna(longitude)

rawcensustractandblock = property_2016['rawcensustractandblock'].value_counts().argmax()
property_2016['rawcensustractandblock'] = property_2016['rawcensustractandblock'].fillna(rawcensustractandblock)

assessmentyear = property_2016['assessmentyear'].value_counts().argmax()
property_2016['assessmentyear'] = property_2016['assessmentyear'].fillna(assessmentyear)

bedroomcnt = property_2016['bedroomcnt'].value_counts().argmax()
property_2016['bedroomcnt'] = property_2016['bedroomcnt'].fillna(bedroomcnt)

bathroomcnt = property_2016['bathroomcnt'].value_counts().argmax()
property_2016['bathroomcnt'] = property_2016['bathroomcnt'].fillna(bathroomcnt)

roomcnt = property_2016['roomcnt'].value_counts().argmax()
property_2016['roomcnt'] = property_2016['roomcnt'].fillna(roomcnt)

propertycountylandusecode = property_2016['propertycountylandusecode'].value_counts().argmax()
property_2016['propertycountylandusecode'] = property_2016['propertycountylandusecode'].fillna(propertycountylandusecode)

regionidzip = property_2016['regionidzip'].value_counts().argmax()
property_2016['regionidzip'] = property_2016['regionidzip'].fillna(regionidzip)


# In[65]:


from sklearn.ensemble import AdaBoostClassifier
def adaBoost():
    clf = AdaBoostClassifier()
    scores1 = cross_val_score(clf, x_train, y_train, cv=100)
    scores1.max()


# In[66]:


trainWithMonths = trainData


# In[67]:



trainWithMonths['sale_month'] = trainWithMonths['transactiondate'].apply(lambda x: (x.to_pydatetime()).month)
trainWithMonths['sale_day'] = trainWithMonths['transactiondate'].apply(lambda x: (x.to_pydatetime()).day)
trainWithMonths['sale_year'] = trainWithMonths['transactiondate'].apply(lambda x: (x.to_pydatetime()).year)


# In[68]:


property_2016['taxpersqft'] = property_2016['taxvaluedollarcnt'] / property_2016['calculatedfinishedsquarefeet']
property_2016['bathpersqft'] = property_2016['bathroomcnt'] / property_2016['calculatedfinishedsquarefeet']
property_2016['roompersqft'] = property_2016['roomcnt'] / property_2016['calculatedfinishedsquarefeet']
property_2016['bedroompersqft'] = property_2016['bedroomcnt'] / property_2016['calculatedfinishedsquarefeet']


# In[69]:


def label():
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size = 0.15, random_state = 0)

    ylabel=LabelEncoder()
    ylabel.fit(y_train)
    y_train= ylabel.transform(y_train)
    y_test= ylabel.fit_transform(y_test)

    X_train["Mean Temperature"].fillna(X_train["Mean Temperature"].mean(), inplace=True)
    X_test["Mean Temperature"].fillna(X_test["Mean Temperature"].mean(), inplace=True)

    rand_forest = RandomForestClassifier()
    rand_forest.fit(X_train, y_train)
    y_predrf = rand_forest.predict(X_test)

    print ("Accuracy is ", accuracy_score(y_test,y_predrf)*100)

    fileObject = open('models/cw_r_forest','wb')
    pickle.dump(rand_forest, fileObject) 
    fileObject.close()


# In[70]:


merged_data = trainData.merge(property_2016,on='parcelid',how='left')


# In[71]:


trainData.drop(['sale_month','sale_year'],axis=1,inplace=True)
trainData.drop(['sale_day'],axis=1,inplace=True)


# In[72]:


import gc


# In[73]:


for c, dtype in zip(merged_data.columns, merged_data.dtypes):	
    if dtype == np.float64 or dtype == np.int64:		
        merged_data[c] = merged_data[c].astype(np.float32)

df_train = merged_data


# In[74]:


merged_data.drop(['sale_month'],axis=1,inplace=True)
merged_data.drop(['sale_year','sale_day'],axis=1,inplace=True)


# In[75]:


x_train = df_train.drop(['parcelid'], axis=1)
x_train = x_train.drop(['logerror' ], axis=1)
x_train = x_train.drop(['transactiondate'], axis=1)
x_train = x_train.drop(['propertyzoningdesc'], axis=1)
x_train = x_train.drop(['propertycountylandusecode'], axis=1)

y_train = df_train['logerror'].values


# In[76]:


train_columns = x_train.columns
temp = x_train.dtypes[x_train.dtypes == object].index.values
for c in temp:
    x_train[c] = (x_train[c] == True)
    
split = 90000
x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]
x_train = x_train.values.astype(np.float32, copy=False)
x_valid = x_valid.values.astype(np.float32, copy=False)

d_train = lgb.Dataset(x_train, label=y_train)
d_valid = lgb.Dataset(x_valid, label=y_valid)

params = {}
params['learning_rate'] = 0.002
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'mae'
params['sub_feature'] = 0.5
params['num_leaves'] = 60
params['min_data'] = 500
params['min_hessian'] = 1


# In[77]:


watchlist = [d_valid]
clf = lgb.train(params, d_train, 500, watchlist)

sample = pd.read_csv('sample_submission.csv')
sample['parcelid'] = sample['ParcelId']
df_test = sample.merge(property_2016, on='parcelid', how='left')

x_test = df_test[train_columns]

temp2 = x_test.dtypes[x_test.dtypes == object].index.values
for c in temp2:
    x_test[c] = (x_test[c] == True)
x_test = x_test.values.astype(np.float32, copy=False)

clf.reset_parameter({"num_threads":1})
y_test = clf.predict(x_test)


sub = pd.read_csv('sample_submission.csv')
for c in sub.columns[sub.columns != 'ParcelId']:
    sub[c] = y_test

sub.to_csv('lgb_results_final.csv', index=False, float_format='%.4f')
print("Done")

