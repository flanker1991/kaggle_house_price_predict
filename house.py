import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression,LinearRegression,Ridge,ElasticNet,RANSACRegressor,Lasso
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.svm import SVC,SVR,LinearSVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier,VotingClassifier,VotingRegressor,AdaBoostClassifier,RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor
from xgboost import XGBClassifier,XGBRegressor
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV,cross_val_score,cross_val_predict,StratifiedKFold,learning_curve,validation_curve,KFold
from sklearn.metrics import accuracy_score,roc_auc_score,recall_score,confusion_matrix,silhouette_score,r2_score
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from scipy.special import boxcox1p

#stacking建模
def get_stack_data(clfs, x_train, y_train, x_test):
    stack_train=np.zeros((len(x_train),len(clfs)))
    stack_test=np.zeros((len(x_test),len(clfs)))
    for j, clf in enumerate(clfs):
        oof_test= np.zeros((len(x_test),5))
        skf=KFold(n_splits =5,shuffle=False)
        for i,(train_s_index,test_s_index) in enumerate(skf.split(x_train)):
            x_train_s=x_train.loc[train_s_index]
            y_train_s=y_train.loc[train_s_index]
            x_test_s=x_train.loc[test_s_index]
            clf.fit(x_train_s,y_train_s)
            stack_train[test_s_index,j]=clf.predict(x_test_s)
            oof_test[:,i]=clf.predict(x_test)
        stack_test[:,j]=oof_test.mean(axis=1)
    return stack_train,stack_test

#查看缺失值情况
def missing_values(data):
    data_na=pd.DataFrame(data.isnull().sum(), columns=['NAN_num'])
    data_na['NAN_rate'] = data.isnull().mean()
    data_na['dtype'] = data.dtypes 
    data_na = data_na[data_na['NAN_num']>0].sort_values(by='NAN_num',ascending=False)
    return data_na

#特征重要度排序
def feature_importances(clf,x,y):
    clf.fit(x,y)
    importance=clf.feature_importances_
    index=importance.argsort()[::-1]
    plt.figure(figsize=(16,12))
    plt.bar(range(x.shape[1]),importance[index])
    plt.xticks(range(x.shape[1]),x.columns[index],fontsize=25,rotation=90)
    plt.yticks(fontsize=25)
    for i in range(x.shape[1]):
        print(i,x.columns[index[i]],importance[index[i]])

#选择前k个特征
def feature_select(clfs,x,y,k):
    features_top_k=pd.Series()
    for i, clf in enumerate(clfs):
        clf.fit(x,y)
        importance=clf.feature_importances_
        feature_sorted= pd.DataFrame({'feature': list(x), 'importance': importance}).sort_values('importance', ascending=False)
        features = feature_sorted.head(k)['feature']
        features_top_k=pd.concat([features_top_k,features],ignore_index=True).drop_duplicates()
    return features_top_k

#残差图
def plot_residual(x,y,clf,title):
    y_pred=cross_val_predict(clf,x,y,cv=5)
    plt.figure(figsize=(16,12))
    plt.title(title,fontsize=25)
    plt.scatter(y_pred,y_pred-y,c="lightgreen",marker="s")
    plt.xlabel("prediction",fontsize=25)
    plt.xticks(fontsize=25) 
    plt.ylabel("residual",fontsize=25)
    plt.yticks(fontsize=25) 
    plt.hlines(y=0,lw=2,xmin=y.min(),xmax=y.max(),color="red")
    plt.show()

#计算回归器相关度与散点图矩阵
def pred_matrix_corr(x,y,clfs):
    names=[]
    pred_matrix=np.zeros((len(x),len(clfs)))
    pred_matrix=pd.DataFrame(pred_matrix)
    for j,(name,clf) in enumerate(clfs.items()):
        y_pred=cross_val_predict(clf,x,y,cv=5)
        pred_matrix.iloc[:,j]=y_pred
        names.append(name)
    pred_matrix.columns=names
    plt.figure(figsize=(16,16))
    sns.heatmap(pred_matrix.corr(),linewidths=0.2,vmax=1.0,square=True,linecolor='white', annot=True)
    sns.pairplot(pred_matrix,size=6)

#学习曲线
def plot_learning_curve_r(x,y,clf,title):
    plt.figure(figsize=(16,12))
    plt.title(title,fontsize=25)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        clf, x, y, cv=5, n_jobs=-1,scoring='r2', train_sizes=np.linspace(.1,1.0,10))
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    return plt

# 定义交叉验证,用均方根误差来评价模型的拟合程度
def rmse_cv(clf,x, y):
    rmse = np.sqrt(-cross_val_score(clf,x, y, scoring = 'neg_mean_squared_error', cv=5))
    return rmse.mean()

#回归算法网格搜索调优
def gscvr(x,y,clf,param_grid):
        gs=GridSearchCV(clf,param_grid=param_grid,cv=5,scoring='neg_mean_squared_error',n_jobs=-1)
        gs.fit(x,y)
        print(gs.best_score_,gs.best_params_)
        return gs.best_estimator_

#找出离群点
def find_outliers(x,y,clf,z0):
    clf.fit(x,y)
    y_pred =clf.predict(x)
    resid = y-y_pred
    mean_resid = resid.mean()
    std_resid = resid.std()
    z = (resid - mean_resid) / std_resid
    outliers = z[abs(z) > z0].index
    plt.figure(figsize=(12,10))
    plt.scatter(y_train, y_pred)
    plt.scatter(y_train.iloc[outliers], y_pred[outliers])
    plt.plot(range(10, 15), range(10, 15), color="red")
    return outliers

#根据中位数填补类别缺失而数值未缺失的类别
def fill_class_by_median(data,column_null,column_notnull,fill_data):
    median=data.groupby(column_null)[column_notnull].median()
    for i in fill_data.index:
        v = data.loc[i,column_notnull]
        data.loc[i,column_null]=abs(v-median).astype('float64').argmin()
        
#根据类别使用中位数填补数值缺失
def fill_numerical_by_median(data,column_null,column_notnull,fill_data):
    median=data.groupby(column_notnull)[column_null].median()
    for i in fill_data.index:
        v = fill_data.loc[i,column_notnull]
        data.loc[i,column_null]=median[v]

#根据相关类别对应的众数填补类别缺失
def fill_class_by_mode(data,column_null,column_notnull,fill_data):
    for i in fill_data.index:
        v = fill_data.loc[i,column_notnull]
        mode=data[data[column_notnull]==v][column_null].mode()[0]
        data.loc[i,column_null]=mode

#一、数据分析与可视化
#1.1数据概况
#导入数据
train=pd.read_csv('D:/data_project/house/train.csv')
test=pd.read_csv('D:/data_project/house/test.csv')
submission=pd.read_csv('D:/data_project/house/sample_submission.csv')
pd.set_option("display.max_columns",500)
pd.set_option("display.max_rows",300)
#合并测试集与训练集
total_initial=pd.concat([train,test],ignore_index=True,sort=False)
total=total_initial.copy()
total=total.drop('Id',axis=1)
#列名修改
columns=total.columns.values.tolist()
new_columns=[]
for i, column in enumerate(columns):
    new_columns.append(column.lower())
total.columns= new_columns
total.rename(columns={'saleprice':'price'},inplace=True)


#总览数据
total.head()
total.info()
total.describe()

#相关系数
cor=total.corr()
cor_sorted=cor['price'].sort_values(ascending=False)
#找出与price相关度前k个变量并做相关图
columns=abs(cor['price']).nlargest(10).index
plt.figure(figsize=(25,25))
sns.set(font_scale=2) 
sns.heatmap(total[columns].corr(),linewidths=0.2,vmax=1.0,square=True,linecolor='white', annot=True)

#房价分布
train=total[:1460]
plt.figure(figsize=(12,8))
sns.distplot(train.price,label='skewness:{:.2f}'.format(train.price.skew())).legend()


#1.2 缺失值处理
#1.2.1 查看缺失值情况
missing_values(total)

#1.2.2poolqc,poolarea
total.poolqc.value_counts()
total.poolarea.value_counts().head()
#poolqc缺失但poolarea不等于0
pool_1=total[(total.poolqc.isnull()) & (total.poolarea!=0)][['poolqc','poolarea']]
fill_class_by_median(total,'poolqc','poolarea',pool_1)
#poolqc不缺失但poolarea等于0
pool_2=total[(total.poolqc.notnull()) & (total.poolarea==0)][['poolqc','poolarea']]
#其余poolqc填充为None
total['poolqc']=total['poolqc'].fillna('None')

#1.2.3 miscfeature,miscval
total.miscfeature.value_counts()
total.miscval.value_counts().head()
#miscfeature缺失但miscval不等于0
mis_1=total[(total.miscfeature.isnull()) & (total.miscval!=0)][['miscfeature','miscval']]
fill_class_by_median(total,'miscfeature','miscval',mis_1)
#miscfeature未缺失但miscval等于0
mis_2=total[(total.miscfeature.notnull()) & (total.miscval==0)][['miscfeature','miscval']]
fill_numerical_by_median(total,'miscval','miscfeature',mis_2)
#其余miscfeature填充为None
total['miscfeature']=total['miscfeature'].fillna('None')

#1.2.4 alley,fence
total['alley']=total['alley'].fillna('None')
total['fence']=total['fence'].fillna('None')

#1.2.5 fireplacequ,fireplaces
total.fireplacequ.value_counts()
total.fireplaces.value_counts().head()
#fireplacequ缺失但fireplaces不等于0
fire_1=total[(total.fireplacequ.isnull()) & (total.fireplaces!=0)][['fireplacequ','fireplaces']]
#fireplacequ未缺失但fireplaces等于0
mis_2=total[(total.fireplacequ.notnull()) & (total.fireplaces==0)][['fireplacequ','fireplaces']]
#fireplacequ填充为None
total['fireplacequ']=total['fireplacequ'].fillna('None')

#1.2.6 garage
# 找出所有garage特征的缺失情况
columns=pd.Series(total.columns)
garage=columns[columns.str.contains('garage')].values
missing_values(total).loc[garage,:]

#garagecond,garagefinish,garagequal缺失而garagetype未缺失的行
garage_1=total[(total.garagequal.isnull()) & (total.garagetype.notnull())][garage]
fill_class_by_mode(total,'garagecond','garagetype',garage_1)
fill_class_by_mode(total,'garagefinish','garagetype',garage_1)
fill_class_by_mode(total,'garagequal','garagetype',garage_1)
#garagefinish,garagequal,garagecond,garagetype其余缺失值填补为None
columns=['garagefinish','garagequal','garagecond','garagetype']
imp=SimpleImputer(strategy='constant',fill_value='None')
total.loc[:,columns]=imp.fit_transform(total[columns])

#garagecars,garagearea以garagetype对应的中位数填补
garage_2=total[total.garagecars.isnull()][garage]
fill_numerical_by_median(total,'garagecars','garagetype',garage_2)
fill_numerical_by_median(total,'garagearea','garagetype',garage_2)

#1.2.7 bsmt
# 找出所有bsmt特征的缺失情况
columns=pd.Series(total.columns)
bsmt=columns[columns.str.contains('bsmt')].values
missing_values(total).loc[bsmt,:]
#计算bsmt全部都缺失的行数
total[total.bsmtcond.isnull() & total.bsmtexposure.isnull()
& total.bsmtqual.isnull() & total.bsmtfintype1.isnull()
& total.bsmtfintype2.isnull()].shape[0]

#bsmtcond缺失而bsmtfintype1未缺失的
bsmt_1=total[(total.bsmtcond.isnull()) & (total.bsmtfintype1.notnull())][bsmt]
fill_class_by_mode(total,'bsmtcond','bsmtqual',bsmt_1)
#bsmtqual缺失而bsmtfintype1未缺失的
bsmt_2=total[(total.bsmtqual.isnull()) & (total.bsmtfintype1.notnull())][bsmt]
fill_class_by_mode(total,'bsmtqual','bsmtcond',bsmt_2)
#bsmtexposure缺失而bsmtfintype1未缺失的
bsmt_3=total[(total.bsmtexposure.isnull()) & (total.bsmtfintype1.notnull())][bsmt]
fill_class_by_mode(total,'bsmtexposure','bsmtqual',bsmt_3)
#bsmtfintype2缺失而bsmtfintype1未缺失的
bsmt_4=total[(total.bsmtfintype2.isnull()) & (total.bsmtfintype1.notnull())][bsmt]
fill_class_by_median(total,'bsmtfintype2','bsmtfinsf2',bsmt_4)

#'bsmtcond','bsmtqual','bsmtexposure','bsmtfintype2','bsmtfintype1'其余缺失值填补为0
columns=['bsmtcond','bsmtqual','bsmtexposure','bsmtfintype2','bsmtfintype1']
imp=SimpleImputer(strategy='constant',fill_value='None')
total.loc[:,columns]=imp.fit_transform(total[columns])

#totalbsmtsf,bsmtfinsf1,bsmtfinsf2,bsmtunfsf,bsmtfullbath,bsmthalfbath
total[total.bsmtfullbath.isnull()][bsmt]
#全部填补为0
columns=['totalbsmtsf','bsmtfinsf1','bsmtfinsf2','bsmtunfsf','bsmtfullbath','bsmthalfbath']
imp=SimpleImputer(strategy='constant',fill_value=0)
total.loc[:,columns]=imp.fit_transform(total[columns])

#1.2.8 masvnr,exter
# 找出所有特征的缺失情况
columns=pd.Series(total.columns)
masvnr=columns[columns.str.contains('masvnr') | columns.str.contains('exter')].values
missing_values(total).loc[masvnr,:]
#masvnrtype缺失而masvnrarea 未缺失的
masvnr_1=total[total.masvnrtype.isnull() & total.masvnrarea.notnull()][masvnr]
total.masvnrtype.value_counts()
total.masvnrarea.value_counts().head()
sns.boxplot('masvnrtype','masvnrarea',data=total)
total.loc[masvnr_1.index,'masvnrtype']='BrkFace'

#masvnrtype为none但masvnrarea 不为0
masvnr_2=total[total.masvnrtype=='None'][total.masvnrarea!=0][masvnr]
total[total.masvnrarea==1].shape[0]
total.masvnrarea.hist(bins=30)
total.masvnrarea.describe()
#masvnrarea=1的改为0,修改异常值
total.loc[total[total.masvnrarea==1].index,'masvnrarea']=0

#再次计算masvnrtype为none但masvnrarea 不为0
masvnr_3=total[total.masvnrtype=='None'][total.masvnrarea!=0][masvnr]
total.loc[masvnr_3.index,'masvnrtype']='BrkFace'

#masvnrtype不为none但masvnrarea 为0
masvnr_4=total[total.masvnrtype!='None'][total.masvnrarea==0][masvnr]
fill_numerical_by_median(total,'masvnrarea','masvnrtype',masvnr_4)
total.loc[masvnr_4.index,masvnr]

#剩余缺失的masvnrtype填补为none,masvnrarea填补为0
total['masvnrtype']=total['masvnrtype'].fillna('None')
total['masvnrarea']=total['masvnrarea'].fillna(0)

#exterior1st,exterior2nd
masvnr_4=total[total.exterior1st.isnull()][masvnr]
total.exterior1st.value_counts()
fill_class_by_mode(total,'exterior1st','exterqual',masvnr_4)
fill_class_by_mode(total,'exterior2nd','exterqual',masvnr_4)

#1.2.9 kitchenqual,kitchenabvgr
total.kitchenqual.value_counts()
total.kitchenabvgr.value_counts().head()
#kitchenqual缺失的
kitchen_1=total[total.kitchenqual.isnull()][['kitchenqual','kitchenabvgr']]
fill_class_by_mode(total,'kitchenqual','kitchenabvgr',kitchen_1)
#kitchenabvgr等于0的
kitchen_2=total[total.kitchenabvgr==0][['kitchenqual','kitchenabvgr']]
fill_class_by_mode(total,'kitchenabvgr','kitchenqual',kitchen_2)

#1.2.10 saletype
total.saletype.value_counts()
total.salecondition.value_counts()
sale_1=total[total.saletype.isnull()][['saletype','salecondition']]
fill_class_by_mode(total,'saletype','salecondition',sale_1)

#1.2.11 'electrical','functional','utilities','mszoning'
columns=['electrical','functional','utilities','mszoning']
imp=SimpleImputer(strategy='most_frequent')
total.loc[:,columns]=imp.fit_transform(total[columns])


#1.4 数据变换
'''
#1.4.1有序类别映射到数字
dict_model1={'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'None':0}
columns=['fireplacequ','bsmtcond','bsmtqual','exterqual','extercond',
         'garagequal','garagecond','heatingqc','kitchenqual','poolqc']
for i in columns:
    total[i]=total[i].map(dict_model1)
    
total['street']=total.street.map({'Grvl':1,'Pave':0})
total['alley']=total.alley.map({'Grvl':2,'Pave':1,'None':0})
total['lotshape']=total.lotshape.map({'Reg':4,'IR1':3,'IR2':2,'IR3':1})
total['landcontour']=total.landcontour.map({'Lvl':4,'Bnk':3,'HLS':2,'Low':1})
total['utilities']=total.utilities.map({'AllPub':4,'NoSewr':3,'NoSeWa':2,'ELO':1})
total['landslope']=total.landslope.map({'Gtl':3,'Mod':2,'Sev':1})
total['bsmtexposure']=total.bsmtexposure.map({'Gd':4,'Av':3,'Mn':2,'No':1,'None':0})
total['bsmtfintype1']=total.bsmtfintype1.map({'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'None':0})
total['bsmtfintype2']=total.bsmtfintype2.map({'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'None':0})
total['centralair']=total.centralair.map({'N':0,'Y':1})
total['electrical']=total.electrical.map({'SBrkr':5,'FuseA':4,'FuseF':3,'FuseP':2,'Mix':1})
total['functional']=total.functional.map({'Typ':8,'Min1':7,'Min2':6,'Mod':5,'Maj1':4,'Maj2':3,'Sev':2,'Sal':1})
total['garagetype']=total.garagetype.map({'2Types':6,'Attchd':5,'Basment':4,'BuiltIn':3,'CarPort':2,'Detchd':1,'None':0})
total['garagefinish']=total.garagefinish.map({'Fin':3,'RFn':2,'Unf':1,'None':0})
total['fence']=total.fence.map({'GdPrv':4,'MnPrv':3,'GdWo':2,'MnWw':1,'None':0})
total['paveddrive']=total.paveddrive.map({'Y':3,'P':2,'N':1})
total['housestyle']=total.housestyle.map({'1Story':1,'1.5Fin':2,'1.5Unf':3,'2Story':4,'2.5Fin':5,'2.5Unf':6,'SFoyer':7,'SLvl':8})
total['masvnrtype']=total.masvnrtype.map({'BrkCmn':1,'BrkFace':1,'CBlock':1,'Stone': 1,'None': 0})

total['condition1']=total['condition1'].map(lambda x : 1 if x=='Norm' else 0)
total['condition2']=total['condition2'].map(lambda x : 1 if x=='Norm' else 0)
total['roofstyle']=total['roofstyle'].map(lambda x : 0 if x=='Gable' else 1)
'''
#1.4.2时间特征
#提取包含年份的列
columns=pd.Series(total.columns)
time=columns[columns.str.contains('year') | columns.str.contains('yr')].values
#创建houseage,houseremodelage,调整错误值
total['houseage']=total.yrsold-total.yearbuilt
total['hrage']=total.yrsold-total.yearremodadd
total.hrage.value_counts()
time_1=total[total.hrage<0][time]
total.groupby('yrsold').yrsold.count()
total.loc[time_1.index,'yrsold']=2009
total['houseage']=total.yrsold-total.yearbuilt
total['hrage']=total.yrsold-total.yearremodadd
#创建是否remodel
total['remodel']=(total.yearbuilt != total.yearremodadd)*1

#garageyrblt
sns.set(font_scale=2)
sns.pairplot(total[time],size=6)
#异常值处理
total[total.garageyrblt==total['garageyrblt'].max()]
index=total[total.garageyrblt==total['garageyrblt'].max()].index
total.loc[index,'garageyrblt']=2007
#线性回归预测garageyrblt的两个缺失行
data_notnull = total.loc[(total.garageyrblt.notnull())]
data_isnull = total.loc[garage_1.index]
#建模预测并填补
x= data_notnull[['yearbuilt','yearremodadd']]
y= data_notnull[['garageyrblt']]
x_t=data_isnull[['yearbuilt','yearremodadd']]
lr = LinearRegression().fit(x,y)
data_predict=lr.predict(x_t).round()
total.loc[garage_1.index,['garageyrblt']]=data_predict
#分组
year_map = pd.concat(pd.Series('YearGroup' + str(i+1), index=range(1871+i*20,1891+i*20)) for i in range(0, 7))
total['garageyrblt']= total['garageyrblt'].map(year_map)
total['garageyrblt']= total['garageyrblt'].fillna('None')

#1.4.3 部分属性二值化
#total['mssubclass_class']= total.mssubclass.map({20: 1,30: 0,40: 0,45: 0,50: 0,60: 1,70: 0,75: 0,80: 0,85: 0,90: 0,120: 1,150: 0,160: 0,180: 0,190: 0})

#1.4.4数据偏度调整
#目标变量
sns.distplot(total.price.dropna(),label='skewness:{:.2f}'.format(total.price.dropna().skew())).legend()
total['price'].loc[total.price.notnull()] = np.log1p(total['price'].loc[total.price.notnull()]) 
sns.distplot(total.price.dropna(),label='skewness:{:.2f}'.format(total.price.dropna().skew())).legend()
#其余数值特征
# 计算特征偏度
columns = total.dtypes[total.dtypes != "object"].index.tolist()
skewness=total[columns].apply(lambda x: x.dropna().skew()).sort_values(ascending=False)
#对偏度大于0.75的特征进行Box Cox转换
skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))
skewed_features = skewness.index
lam = 0.15
for i in skewed_features:
    total[i] = boxcox1p(total[i], lam)

#1.4.5无序类别独热处理
one_hot_columns=total.dtypes[total.dtypes=='object'].index
total=pd.concat([total,pd.get_dummies(total[one_hot_columns]).astype('float')],axis=1)
total=total.drop(one_hot_columns,axis=1)

#1.4.6 lotfrontage回归填补
#找出与lotfrontage相关度前10的变量
total_cor=total.corr()
lotfrontage_cor_sort=total_cor['lotfrontage'].sort_values(ascending=False)
columns=abs(total_cor['lotfrontage']).nlargest(20).index.tolist()
columns.remove('price')
#二次多项式岭回归填补缺失值
#提取数据
lot=total[columns]
lot_notnull = lot.loc[(lot['lotfrontage'].notnull())]
lot_isnull = lot.loc[(lot['lotfrontage'].isnull())]
#建模
x_train_lot = lot_notnull.drop(['lotfrontage'],axis=1)
y_train_lot = lot_notnull['lotfrontage']
x_test_lot = lot_isnull.drop(['lotfrontage'],axis=1)
pipe_ridge=Pipeline([('scl',StandardScaler()),
                  ('poly',PolynomialFeatures(2)),
                  ('clf',Ridge())])
param_grid={'clf__alpha':np.linspace(.1,1.0,10)}
pipe_ridge=gscvr(x_train_lot,y_train_lot,pipe_ridge,param_grid)
#预测并填补
pipe_ridge.fit(x_train_lot,y_train_lot)
lot_predict=pipe_ridge.predict(x_test_lot)
total.loc[total['lotfrontage'].isnull(),['lotfrontage']]=lot_predict

#1.5创建新特征
#1.5.1 增加总面积 
total['totalsf'] = total['totalbsmtsf'] + total['1stflrsf'] + total['2ndflrsf']


#二、特征选择
#变量相关性排序
total_cor=total.corr()
cor_sorted=total_cor['price'].sort_values(ascending=False)
#取出满足相关性要求的变量
columns=cor_sorted[abs(cor_sorted) >0.02].index
total_s=total[columns]
train_index=total_s[total_s.price.notnull()].index
test_index=total_s[total_s.price.isnull()].index
x_train=total_s.loc[train_index].drop('price',axis=1)
y_train=total_s.loc[train_index,'price']
x_test=total_s.loc[test_index].drop('price',axis=1)

columns.shape
#2.2离群点处理
#建立岭回归和lasso
ridge=Pipeline([('scl',StandardScaler()),
                  ('clf',Ridge())]) 
param_grid={'clf__alpha':[20,50,100,200,500]}
ridge=gscvr(x_train,y_train,ridge,param_grid)
cross_val_score(ridge,x_train,y_train,cv=5,scoring='r2').mean()
lasso=Pipeline([('scl',StandardScaler()),
                  ('clf',Lasso())]) 
param_grid={'clf__alpha':[0.001,0.005,0.01,0.05,0.1]}
lasso=gscvr(x_train,y_train,lasso,param_grid)
cross_val_score(lasso,x_train,y_train,cv=5,scoring='r2').mean()
#找出共同的离群点
outliers1=find_outliers(x_train,y_train,ridge,3)
outliers2=find_outliers(x_train,y_train,lasso,3)
outliers = []
for i in outliers1:
    for j in outliers2:
        if i == j:
            outliers.append(i)
#删除离群点
x_train=x_train.drop(outliers,inplace=False)
y_train=y_train.drop(outliers,inplace=False)
x_train=x_train.reset_index(drop=True)
y_train=y_train.reset_index(drop=True)


#三、模型评估
#3.1 ridge
#调参建模
ridge=Pipeline([#('poly',PolynomialFeatures(2)),
                  ('scl',StandardScaler()),
                  ('clf',Ridge())]) 
param_grid={'clf__alpha':[10,20,50,100,200]}
ridge=gscvr(x_train,y_train,ridge,param_grid)
#交叉验证精度&学习曲线
rmse_cv(ridge,x_train,y_train)
plot_learning_curve_r(x_train,y_train,ridge,'ridge learning_curve')


#3.2 lasso
lasso=Pipeline([#('poly',PolynomialFeatures(2)),
                  ('scl',StandardScaler()),
                  ('clf',Lasso())]) 
param_grid={'clf__alpha':[0.0005,0.001,0.005,0.01,0.05,0.1]}
lasso=gscvr(x_train,y_train,lasso,param_grid)
#交叉验证精度&学习曲线
rmse_cv(lasso,x_train,y_train)
plot_learning_curve_r(x_train,y_train,lasso,'en learning_curve')


#3.3 弹性网
#调参建模
en=Pipeline([#('poly',PolynomialFeatures(2)),
                  ('scl',StandardScaler()),
                  ('clf',ElasticNet())]) 
param_grid={'clf__alpha':[0.001,0.01,0.05,0.1,0.5,1,5,10,50],
            'clf__l1_ratio':np.linspace(0,1.0,15)}
en=gscvr(x_train,y_train,en,param_grid)
#交叉验证精度&学习曲线
rmse_cv(en,x_train,y_train)
plot_learning_curve_r(x_train,y_train,en,'en learning_curve')


#3.4 SVM
#调参建模
svr=Pipeline([('scl',StandardScaler()),
                  ('clf',SVR(kernel='rbf'))]) 
param_grid={'clf__gamma':[0.0001,0.0002,0.001],
            'clf__epsilon':[0.005,0.01,0.03,0.05],
            'clf__C':[10,20,50]}
svr=gscvr(x_train,y_train,svr,param_grid)
#交叉验证精度&学习曲线
rmse_cv(svr,x_train,y_train)
plot_learning_curve_r(x_train,y_train,svr,'svr learning_curve')


#3.5随机森林
#调参建模
rfr=RandomForestRegressor(n_estimators=300,n_jobs=-1)
param_grid={#'min_samples_leaf':[3,5,10],
            #'max_depth':[20,30,40,55],
            'max_features':[20,40,60,80]}
rfr=gscvr(x_train,y_train,rfr,param_grid)
#交叉验证精度&学习曲线
rmse_cv(rfr,x_train,y_train)
plot_learning_curve_r(x_train,y_train,rfr,'rfr learning_curve')

#3.6极端随机树
#调参建模
etr=ExtraTreesRegressor(n_estimators=500,n_jobs=-1)
param_grid={#'min_samples_leaf':[3,5,10],
            'max_depth':[3,5,8],
            'max_features':[40,60,80,120,160]}
etr=gscvr(x_train,y_train,etr,param_grid)
#交叉验证精度&学习曲线
rmse_cv(etr,x_train,y_train)
plot_learning_curve_r(x_train,y_train,etr,'etr learning_curve')

 
#3.7xgboost
#调参建模
xgbr=XGBRegressor(colsample_bytree=0.6,
                 learning_rate=0.07,
                 min_child_weight=1.5,
                 n_estimators=400,
                 reg_alpha=0.65,
                 reg_lambda=0.45,
                 #subsample=0.8,
                 gamma=0)
param_grid={#'learning_rate':[0.02,0.05,0.1],
            'max_depth':[3],
            #'gamma':[0,0.01],
            'subsample':[0.4,0.6,0.8]}
xgbr=gscvr(x_train,y_train,xgbr,param_grid)
#交叉验证精度&学习曲线
rmse_cv(xgbr,x_train,y_train)
plot_learning_curve_r(x_train,y_train,xgbr,'xgbr learning_curve')


#3.8gradientboost
#调参建模
gbr=GradientBoostingRegressor(
                 learning_rate=0.07,
                 n_estimators=300)
param_grid={#'learning_rate':np.logspace(-2,0,3),
            'max_depth':[2,3],
            'max_features':[30,40,60,100,150]}
gbr=gscvr(x_train,y_train,gbr,param_grid)
#交叉验证精度&学习曲线
rmse_cv(gbr,x_train,y_train)
plot_learning_curve_r(x_train,y_train,gbr,'gbr learning_curve')


#3.9模型相关性比较
clfs={'en':en,'xgbr':xgbr,'SVR':svr}
pred_matrix_corr(x_train,y_train,clfs)

#3.10 vote
vote= VotingRegressor([('SVR',svr),('en',en),('gbr',gbr)],weights=[1,2,1])
rmse_cv(vote,x_train,y_train)
plot_learning_curve_r(x_train,y_train,vote,'vote learning_curve')


#四、最终预测
#4.1voting
model=vote
model.fit(x_train,y_train)
pred=np.expm1(model.predict(x_test))
submission['SalePrice']=pred
submission.to_csv("d:/data_project/house/result5212.csv", index = False)

#4.2stacking
#选出融合模型
clfs=[svr,en,lasso,xgbr,gbr]
#获取stack_train,stack_test
stack_train,stack_test=get_stack_data(clfs, x_train, y_train, x_test)
#stacking第二步调优
#弹性网络
stack_en=Pipeline([('scl',StandardScaler()),
                  ('clf',ElasticNet())]) 
param_grid={'clf__alpha':[0.001,0.01,0.05,0.1,0.5,1,10,30],
            'clf__l1_ratio':np.linspace(0,1.0,15)}
stack_en=gscvr(stack_train,y_train,stack_en,param_grid)
rmse_cv(stack_en,stack_train,y_train)
#SVR
stack_svr=Pipeline([('scl',StandardScaler()),
                  ('clf',SVR(kernel='rbf'))]) 
param_grid={'clf__gamma':[0.0002,0.0005,0.001],
            'clf__epsilon':[0.005,0.01,0.03,0.05],
            'clf__C':[100,1000,10000,100000]}
stack_svr=gscvr(stack_train,y_train,stack_svr,param_grid)
rmse_cv(stack_svr,stack_train,y_train)
#最终预测
model=stack_en
model.fit(stack_train,y_train)
pred=np.expm1(model.predict(stack_test))
submission['SalePrice']=pred
submission.to_csv("d:/data_project/house/result5213.csv", index = False)





