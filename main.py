# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 16:39:13 2018

@author: xiaoma
"""

import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import StandardScaler     #对于数值特征
from sklearn.preprocessing import OneHotEncoder     #对于id类特征
from scipy import sparse
from sklearn import metrics
import numpy as np
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics.ranking import roc_auc_score
from scipy.sparse.construct import hstack

class solution():
    def __init__(self,evalortest = 'eval'):
        self.lenth_eval = 913850
        self.evalortest = evalortest
        self.num_leaf = 64
        pass
    
    def read_data(self):
        offline_path = "./data/offline.csv"
        online_path = "./data/online.csv"
        test_path = "./data/ccf_offline_stage1_test_revised.csv"
        
        
        train_data = pd.read_table(offline_path,sep = ',')
        print(train_data)
        train_data.drop_duplicates(inplace = True)
        #print(train_data)
        test_data = pd.read_table(test_path,sep = ',',na_values='null')
        test_data = test_data.fillna(-1)
        test_data.drop_duplicates(inplace = True)
        data = train_data.append(test_data)
        #print(data)
        self.lenth_train = len(train_data)
        return data

    def base_feature(self,data):
        data= data.drop(['Date','Discount_rate'],axis = 1)
        
        return data
    '''
    def processdata(self,data):
        #对id类onehot编码
        feature_train = data.iloc[0:self.lenth_eval,:]
        feature_eval = data.iloc[self.lenth_eval:self.lenth_train,:] 
        print(feature_train.shape)
        print(feature_eval.shape)
        
        enc = OneHotEncoder()
        feats = ["Coupon_id"] #, "Merchant_id", "User_id"]  
        for i, feat in enumerate(feats):
            
            #print(feat)
            #print(i)
            data_train = enc.fit_transform(feature_train[feat].values.reshape(-1, 1))  
            data_eval = enc.fit_transform(feature_eval[feat].values.reshape(-1, 1))  
            
            #print(data_1)
            if i == 0:  
                en_data_train,en_data_eval = data_train, data_eval
            else:  
                en_data_train,en_data_eval = sparse.hstack((en_data_train, data_train)),sparse.hstack((en_data_eval, data_eval))
                
        #对数值类标准化
        
        #en_data_train = []
        #en_data_eval = []
        print(en_data_train[0])
        print(en_data_eval.shape)
        ss = StandardScaler() 
        feats = ["Date_received", "Distance"] 
        
        data_train = ss.fit_transform(feature_train[feats].values)  
        data_eval = ss.fit_transform(feature_eval[feats].values)
        #x_test  = ss.fit_transform(df_test[feats].values)  
        #en_data = sparse.hstack((en_data, data_1))
        en_data_train,en_data_eval = sparse.hstack((en_data_train, data_train)),sparse.hstack((en_data_eval, data_eval))
        #print(en_data)
        print(en_data_train.shape)
        print(en_data_eval.shape)
        return en_data_train,en_data_eval      
        
        #return feature_train,feature_eval
    '''    
        
    def feature_filter(self,data):
        basefeature = data[['Coupon_id','Date_received','Distance','Merchant_id','User_id']]
        data_label = data['label']
        #feature = pd.concat()
        if self.evalortest == 'eval':
            feature = basefeature.iloc[0:self.lenth_train,:]
            target = data_label.iloc[0:self.lenth_train]
            
            return feature,target
            #print(feature)
            
        if self.evalortest == 'test':
            feature = basefeature.iloc[self.lenth_train:,:]
            #target = data_label.iloc[0:self.lenth_train]
            
            return feature
    
    def lgbmparam(self):
        param = {  
            'boosting_type': 'gbdt',  
            'objective': 'binary',  
            'metric': {'binary_logloss', 'auc'},  
            'num_leaves': 5,  
            'max_depth': 6,  
            'min_data_in_leaf': 450,  
            'learning_rate': 0.1,  
            'feature_fraction': 0.9,  
            'bagging_fraction': 0.95,  
            'bagging_freq': 5,  
            'lambda_l1': 1,    
            'lambda_l2': 0.001,  # 越小l2正则程度越高  
            'min_gain_to_split': 0.2,  
            'verbose': 5,  
            'is_unbalance': True  
        }
        return param
    def LGBMtrain(self,feature,target):
        Train_data = lgb.Dataset(feature.values,label = target.values)
        param =  {'learning_rate':0.01,'subsample':0.35,'num_leaves':31, 'num_trees':1000, 'objective':'binary','metric':'auc'}
        num_round = 3000
        progress = dict()
        params = self.lgbmparam()
        bst = lgb.train(params, Train_data, num_round,evals_result = progress)
        bst.save_model('model', num_iteration=bst.best_iteration)
        
       
    def LGBMtest(self,feature):
        model = lgb.Booster(model_file = 'model')
        preds = model.predict(feature.values)
        sub = pd.DataFrame()
        sub['User_id'] = feature['User_id']
        sub['Coupon_id']= feature['Coupon_id']
        sub['Date_received'] = feature['Date_received']
        sub['Probability'] = preds
        sub.to_csv("./data/submit.csv", sep = ",", index = False, line_terminator = '\r')
        return preds
     
    def LGBMeval(self,feature,target):
        feature_train = feature.iloc[0:self.lenth_eval,:]
        feature_eval = feature.iloc[self.lenth_eval:,:]
        label_train = target.iloc[0:self.lenth_eval]
        label_eval = target.iloc[self.lenth_eval:]
        #process = dict()
        Train_data = lgb.Dataset(feature_train.values,label = label_train.values)
        Test_data = lgb.Dataset(feature_eval.values,label = label_eval.values)
        #param = {'learning_rate':0.01,'subsample':0.35,'num_leaves':31, 'objective':'binary','metric':'auc'}
        param = self.lgbmparam()
        num_round = 3000
        lgb.train(param, Train_data, num_round,valid_sets = Test_data,early_stopping_rounds=500)
        #auc(bst,feature_eval.values,label_eval.values)
        #lossList = process['valid_0']['auc']
        #return lossList
        
    def LRtrain(self,feature,target):
        lr = LogisticRegression()  
        lr.fit(feature.values,target.values)
        #bst.save_model('model_lr')
        joblib.dump(lr, "model_lr")
        
    def LRtest(self,feature):
        clf = joblib.load("model_lr")
        preds = clf.predict_proba(feature.values)[:,1]
        sub = pd.DataFrame()
        sub['User_id'] = feature['User_id']
        sub['Coupon_id']= feature['Coupon_id']
        sub['Date_received'] = feature['Date_received']
        sub['Probability'] = preds
        sub.to_csv("./data/submit_lr.csv", sep = ",", index = False, line_terminator = '\r')
       
    def LReval(self,feature,target):
        feature_train = feature.iloc[0:self.lenth_eval,:]
        feature_eval = feature.iloc[self.lenth_eval:,:]
        label_train = target.iloc[0:self.lenth_eval]
        label_eval = target.iloc[self.lenth_eval:]
        lr = LogisticRegression()  
        
        lr.fit(feature_train.values,label_train.values)
        #lr.fit(grd_enc.transform(grd.apply(feature_train)[:, :, 0]), label_train)  
        proba_test = lr.predict_proba(feature_eval.values)[:,1]
        
        print(proba_test)
        #print(label_eval.values)
        #proba_test = np.array(proba_test)
        #print(proba_test)
        #print(proba_test.shape)
        #label_eval.values = np.array(label_eval.values)
        #fpr, tpr, thresholds = metrics.roc_curve(label_eval.values, proba_test, pos_label=2)
        #test_auc = metrics.roc_auc_score(fpr,tpr)
        
        au = metrics.auc(label_eval.values, proba_test,reorder=True)
        print(au)
        
        #print(test_auc)
        #print('auc:',au)
    def GBDTparam(self):    
        params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'binary_logloss','auc'},
        'num_leaves': 64,
        'num_trees': 100,
        'learning_rate': 0.01,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
        }
        return params
    
    def GBDTLReval(self,feature,target):
        feature_train = feature.iloc[0:self.lenth_eval,:]
        feature_eval = feature.iloc[self.lenth_eval:,:]
        label_train = target.iloc[0:self.lenth_eval]
        label_eval = target.iloc[self.lenth_eval:]
        GBDT = GradientBoostingClassifier(n_estimators=10)
        GBDT.fit(feature_train.values, label_train.values)
        
        y_pred_gbdt = GBDT.predict_proba(feature_eval.values)[:, 1]
        gbdt_auc = roc_auc_score(label_eval.values, y_pred_gbdt)
        print('gbdt auc: %.5f' % gbdt_auc)
        
        X_train_leaves = GBDT.apply(feature_train)[:,:,0]
        X_test_leaves = GBDT.apply(feature_eval)[:,:,0]
        
        (train_rows, cols) = X_train_leaves.shape
        
        gbdtenc = OneHotEncoder()
        X_trans = gbdtenc.fit_transform(np.concatenate((X_train_leaves, X_test_leaves), axis=0))

        # 定义LR模型
        lr = LogisticRegression()
        # lr对gbdt特征编码后的样本模型训练
        lr.fit(X_trans[:train_rows, :], label_train)
        # 预测及AUC评测
        y_pred_gbdtlr1 = lr.predict_proba(X_trans[train_rows:, :])[:, 1]
        gbdt_lr_auc1 = roc_auc_score(label_eval, y_pred_gbdtlr1)
        
            # 定义LR模型
        lr = LogisticRegression(n_jobs=-1)
        # 组合特征
        X_train_ext = hstack([X_trans[:train_rows, :], feature_train])
        X_test_ext = hstack([X_trans[train_rows:, :], feature_eval])

        print(X_train_ext.shape)
        # lr对组合特征的样本模型训练
        lr.fit(X_train_ext, label_train)

        # 预测及AUC评测
        y_pred_gbdtlr2 = lr.predict_proba(X_test_ext)[:, 1]
        gbdt_lr_auc2 = roc_auc_score(label_eval, y_pred_gbdtlr2)
        print('基于组合特征的LR AUC: %.5f' % gbdt_lr_auc2)
        
        print('基于GBDT特征编码后的LR AUC: %.5f' % gbdt_lr_auc1)

        print(X_train_leaves.shape)
        print(X_test_leaves.shape)
        #print(X_train_leaves.shape)
        print(X_train_leaves)
        print(X_test_leaves)

        

    '''    
    def GBDTeval(self,feature,target):
        feature_train = feature.iloc[0:self.lenth_eval,:]
        feature_eval = feature.iloc[self.lenth_eval:,:]
        label_train = target.iloc[0:self.lenth_eval]
        label_eval = target.iloc[self.lenth_eval:]
        #process = dict()
        Train_data = lgb.Dataset(feature_train.values,label = label_train.values)
        Test_data = lgb.Dataset(feature_eval.values,label = label_eval.values)
        param = self.GBDTparam()
        gbm = lgb.train(param,Train_data,num_boost_round=100,valid_sets=Test_data)
        
        gbm.save_model('model.txt')
        y_pred = gbm.predict(feature_train, pred_leaf=True) 
        #print(y_pred)
        #print(np.array(y_pred).shape)
        #print(y_pred[0])


        transformed_training_matrix = np.zeros([len(y_pred), len(y_pred[0]) * self.num_leaf],dtype=np.int64)  # N * num_tress * num_leafs
        for i in range(0, len(y_pred)):
            temp = np.arange(len(y_pred[0])) * self.num_leaf + np.array(y_pred[i])
            transformed_training_matrix[i][temp] += 1
            
        y_pred = gbm.predict(feature_eval, pred_leaf=True)
        
        print('Writing transformed testing data')
        transformed_testing_matrix = np.zeros([len(y_pred), len(y_pred[0]) * self.num_leaf], dtype=np.int64)
        for i in range(0, len(y_pred)):
            temp = np.arange(len(y_pred[0])) * self.num_leaf + np.array(y_pred[i])
            transformed_testing_matrix[i][temp] += 1
            
        lm = LogisticRegression(penalty='l2',C=0.05) # logestic model construction
        lm.fit(transformed_training_matrix,label_train.values)  # fitting the data
        y_pred_test = lm.predict_proba(transformed_testing_matrix)
    
        print(y_pred_test)
        au = metrics.auc(label_eval.values, y_pred_test,reorder=True)
        print(au)
    '''
            

        
        
        
def train():
    model = solution('eval')
    data = model.read_data()
    data = model.base_feature(data)
    data,target = model.feature_filter(data)
    #model.LGBMtrain(data,target)
    model.LRtrain(data,target)
    
def test():
    model = solution('test')
    data = model.read_data()
    
    data = model.base_feature(data)
    data = model.feature_filter(data)
    #print(data)
    #preds = model.LGBMtest(data)  
    model.LRtest(data)
    
def modeleval():
    model = solution('eval')
    data = model.read_data()
    #print(data)
    data = model.base_feature(data)
    #print(data)
    #data_train,data_eval = model.processdata(data)
    data,target = model.feature_filter(data)
    model.LGBMeval(data,target)
    #model.LReval(data,target)    
    #model.GBDTeval(data,target)
    #model.GBDTLReval(data,target)   
        
        
if __name__ == "__main__":
    train()
    test()
    #modeleval()
    
    
    
    