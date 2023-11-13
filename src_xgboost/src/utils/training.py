# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

'''
This code has the functions needed for train-test-split, clustering and model training
'''

import time

import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

#将数据集分为训练集和测试集
def split_data(raw_data,flag):
    if flag:
        from sklearnex import patch_sklearn  # pylint: disable=E0401, C0415
        patch_sklearn()
    from sklearn.model_selection import train_test_split
    features_data = raw_data.drop(columns=['Class'])
    class_data = raw_data['Class']
    start_time = time.time()
    X_train, X_test, y_train, y_test = train_test_split(features_data, class_data, stratify=class_data, test_size=0.3, random_state=42)
    split_time = time.time() - start_time
    return X_train, X_test, y_train, y_test, split_time

#data_raw：要进行聚类的原始数据
#features_of_interest：要执行聚类的特征。
#epsilon：两个样本被认为在彼此邻近的最大距离。
#min_samp：在邻域内被视为核心点的样本数（或总权重）。
#flag：一个布尔标志。如果为True，则导入并修补sklearnex模块。
def DBSCAN_Clustering(data_raw, features_of_interest, epsilon, min_samp, flag):
    if flag:
        from sklearnex import patch_sklearn  # pylint: disable=E0401, C0415
        patch_sklearn()
    from sklearn.cluster import DBSCAN  # pylint: disable=C0415
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    data_for_clustering = data_raw[features_of_interest]
    data_for_clustering_scaled = scaler.fit_transform(data_for_clustering)
    lst_clustering_time = []
    
    start_time = time.time()
    db = DBSCAN(eps=epsilon, min_samples=min_samp, n_jobs=-1).fit(data_for_clustering_scaled)
    lst_clustering_time.append(time.time()-start_time)
    clustering_time = min(lst_clustering_time)
    data_for_clustering['Clusters'] = db.labels_
    return data_for_clustering, clustering_time

def xgb_model_train(df_for_training, class_for_training, param_dict,flag):
    xgb_clt = xgb.XGBClassifier(**param_dict, random_state=42)
    lst_training_time = []
    
    start_time = time.time()
    xgb_clt.fit(df_for_training.drop(columns=['Clusters']), class_for_training)
    lst_training_time.append(time.time()-start_time)
    train_time = min(lst_training_time)
    return xgb_clt, train_time

def xgb_model_hyper(df_for_training, class_for_training, param_dict,flag):
    if flag:
        from sklearnex import patch_sklearn  # pylint: disable=E0401, C0415
        patch_sklearn()
    from sklearn.model_selection import GridSearchCV
    xgb_clt = xgb.XGBClassifier(**param_dict, random_state=42)
    gs = GridSearchCV(xgb_clt, param_grid=param_dict)
    lst_hyper_time = []
    
    start_time = time.time()
    gs_results = gs.fit(df_for_training.drop(columns=['Clusters']), class_for_training)
    lst_hyper_time.append(time.time()-start_time)
    hyper_time = min(lst_hyper_time)
    best_grid = gs_results.best_estimator_
    return best_grid, hyper_time