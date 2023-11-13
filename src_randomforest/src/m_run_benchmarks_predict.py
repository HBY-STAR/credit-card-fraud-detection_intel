# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

'''
Master code for executing prediction benchmarks
'''

if __name__ == "__main__":

    import argparse
    import logging
    logging.getLogger('matplotlib.font_manager').disabled = True
    from utils.prediction import rf_model_predict
    from utils.data_processing import read_data
    import joblib
    import pathlib
    import pandas as pd
    import warnings
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    from sklearn import preprocessing
    lbl = preprocessing.LabelEncoder()
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()

    parser.add_argument('-l',
                        '--logfile',
                        type=str,
                        default="m_predict.log",
                        help="log file to output benchmarking results to")

    parser.add_argument('-i',
                        '--intel',
                        default=False,
                        action="store_true",
                        help="use intel accelerated technologies where available")
    parser.add_argument('-mc',
                        '--clusteredmodel',
                        type=str,
                        default="Clustered_RF_Classifier.pkl",
                        help="provide clustered model")
    parser.add_argument('-mf',
                        '--fullmodel',
                        type=str,
                        default="Full_RF_Classifier.pkl",
                        help="provide full model")

    FLAGS = parser.parse_args()

    if FLAGS.logfile == "":
        logging.basicConfig(level=logging.INFO)
    else:
        path = pathlib.Path(FLAGS.logfile)
        path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=FLAGS.logfile, level=logging.INFO)
        
    logger = logging.getLogger()
    intel_flag = FLAGS.intel

    model_clustered = joblib.load(FLAGS.clusteredmodel)
    model_full = joblib.load(FLAGS.fullmodel)

    filepath = "..\data\creditcard_test.csv"
    
    logger.info("Predict:")
    logger.info("=======> Reading Data...")
    test_data, read_time = read_data(filepath,intel_flag)
    logger.info("=======> Read Data time = %s", str(read_time))
    logger.info("\n")
    
    for multiplier in [0.5, 1, 2, 5, 10]:
        if multiplier == 0.5:
            test_data_large = test_data.sample(42500)
        else:
            test_data_large = pd.concat([test_data]*multiplier)
        test_data_large = test_data_large.sample(frac=1).reset_index(drop=True)
        X_test = test_data_large.drop(columns='Class')
        y_test = test_data_large['Class']
        
        logger.info("=======> Running the model on dataframe length of = %s", str(len(X_test)))

        logger.info("=======> Testing model_cluster on test data")
        rec_score, macro_f1score, positives, pred_time = rf_model_predict(intel_flag, model_clustered, X_test, y_test, 'clustered.png')
        logger.info("=======> Recall of model on test data = %s", str(rec_score))
        logger.info("=======> Macro f1 score of model on test data = %s", str(macro_f1score))
        logger.info("=======> Total Positives Predicted = %s", str(positives))
        logger.info("=======> RF prediction time for clustered data = %s", str(pred_time))

        logger.info("\n")

        logger.info("=======> Testing model_full on test data")
        rec_score, macro_f1score, positives, pred_time = rf_model_predict(intel_flag, model_full, X_test, y_test, 'full.png')
        logger.info("=======> Recall of model on full data = %s", str(rec_score))
        logger.info("=======> Macro f1 score of model on full data = %s", str(macro_f1score))
        logger.info("=======> Total Positives Predicted = %s", str(positives))
        logger.info("=======> RF prediction time for full data = %s", str(pred_time))
        
        logger.info("\n")  
            
    # 画出ROC曲线
    # cluster
    test_data_large = pd.concat([test_data]*1)
    test_data_large = test_data_large.sample(frac=1).reset_index(drop=True)
    X_test = test_data_large.drop(columns='Class')
    y_test = test_data_large['Class']
    size = len(y_test)
    params = {'legend.fontsize': 'x-large',
              'figure.figsize': (12, 9),
              'savefig.dpi': 200,
              'figure.dpi':200,
             'axes.labelsize': 'x-large',
             'axes.titlesize':'x-large',
             'xtick.labelsize':'x-large',
             'ytick.labelsize':'x-large'}
    plt.rcParams.update(params)
    y_score = model_clustered.predict_proba(X_test)
    #print(y_score)
    fpr,tpr,threshold = roc_curve(y_test, y_score[:,1]) ###计算真正率和假正率
    #print(tpr)
    #print(fpr)
    roc_auc = auc(fpr,tpr)
    lw = 2
    plt.plot(fpr, tpr, color='blue', 
             lw=lw, label='Cluster ROC (area = %0.5f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
    
    # full
    y_score = model_full.predict_proba(X_test)
    fpr,tpr,threshold = roc_curve(y_test, y_score[:,1])
    roc_auc = auc(fpr,tpr) 
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', 
             lw=lw, label='Full ROC (area = %0.5f)' % roc_auc)
    
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='r',
             label='Chance', alpha=.6)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('RF ROC: Test-data-num-'+str(size))
    plt.legend(loc="lower right")
    if FLAGS.intel == False:
        plt.savefig("RF_ROC_patch")
    else:
        plt.savefig("RF_ROC_not_patch")
    plt.show()


