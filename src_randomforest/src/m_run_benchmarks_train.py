# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

'''
Master code for executing training benchmarks
'''

if __name__ == "__main__":

    import argparse
    import logging
    logging.getLogger('matplotlib.font_manager').disabled = True
    import numpy as np
    np.random.seed(42)
    from utils.training import split_data, DBSCAN_Clustering,rf_model_train
    from utils.data_processing import read_data, filter_clusters
    import joblib
    import pathlib
    import warnings
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()

    parser.add_argument('-l',
                        '--logfile',
                        type=str,
                        default="m_train.log",
                        help="log file to output benchmarking results to")

    parser.add_argument('-i',
                        '--intel',
                        default=False,
                        action="store_true",
                        help="use intel accelerated technologies where available")

    FLAGS = parser.parse_args()

    if FLAGS.logfile == "":
        logging.basicConfig(level=logging.INFO)
    else:
        path = pathlib.Path(FLAGS.logfile)
        path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=FLAGS.logfile, level=logging.INFO)
    logger = logging.getLogger()
    intel_flag = FLAGS.intel
    MODEL_FILE1 = 'RF_Classifier.pkl'
    filepath = "..\data\creditcard.csv"
    
    logger.info("Train:")
    logger.info("=======> Reading Data...")
    credit_card_data, read_time = read_data(filepath,intel_flag)
    logger.info("=======> Read Data time = %s", str(read_time))
    logger.info("\n")
    logger.info("=======> Splitting Data into Train/Test...")
    X_train, X_test, y_train, y_test, split_time= split_data(credit_card_data, intel_flag)
    logger.info("=======> Split time = %s", str(split_time))
    logger.info("\n")

    most_important_names = ['V16', 'V14']
    eps_val = 0.3
    minimum_samples = 20
    logger.info("=======> Running DBSCAN Clustering and filtering for classification...")
    df_clusters, cluster_time = DBSCAN_Clustering(X_train, most_important_names, eps_val, minimum_samples, intel_flag)
    X_train_clustered, y_train_clustered = filter_clusters(X_train, y_train, df_clusters)
    logger.info("=======> DBSCAN Clustering time = %s", str(cluster_time))
    logger.info("\n")

    params = {'max_depth': 5,
              'min_samples_split': 20,
              'n_estimators' : 100,
              'max_features': 20,
              }

    X_test['Class'] = y_test
    X_test.to_csv("..\data\creditcard_test.csv", index=False)
    
    logger.info("=======> Length of training dataframe post clustering = %s", str(len(X_train_clustered)))
    logger.info("=======> Value counts of labeled data on training set = %s", str(sum(y_train_clustered)/len(y_train_clustered)))
    logger.info("=======> Value counts of labeled data on test set = %s", str(sum(y_test)/len(y_test)))
    logger.info("=======> Length of full training dataframe = %s", str(len(X_train)))
    logger.info("\n")

    logger.info("=======> Training model_cluster on clustered data")
    rf_model, train_time = rf_model_train(X_train_clustered, y_train_clustered, params,intel_flag)
    logger.info("=======> RF training time for clustered data = %s", str(train_time))
    MODEL_FILE = 'Clustered_' + MODEL_FILE1
    joblib.dump(rf_model, MODEL_FILE)
    
    logger.info("\n")

    logger.info("=======> Training model on Full data")
    rf_model_full, train_time = rf_model_train(X_train, y_train, params, intel_flag)
    logger.info("=======> RF training time for full data = %s", str(train_time))
    MODEL_FILE = 'Full_' + MODEL_FILE1
    joblib.dump(rf_model_full, MODEL_FILE)
    logger.info("\n\n\n\n")

