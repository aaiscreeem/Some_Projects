ENTRY_NUMBER_LAST_DIGIT = 7 # change with yours
ENTRY_NUMBER = '2022MT11257'
PRE_PROCESSING_CONFIG ={
    "hard_margin_linear" : {
        "use_pca" : False,
        "threshold_variance":1.0
    },

    "hard_margin_rbf" : {
        "use_pca" : True,
        "threshold_variance":0.3
    },

    "soft_margin_linear" : {
        "use_pca" : False,
        "threshold_variance":1.0
    },

    "soft_margin_rbf" : {
        "use_pca" : True,
        "threshold_variance":0.3
    },

    "AdaBoost" : {
        "use_pca" : True,
        "threshold_variance":0.8
    },

    "RandomForest" : {
        "use_pca" : True,
        "threshold_variance":0.35
    }
}


SVM_CONFIG = {
    "hard_margin_linear" : {
        "C" : 1e9,
        "kernel" : 'linear',
        "zero_threshold" : 1e-5,  # a data point is support vector if alpha is greater than this value, to handle numerical errors
        "val_score" : 0.87, 
                         
        
    },
    "hard_margin_rbf" : {
        "C" : 1e9,
        "kernel" : 'rbf',
        "gamma" : 1e-2,
        "zero_threshold" : 1e-5,
        "val_score" : 0, # add the validation score you get on val set for the set hyperparams.
                         # Diff in your and our calculated score will results in severe penalites
        # add implementation specific hyperparams below (with one line explanation)
    },

    "soft_margin_linear" : {
        "C" : 1e-2, # add your best hyperparameter
        "kernel" : 'linear',
        "zero_threshold" : 1e-5,
        "val_score" : 0.93, # add the validation score you get on val set for the set hyperparams.
                         # Diff in your and our calculated score will results in severe penalites
        # add implementation specific hyperparams below (with one line explanation)
    },

    "soft_margin_rbf" : {
         "C" : 1e-2, # add your best hyperparameter
         "kernel" : 'rbf',
         "gamma" : 1e-2,
         "zero_threshold" : 1e-5,
         "val_score" : 0.84, # add the validation score you get on val set for the set hyperparams.
                          # Diff in your and our calculated score will results in severe penalites
         # add implementation specific hyperparams below (with one line explanation)
    }
}

ENSEMBLING_CONFIG = {
    'AdaBoost':{
        'num_trees' : 36,
        "val_score" : 0.875,   
    },

    'RandomForest':{
        'num_trees' : 20,
        "bootstrap_fraction" : 0.3,
        "min_samples_split" : 7,
        "max_depth" : 20,
        "min_gain" : 0.2,
        "criterion" : "gini",
        "class_weight" : 1.0,
        "k_features" : None,
        "weighted_forest" : True,
        "val_score" : 0.85,
    }
}