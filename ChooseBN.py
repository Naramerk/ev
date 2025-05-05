from bamt.networks.hybrid_bn import HybridBN
import pandas as pd
import logging
from bamt.preprocess.discretization import code_categories, get_nodes_type
from bamt.networks.continuous_bn import ContinuousBN
from bamt.networks.discrete_bn import DiscreteBN

def selectBN(X, preprocessor,flg=0):
    #1 добавить строку
    cat_columns = X.select_dtypes(include=["object", "category", "bool", "string", "int64"]).columns.tolist()
    if cat_columns:
        X, cat_encoder = code_categories(X, method="label", columns=cat_columns)
    discretized_data, est = preprocessor.apply(X)
    #discretized_data = pd.DataFrame(discretized_data, columns=X.columns)
    info = preprocessor.info  # Get information about the data types after preprocessing
    # Get types of nodes
    print(info)
    if flg==1:
        return info
    get_nodes_type = info['types']
    print('get_nodes_type',get_nodes_type)
    
    # Check for discrete and continuous columns
    values = get_nodes_type.values()
    has_discrete = any(value in ['disc', 'disc_num'] for value in values)
    has_continuous = any(value in ['cont'] for value in values)
    
    # If we have categorical columns, we have discrete data
    if cat_columns:
        has_discrete = True
    
    if has_continuous and not has_discrete:
        bn = ContinuousBN(use_mixture=True)
        logging.info("Using ContinuousBN")
    elif has_discrete and not has_continuous:
        bn = DiscreteBN()
        logging.info("Using DiscreteBN")
    else:
        bn = HybridBN(has_logit=True, use_mixture=True)
        logging.info("Using HybridBN")
    bn.add_nodes(info)
    return bn, discretized_data