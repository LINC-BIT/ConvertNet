from typing import List
import torch
from utils.common.log import logger


def read_data_points_v2(data_points_file_path_list):
    res = []
    for data_points_file_path in data_points_file_path_list:
        raw_data_points = torch.load(data_points_file_path)
        res += raw_data_points
    logger.info(f'read {len(res)} data points')  
    res = [(i[0], i[1]) for i in res]
    return res

def read_data_points(data_points_file_path, use_aux_info=False, source_target_features_mean_covariance_file_path=None):
    raw_data_points = torch.load(data_points_file_path)
    if use_aux_info:
        return raw_data_points
        
    logger.info(f'read {len(raw_data_points)} data points')  
    
    res = [(i[0], i[1]) for i in raw_data_points]  

    if source_target_features_mean_covariance_file_path is not None:
        source_target_features_mean_covariance = torch.load(source_target_features_mean_covariance_file_path)
        for i in range(len(res)):
            res[i][0]['source_features_mean'] = source_target_features_mean_covariance[i][0]
            res[i][0]['source_features_var'] = source_target_features_mean_covariance[i][1]
            res[i][0]['target_features_mean'] = source_target_features_mean_covariance[i][2]
            res[i][0]['target_features_var'] = source_target_features_mean_covariance[i][3]
            
    return res


def find_data_points_with_other_variables_constant(data_points, other_variables_value, other_variables_allow_condition):
    res = []
    for x, y in data_points:
        ok = True
        for k, v in other_variables_value.items():
            if isinstance(other_variables_allow_condition[k], (int, float)) and abs(x[k] - v) > other_variables_allow_condition[k]:
                ok = False
                break
            if not other_variables_allow_condition[k](x[k], v):
                ok = False
                break
        
        if ok:
            res += [(x, y)]
            
    logger.info(f'{len(res)} data points preserved')  
    
    return res
