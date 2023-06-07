import math
import numpy as np
import lightgbm as lgb
from tqdm import tqdm
from statistics import stdev

import sekitoba_library as lib
import sekitoba_data_manage as dm
#from learn import simulation

def lg_main( data ):
    lgb_train = lgb.Dataset( np.array( data["teacher"] ), np.array( data["answer"] ) )
    lgb_vaild = lgb.Dataset( np.array( data["test_teacher"] ), np.array( data["test_answer"] ) )
    lgbm_params =  {
        #'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression_l2',
        'metric': 'l2',
        'early_stopping_rounds': 30,
        'learning_rate': 0.01182968479021641,
        'num_iteration': 9778,
        'min_data_in_bin': 1,
        'max_depth': 303,
        'num_leaves': 204,
        'min_data_in_leaf': 1,
        'lambda_l1': 0.0301332259005943,
        'lambda_l2': 0.05523927345322703
    }

    bst = lgb.train( params = lgbm_params,
                     train_set = lgb_train,     
                     valid_sets = [lgb_train, lgb_vaild ],
                     verbose_eval = 10,
                     num_boost_round = 5000 )
    
    dm.pickle_upload( lib.name.model_name(), bst )
        
    return bst
"""
def lg_main( data, prod = False ):
    max_pos = np.max( np.array( data["answer"] ) )
    lgb_train = lgb.Dataset( np.array( data["teacher"] ), np.array( data["answer"] ), group = np.array( data["query"] ) )
    lgb_vaild = lgb.Dataset( np.array( data["test_teacher"] ), np.array( data["test_answer"] ), group = np.array( data["test_query"] ) )
    lgbm_params =  {
        #'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'lambdarank',
        'metric': 'ndcg',   # for lambdarank
        'ndcg_eval_at': [1,2,3],  # for lambdarank
        'label_gain': list(range(0, np.max( np.array( data["answer"], dtype = np.int32 ) ) + 1)),
        'max_position': int( max_pos ),  # for lambdarank
        'early_stopping_rounds': 30,
        'learning_rate': 0.05,
        'num_iteration': 300,
        'min_data_in_bin': 1,
        'max_depth': 200,
        'num_leaves': 175,
        'min_data_in_leaf': 25,
    }

    bst = lgb.train( params = lgbm_params,
                     train_set = lgb_train,     
                     valid_sets = [lgb_train, lgb_vaild ],
                     verbose_eval = 10,
                     num_boost_round = 5000 )
    
    dm.pickle_upload( lib.name.model_name(), bst )
        
    return bst
"""

def standardization( data ):
    ave = sum( data ) / len( data )
    std = stdev( data )

    for i in range( 0, len( data ) ):
        data[i] = ( data[i] - ave ) / std

    return data

def data_check( data ):
    result = {}
    result["teacher"] = []
    result["test_teacher"] = []
    result["answer"] = []
    result["test_answer"] = []
    result["query"] = []
    result["test_query"] = []

    for i in range( 0, len( data["teacher"] ) ):
        year = data["year"][i]
        query = len( data["teacher"][i] )

        if year in lib.test_years:
            result["test_query"].append( query )
        else:
            result["query"].append( query )

        n = int( query / 3 )
        corner_horce_body_list = []

        for r in range( 0, query ):
            corner_horce_body_list.append( data["horce_body"][i][r] )

        corner_horce_body_list = standardization( corner_horce_body_list )

        for r in range( 0, query ):
            current_data = data["teacher"][i][r]
            first_rank = int( data["answer"][i][r] )
            current_answer = first_rank

            if first_rank / n < 1:
                current_answer -= 1
            elif first_rank / n > 2:
                current_answer += 1

            if first_rank == 1 or first_rank == 2:
                current_answer -= 1

            if corner_horce_body_list[r] < -1:
                current_answer -= 1
            elif 1 < corner_horce_body_list[r]:
                current_answer += 1
                
            if year in lib.test_years:
                result["test_teacher"].append( current_data )
                result["test_answer"].append( current_answer )
            else:
                result["teacher"].append( current_data )
                result["answer"].append( current_answer  )

    return result

def score_check( simu_data, model ):
    score1 = 0
    score2 = 0
    count = 0
    simu_predict_data = {}
    predict_use_data = []

    for race_id in simu_data.keys():
        for horce_id in simu_data[race_id].keys():
            predict_use_data.append( simu_data[race_id][horce_id]["data"] )

    c = 0
    predict_data = model.predict( np.array( predict_use_data ) )

    for race_id in simu_data.keys():
        year = race_id[0:4]
        check_data = []
        simu_predict_data[race_id] = {}
        all_horce_num = len( simu_data[race_id] )
        
        for horce_id in simu_data[race_id].keys():
            predict_score = min( predict_data[c], all_horce_num )
            c += 1
            answer_rank = simu_data[race_id][horce_id]["answer"]["first_passing_rank"]
            check_data.append( { "horce_id": horce_id, "answer": answer_rank, "score": predict_score } )

        check_data = sorted( check_data, key = lambda x: x["score"] )
        before_score = 1
        next_rank = 1
        continue_count = 1
        
        for i in range( 0, len( check_data ) ):
            predict_score = -1
            current_score = int( check_data[i]["score"] + 0.5 )

            if continue_count >= 2:
                next_rank += continue_count
                continue_count = 0
            
            if i == 0:
                predict_score = 1
            elif before_score == current_score:
                continue_count += 1
                predict_score = next_rank
            else:
                next_rank += continue_count
                continue_count = 1
                predict_score = next_rank

            check_answer = check_data[i]["answer"]
            before_score = current_score
            #predict_score = int( check_data[i]["score"] + 0.5 )
            simu_predict_data[race_id][check_data[i]["horce_id"]] = predict_score

            if year in lib.test_years:
                score1 += math.pow( predict_score - check_answer, 2 )
                score2 += math.pow( max( int( check_data[i]["score"] + 0.5 ), 1 ) - check_answer, 2 )
                count += 1            
            
    score1 /= count
    score1 = math.sqrt( score1 )
    print( "score1: {}".format( score1 ) )

    score2 /= count
    score2 = math.sqrt( score2 )
    print( "score2: {}".format( score2 ) )

    dm.pickle_upload( "predict_first_passing_rank.pickle", simu_predict_data )
    
"""    
def score_check( simu_data, model ):
    score = 0
    count = 0
    simu_predict_data = {}
    
    for race_id in simu_data.keys():
        predict_data = []
        simu_predict_data[race_id] = {}
        
        for horce_id in simu_data[race_id].keys():
            predict_score = model.predict( np.array( [ simu_data[race_id][horce_id]["data"] ] ) )[0]
            first_passing_rank = simu_data[race_id][horce_id]["answer"]["first_passing_rank"]
            predict_data.append( { "score": predict_score, "rank": first_passing_rank, "horce_id": horce_id } )

        predict_data = sorted( predict_data, key = lambda x: x["score"] )
        #print( predict_data )
        for i in range( 0, len( predict_data ) ):
            predict_rank = i + 1
            horce_id = predict_data[i]["horce_id"]
            score += math.pow( predict_rank - predict_data[i]["rank"], 2 )
            count += 1
            simu_predict_data[race_id][horce_id] = predict_rank

    score /= count
    score = math.sqrt( score )
    print( "score: {}".format( score ) )
    dm.pickle_upload( "predict_first_passing_rank.pickle", simu_predict_data )
"""

def importance_check( model ):
    result = []
    importance_data = model.feature_importance()
    f = open( "common/rank_score_data.txt" )
    all_data = f.readlines()
    f.close()
    c = 0

    for i in range( 0, len( all_data ) ):
        str_data = all_data[i].replace( "\n", "" )

        if "False" in str_data:
            continue

        result.append( { "key": str_data, "score": importance_data[c] } )
        c += 1

    result = sorted( result, key = lambda x: x["score"], reverse= True )

    for i in range( 0, len( result ) ):
        print( "{}: {}".format( result[i]["key"], result[i]["score"] ) )

def main( data, simu_data, learn = True ):
    learn_data = data_check( data )

    if learn:
        model = lg_main( learn_data )
        importance_check( model )
    else:
        model = dm.pickle_load( lib.name.model_name() )
        
    score_check( simu_data, model )
    #score_check( learn_data["test_teacher"], learn_data["test_answer"], model )

    return model
