import optuna
import math
import numpy as np
import lightgbm as lgb
from tqdm import tqdm
from statistics import stdev

import sekitoba_library as lib
import sekitoba_data_manage as dm
#from learn import simulation

data = {}
simu_data = {}

def objective( trial ):
    lgb_train = lgb.Dataset( np.array( data["teacher"] ), np.array( data["answer"] ) )
    lgb_vaild = lgb.Dataset( np.array( data["test_teacher"] ), np.array( data["test_answer"] ) )

    learning_rate = trial.suggest_float( 'learning_rate', 0.005, 0.03 )
    num_leaves =  trial.suggest_int( "num_leaves", 50, 300 )
    max_depth = trial.suggest_int( "max_depth", 200, 500 )
    num_iteration = trial.suggest_int( "num_iteration", 5000, 15000 )
    min_data_in_leaf = trial.suggest_int( "min_data_in_leaf", 1, 50 )
    lambda_l1 = trial.suggest_float( "lambda_l1", 0, 1 )
    lambda_l2 = trial.suggest_float( "lambda_l2", 0, 1 )

    lgbm_params =  {
        #'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression_l2',
        'metric': 'l2',
        'early_stopping_rounds': 30,
        'learning_rate': learning_rate,
        'num_iteration': num_iteration,
        'min_data_in_bin': 1,
        'max_depth': max_depth,
        'num_leaves': num_leaves,
        'min_data_in_leaf': min_data_in_leaf,
        'lambda_l1': lambda_l1,
        'lambda_l2': lambda_l2
    }

    model = lgb.train( params = lgbm_params,
                     train_set = lgb_train,     
                     valid_sets = [lgb_train, lgb_vaild ],
                     verbose_eval = 10,
                     num_boost_round = 5000 )

    return score_check( model ) * 10

def best_model_create( params ):
    lgb_train = lgb.Dataset( np.array( data["teacher"] ), np.array( data["answer"] ) )
    lgb_vaild = lgb.Dataset( np.array( data["test_teacher"] ), np.array( data["test_answer"] ) )

    lgbm_params =  {
        #'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression_l2',
        'metric': 'l2',
        'early_stopping_rounds': 30,
        'learning_rate': params["learning_rate"],
        'num_iteration': params["num_iteration"],
        'min_data_in_bin': 1,
        'max_depth': params["max_depth"],
        'num_leaves': params["num_leaves"],
        'min_data_in_leaf': params["min_data_in_leaf"],
        'lambda_l1': params["lambda_l1"],
        'lambda_l2': params["lambda_l2"]
    }

    model = lgb.train( params = lgbm_params,
                     train_set = lgb_train,     
                     valid_sets = [lgb_train, lgb_vaild ],
                     verbose_eval = 10,
                     num_boost_round = 5000 )

    score_check( model, upload = True )
    dm.pickle_upload( lib.name.model_name(), model )

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

def score_check( model, upload = False ):
    score1 = 0
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
                count += 1            
            
    score1 /= count
    score1 = math.sqrt( score1 )
    print( "score1: {}".format( score1 ) )

    if upload:
        dm.pickle_upload( "predict_first_passing_rank.pickle", simu_predict_data )

    return score1

def main( arg_data, arg_simu_data ):
    global data
    global simu_data
    simu_data = arg_simu_data
    data = data_check( arg_data )

    study = optuna.create_study()
    study.optimize(objective, n_trials=100)
    print( study.best_params )

    best_model_create( study.best_params )
    f = open( "best_params.txt", "w" )
    f.write( str( study.best_params) )
    f.close()
