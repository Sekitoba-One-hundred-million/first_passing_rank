import math
import numpy as np
from tqdm import tqdm

import SekitobaLibrary as lib
import SekitobaDataManage as dm

def teacher_stand( data, simu_data, state = "test" ):
    data_list = []
    """
    for i in range( 0, len( data["answer"] ) ):
        data_check = lib.testYearCheck( data["year"][i], state )

        if not data_check == "teacher":
            continue

        for r in range( 0, len( data["answer"][i] ) ) :
            data_list.append( data["answer"][i][r] )

    ave_data = lib.average( data_list )
    std_data = lib.stdev( data_list )

    for i in range( 0, len( data["answer"] ) ):
        for r in range( 0, len( data["answer"][i] ) ) :
            value = data["answer"][i][r]

            if not value == lib.base_abort:
                data["answer"][i][r] = ( value - ave_data ) / std_data
    """
    for t in tqdm( range( 0, len( data["teacher"][0][0] ) ) ):
        data_list.clear()
        
        for i in range( 0, len( data["teacher"] ) ):
            data_check = lib.testYearCheck( data["year"][i], state )

            if not data_check == "teacher":
                continue

            for r in range( 0, len( data["teacher"][i] ) ):
                data_list.append( data["teacher"][i][r][t] )

        ave_data = lib.average( data_list )
        std_data = lib.stdev( data_list )

        if std_data == 0:
            continue

        for i in range( 0, len( data["teacher"] ) ):
            for r in range( 0, len( data["teacher"][i] ) ):
                value = data["teacher"][i][r][t]

                if not value == lib.base_abort:
                    data["teacher"][i][r][t] = ( value - ave_data ) / std_data

        for race_id in simu_data.keys():
            for horce_id in simu_data[race_id].keys():
                value = simu_data[race_id][horce_id]["data"][t]

                if not value == lib.base_abort:
                    simu_data[race_id][horce_id]["data"][t] = ( value - ave_data ) / std_data

def data_check( data, state = "test" ):
    result = {}
    result["teacher"] = []
    result["test_teacher"] = []
    result["answer"] = []
    result["test_answer"] = []
    result["query"] = []
    result["test_query"] = []

    for i in range( 0, len( data["teacher"] ) ):
        query = len( data["teacher"][i] )
        data_check = lib.testYearCheck( data["year"][i], state )
        
        if data_check == "test":
            result["test_query"].append( query )
        elif data_check == "teacher":
            result["query"].append( query )

        n = int( query / 3 )

        if n == 0:
            n = 1

        one_rank_count = 0

        for r in range( 0, query ):
            if int( data["answer"][i][r] ) == 1:
                one_rank_count += 1

        for r in range( 0, query ):
            current_data = data["teacher"][i][r]
            first_rank = int( data["answer"][i][r] )
            horce_body = data["horce_body"][i][r]
            current_answer = first_rank

            if data_check == "test":
                result["test_teacher"].append( current_data )
                result["test_answer"].append( current_answer )
            elif data_check == "teacher":
                result["teacher"].append( current_data )
                result["answer"].append( current_answer  )

    return result

def score_check( simu_data, model, score_years = lib.test_years, upload = False ):
    score = 0
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
        race_place_num = race_id[4:6]
        check_data = []
        stand_score_list = []
        simu_predict_data[race_id] = {}
        all_horce_num = len( simu_data[race_id] )
        
        for horce_id in simu_data[race_id].keys():
            predict_score = min( predict_data[c], all_horce_num )
            c += 1
            answer_rank = simu_data[race_id][horce_id]["answer"]["first_passing_rank"]
            check_data.append( { "horce_id": horce_id, "answer": answer_rank, "score": predict_score } )
            stand_score_list.append( predict_score )

        stand_score_list = lib.standardization( stand_score_list )
        check_data = sorted( check_data, key = lambda x: x["score"] )
        before_score = 1
        next_rank = 1
        continue_count = 1
        
        for i in range( 0, len( check_data ) ):
            check_answer = check_data[i]["answer"]
            simu_predict_data[race_id][check_data[i]["horce_id"]] = {}
            simu_predict_data[race_id][check_data[i]["horce_id"]]["index"] = i + 1
            simu_predict_data[race_id][check_data[i]["horce_id"]]["score"] = min( max( check_data[i]["score"], 1 ), len( check_data ) )
            simu_predict_data[race_id][check_data[i]["horce_id"]]["stand"] = stand_score_list[i]

            if year in score_years:
                score += math.pow( min( max( int( check_data[i]["score"] ), 1 ), len( check_data ) ) - check_answer, 2 )
                count += 1
            
    score /= count
    score = math.sqrt( score )
    print( "score: {}".format( score ) )

    if upload:
        dm.pickle_upload( "predict_first_passing_rank.pickle", simu_predict_data )

    return score
