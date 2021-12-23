import math
from tqdm import tqdm

import sekitoba_library as lib
import sekitoba_data_manage as dm
from data_analyze.train_index_get import train_index_get
from data_analyze.time_index_get import time_index_get
from data_analyze.jockey_data_get import JockeyData
from data_analyze import parent_data_get

dm.dl.file_set( "race_cource_info.pickle" )
dm.dl.file_set( "race_cource_wrap.pickle" )
dm.dl.file_set( "first_pace_analyze_data.pickle" )
dm.dl.file_set( "race_info_data.pickle" )
dm.dl.file_set( "race_data.pickle" )
dm.dl.file_set( "first_pace_analyze_data.pickle" )
dm.dl.file_set( "passing_data.pickle" )
dm.dl.file_set( "horce_data_storage.pickle" )
dm.dl.file_set( "corner_horce_body.pickle" )
dm.dl.file_set( "baba_index_data.pickle" )
dm.dl.file_set( "parent_id_data.pickle" )
dm.dl.file_set( "time_index_data.pickle" )
dm.dl.file_set( "race_limb_claster_model.pickle" )
dm.dl.file_set( "limb_passing_rank.pickle" )
dm.dl.file_set( "limb_num_horce_body.pickle" )
dm.dl.file_set( "first_horce_body_encoding.pickle" )
dm.dl.file_set( "first_up3_halon.pickle" )

def past_horce_body_get( data, horce_num, race_id_list ):
    ave = 0
    count = 0
    
    for race_id in race_id_list:
        try:
            key = min( data[race_id].keys() )
            ave += data[race_id][key][horce_num]
            count += 1
        except:
            continue

    if count == 0:
        ave = -1
    else:
        ave /= count

    return ave

def main( update = False ):
    result = None
    
    if not update:
        result = dm.pickle_load( "first_horce_body_data.pickle" )
        simu_data = dm.pickle_load( "first_horce_body_simu_data.pickle" )

    if result == None:
        result = {}
        simu_data = {}        
    else:
        return result, simu_data

    result["answer"] = []
    result["answer_rank"] = []
    result["answer_list"] = []
    result["teacher"] = []
    result["year"] = []
    result["query"] = []
    min_horce_body = 10000
    max_horce_body = -1

    race_data = dm.dl.data_get( "race_data.pickle" )
    horce_data = dm.dl.data_get( "horce_data_storage.pickle" )
    race_cource_wrap = dm.dl.data_get( "race_cource_wrap.pickle" )
    race_info = dm.dl.data_get( "race_info_data.pickle" )
    first_pace_analyze_data = dm.dl.data_get( "first_pace_analyze_data.pickle" )
    passing_data = dm.dl.data_get( "passing_data.pickle" )
    race_cource_info = dm.dl.data_get( "race_cource_info.pickle" )
    corner_horce_body = dm.dl.data_get( "corner_horce_body.pickle" )
    baba_index_data = dm.dl.data_get( "baba_index_data.pickle" )
    parent_id_data = dm.dl.data_get( "parent_id_data.pickle" )
    first_horce_body_encoding = dm.dl.data_get( "first_horce_body_encoding.pickle" )
    race_limb_claster_model = dm.dl.data_get( "race_limb_claster_model.pickle" )
    limb_passing_rank = dm.dl.data_get( "limb_passing_rank.pickle" )
    limb_num_horce_body = dm.dl.data_get( "limb_num_horce_body.pickle" )
    first_up3_halon = dm.dl.data_get( "first_up3_halon.pickle" )    
    train_index = train_index_get()
    time_index = time_index_get()
    jockey_data = JockeyData()

    for k in tqdm( race_data.keys() ):
        race_id = lib.id_get( k )
        year = race_id[0:4]
        race_place_num = race_id[4:6]
        day = race_id[9]
        num = race_id[7]

        try:
            current_wrap = race_cource_wrap[race_id]
        except:
            continue

        key_place = str( race_info[race_id]["place"] )
        key_dist = str( race_info[race_id]["dist"] )
        key_kind = str( race_info[race_id]["kind"] )        
        key_baba = str( race_info[race_id]["baba"] )
        ri_list = [ key_place + ":place", key_dist + ":dist", key_kind + ":kind", key_baba + ":baba" ]        
        info_key_dist = key_dist
        
        if race_info[race_id]["out_side"]:
            info_key_dist += "外"
        
        rci_dist = race_cource_info[key_place][key_kind][info_key_dist]["dist"]
        rci_info = race_cource_info[key_place][key_kind][info_key_dist]["info"]

        race_limb = [0] * 9
        popular_limb = -1
        train_index_list = train_index.main( race_data[k], horce_data, race_id )
        time_index_race_data = { "max": 0, "min": 10000, "average": 0, "count": 0, "my": {} }
        speed_index_race_data = { "max": -1000, "min": 10000, "average": 0, "count": 0, "my": {} }
        up_speed_index_race_data = { "max": -1000, "min": 10000, "average": 0, "count": 0, "my": {} }
        pace_speed_index_race_data = { "max": -1000, "min": 10000, "average": 0, "count": 0, "my": {} }
        count = -1
        #omega = omega_data[race_id]

        for kk in race_data[k].keys():
            horce_id = kk
            current_data, past_data = lib.race_check( horce_data[horce_id],
                                                     year, day, num, race_place_num )#今回と過去のデータに分ける
            cd = lib.current_data( current_data )
            pd = lib.past_data( past_data, current_data )
            
            if not cd.race_check():
                continue

            current_time_index = time_index.main( kk, pd.past_day_list() )
            speed, up_speed, pace_speed = pd.speed_index( baba_index_data[horce_id] )
            time_index_race_data["my"][kk] = current_time_index

            if not current_time_index["max"] == 0:
                time_index_race_data["max"] = max( time_index_race_data["max"], current_time_index["max"] )
                time_index_race_data["min"] = max( time_index_race_data["min"], current_time_index["max"] )
                time_index_race_data["average"] += time_index_race_data["max"]
                time_index_race_data["count"] += 1

            if not len( speed ) == 0:
                speed_index_race_data["my"][horce_id] = max( speed )
                up_speed_index_race_data["my"][horce_id] = max( up_speed )
                pace_speed_index_race_data["my"][horce_id] = max( pace_speed )
                speed_index_race_data["max"] = max( max( speed ), speed_index_race_data["max"] )
                speed_index_race_data["min"] = min( max( speed ), speed_index_race_data["max"] )
                speed_index_race_data["average"] += max( speed )
                speed_index_race_data["count"] += 1
                up_speed_index_race_data["max"] = max( max( up_speed ), up_speed_index_race_data["max"] )
                up_speed_index_race_data["min"] = min( max( up_speed ), up_speed_index_race_data["max"] )
                up_speed_index_race_data["average"] += max( up_speed )
                up_speed_index_race_data["count"] += 1
                pace_speed_index_race_data["max"] = max( max( pace_speed ), pace_speed_index_race_data["max"] )
                pace_speed_index_race_data["min"] = min( max( pace_speed ), pace_speed_index_race_data["max"] )
                pace_speed_index_race_data["average"] += max( pace_speed )
                pace_speed_index_race_data["count"] += 1
            else:
                speed_index_race_data["my"][horce_id] = -100
                up_speed_index_race_data["my"][horce_id] = -100
                pace_speed_index_race_data["my"][horce_id] = -100                
            
            limb_math = lib.limb_search( pd )

            if cd.popular() == 1:
                popular_limb = limb_math
                
            race_limb[limb_math] += 1

        if not time_index_race_data["count"] == 0:
            time_index_race_data["average"] /= time_index_race_data["count"]

        if not speed_index_race_data["count"] == 0:
            speed_index_race_data["average"] /= speed_index_race_data["count"]
            up_speed_index_race_data["average"] /= up_speed_index_race_data["count"]
            pace_speed_index_race_data["average"] /= pace_speed_index_race_data["count"]
        
        for kk in race_data[k].keys():
            horce_id = kk
            current_data, past_data = lib.race_check( horce_data[horce_id],
                                                     year, day, num, race_place_num )#今回と過去のデータに分ける
            cd = lib.current_data( current_data )
            pd = lib.past_data( past_data, current_data )

            if not cd.race_check():
                continue

            #pad = parent_data.main( horce_name, cd )
            current_jockey = jockey_data.data_get( horce_id, cd.birthday(), cd.race_num() )
            key_horce_num = str( int( cd.horce_number() ) )
            key_flame_num = str( int( cd.flame_number() ) )
            count += 1
            
            t_instance = []
            change_data = []
            
            limb_math = lib.limb_search( pd )
            past_limb = lib.past_limb( pd )

            try:
                key = min( corner_horce_body[race_id] )
                first_horce_body = corner_horce_body[race_id][key][key_horce_num]
            except:
                first_horce_body = -1

            key_limb = str( int( limb_math ) )
            
            past_horce_body = past_horce_body_get( corner_horce_body, key_horce_num, pd.race_id_get() )
            
            try:
                ln_body = limb_num_horce_body[key_horce_num][key_limb]["data"]
            except:
                ln_body = -1

            try:
                up3 = sum( first_up3_halon[race_id][key_horce_num] ) / len( first_up3_halon[race_id][key_horce_num] )
            except:
                up3 = -1
                
            father_id = parent_id_data[horce_id]["father"]
            mother_id = parent_id_data[horce_id]["mother"]
            father_data = parent_data_get.main( horce_data, passing_data, father_id, baba_index_data )
            mother_data = parent_data_get.main( horce_data, passing_data, mother_id, baba_index_data )
            
            dm.dn.append( t_instance, race_limb[0], "その他の馬の数" )
            dm.dn.append( t_instance, race_limb[1], "逃げaの馬の数" )
            dm.dn.append( t_instance, race_limb[2], "逃げbの馬の数" )
            dm.dn.append( t_instance, race_limb[3], "先行aの馬の数" )
            dm.dn.append( t_instance, race_limb[4], "先行bの馬の数" )
            dm.dn.append( t_instance, race_limb[5], "差しaの馬の数" )
            dm.dn.append( t_instance, race_limb[6], "差しbの馬の数" )
            dm.dn.append( t_instance, race_limb[7], "追いの馬の数" )
            dm.dn.append( t_instance, race_limb[8], "後方の馬の数" )

            for l in past_limb.keys():
                dm.dn.append( t_instance, len( past_limb[l] ), l )

            #dm.dn.append( t_instance, popular_limb, "一番人気の馬の脚質" )                
            dm.dn.append( t_instance, first_horce_body_encoding["place"][key_place], "場所" )
            dm.dn.append( t_instance, float( key_dist ), "距離" )
            dm.dn.append( t_instance, float( key_kind ), "芝かダート" )
            dm.dn.append( t_instance, float( key_baba ), "馬場" )
            dm.dn.append( t_instance, cd.id_weight(), "馬体重の増減" )
            dm.dn.append( t_instance, cd.burden_weight(), "斤量" )
            dm.dn.append( t_instance, first_horce_body_encoding["horce_number"][key_horce_num], "馬番-target" )
            dm.dn.append( t_instance, first_horce_body_encoding["flame_number"][key_flame_num], "枠番-target" )
            dm.dn.append( t_instance, cd.all_horce_num(), "馬の頭数" )
            dm.dn.append( t_instance, rci_dist[0], "最初の直線に距離" )
            dm.dn.append( t_instance, first_horce_body_encoding["limb"][key_limb]["ave"], "過去データからの予想脚質-target-ave" )
            dm.dn.append( t_instance, first_horce_body_encoding["limb"][key_limb]["std"], "過去データからの予想脚質-target-std" )
            dm.dn.append( t_instance, first_horce_body_encoding["limb"][key_limb]["ave"] * 1.7083006 - 5.3876967, "過去データからの予想脚質-target-ab" )            
            dm.dn.append( t_instance, abs( first_horce_body_encoding["limb"][key_limb]["ave"] - first_horce_body_encoding["limb"][key_limb]["std"] ) * 1.6797943 + 1.5751076, "limbave-limbstd-target-ab" )
            dm.dn.append( t_instance, abs( first_horce_body_encoding["limb"][key_limb]["ave"] - up3 ) * 1.5969952 - 27.99388, "limbave-aveup3-target-ab" )
            dm.dn.append( t_instance, speed_index_race_data["my"][horce_id] , "最大のスピード指数" )
            dm.dn.append( t_instance, speed_index_race_data["my"][horce_id] - speed_index_race_data["max"] , "レース内の最大のスピード指数との差" )
            dm.dn.append( t_instance, speed_index_race_data["my"][horce_id] - speed_index_race_data["min"] , "レース内の最小のスピード指数との差" )
            dm.dn.append( t_instance, speed_index_race_data["my"][horce_id] - speed_index_race_data["average"] , "レース内の平均のスピード指数との差" )
            dm.dn.append( t_instance, up_speed_index_race_data["my"][horce_id] , "最大の上り指数" )
            dm.dn.append( t_instance, up_speed_index_race_data["my"][horce_id] - up_speed_index_race_data["max"] , "レース内の最大の上り指数との差" )
            dm.dn.append( t_instance, up_speed_index_race_data["my"][horce_id] - up_speed_index_race_data["min"] , "レース内の最小の上り指数との差" )
            dm.dn.append( t_instance, up_speed_index_race_data["my"][horce_id] - up_speed_index_race_data["average"] , "レース内の平均の上り指数との差" )
            dm.dn.append( t_instance, pace_speed_index_race_data["my"][horce_id] , "最大のペース指数" )
            dm.dn.append( t_instance, pace_speed_index_race_data["my"][horce_id] - pace_speed_index_race_data["max"] , "レース内の最大のペース指数との差" )
            dm.dn.append( t_instance, pace_speed_index_race_data["my"][horce_id] - pace_speed_index_race_data["min"] , "レース内の最小のペース指数との差" )
            dm.dn.append( t_instance, pace_speed_index_race_data["my"][horce_id] - pace_speed_index_race_data["average"] , "レース内の平均のペース指数との差" )

            dm.dn.append( t_instance, pd.diff_pace_time(), "平均のペース誤差" )
            dm.dn.append( t_instance, pd.diff_pace_passing() , "平均のペースと通過順位" )
            dm.dn.append( t_instance, pd.three_average(), "過去3レースの平均順位" )
            dm.dn.append( t_instance, pd.dist_rank_average(), "過去同じ距離の種類での平均順位" )
            dm.dn.append( t_instance, pd.racekind_rank_average(), "過去同じレース状況での平均順位" )
            dm.dn.append( t_instance, pd.baba_rank_average(), "過去同じ馬場状態での平均順位" )
            dm.dn.append( t_instance, pd.jockey_rank_average(), "過去同じ騎手での平均順位" )
            dm.dn.append( t_instance, pd.three_average(), "複勝率" )
            dm.dn.append( t_instance, pd.two_rate(), "連対率" )
            dm.dn.append( t_instance, pd.get_money(), "獲得賞金" )
            dm.dn.append( t_instance, pd.best_weight(), "ベスト体重と現在の体重の差" )
            dm.dn.append( t_instance, pd.race_interval(), "中週" )
            dm.dn.append( t_instance, pd.average_speed(), "平均速度" )
            dm.dn.append( t_instance, pd.pace_up_check(), "ペースと上りの関係" )
            #dm.dn.append( t_instance, pd.first_passing_rank(), "平均通過順位" )
            dm.dn.append( t_instance, train_index_list[key_horce_num]["a"], "調教ペースの傾き" )
            dm.dn.append( t_instance, train_index_list[key_horce_num]["b"], "調教ペースの切片" )
            dm.dn.append( t_instance, time_index_race_data["my"][horce_id]["max"], "タイム指数の最大" )
            dm.dn.append( t_instance, time_index_race_data["my"][horce_id]["max"] - time_index_race_data["max"], "タイム指数の最大との差" )
            dm.dn.append( t_instance, time_index_race_data["my"][horce_id]["max"] - time_index_race_data["min"], "タイム指数の最小との差" )
            dm.dn.append( t_instance, time_index_race_data["my"][horce_id]["max"] - time_index_race_data["average"], "タイム指数の平均との差" )
            dm.dn.append( t_instance, father_data["rank"], "父親の平均順位" )
            dm.dn.append( t_instance, father_data["two_rate"], "父親の連対率" )
            dm.dn.append( t_instance, father_data["three_rate"], "父親の副賞率" )
            dm.dn.append( t_instance, father_data["average_speed"], "父親の平均速度" )
            dm.dn.append( t_instance, father_data["speed_index"], "父親の最大のスピード指数" )
            dm.dn.append( t_instance, father_data["up_speed_index"], "父親の最大の上りスピード指数" )
            dm.dn.append( t_instance, father_data["pace_speed_index"], "父親の最大のペース指数" )
            dm.dn.append( t_instance, father_data["limb"], "父親の脚質" )
            dm.dn.append( t_instance, mother_data["rank"], "母親の平均順位" )
            dm.dn.append( t_instance, mother_data["two_rate"], "母親の連対率" )
            dm.dn.append( t_instance, mother_data["three_rate"], "母親の副賞率" )
            dm.dn.append( t_instance, mother_data["average_speed"], "母親の平均速度" )
            dm.dn.append( t_instance, mother_data["speed_index"], "母親の最大のスピード指数" )
            dm.dn.append( t_instance, mother_data["up_speed_index"], "母親の最大の上りスピード指数" )
            dm.dn.append( t_instance, mother_data["pace_speed_index"], "母親の最大のペース指数" )
            dm.dn.append( t_instance, mother_data["limb"], "母親の脚質" )

            dm.dn.append( t_instance, current_jockey["all"]["rank"], "騎手の過去の平均順位" )
            dm.dn.append( t_instance, current_jockey["all"]["one"], "騎手の過去のone" )
            dm.dn.append( t_instance, current_jockey["all"]["two"], "騎手の過去のtwo" )
            dm.dn.append( t_instance, current_jockey["all"]["three"], "騎手の過去のthree" )
            dm.dn.append( t_instance, current_jockey["all"]["time"], "騎手の過去のタイム" )
            dm.dn.append( t_instance, current_jockey["all"]["up"], "騎手の過去の上り" )
            dm.dn.append( t_instance, current_jockey["all"]["fhb"], "騎手の過去の平均passing" )
            dm.dn.append( t_instance, current_jockey["100"]["rank"], "騎手の過去の100の平均順位" )
            dm.dn.append( t_instance, current_jockey["100"]["one"], "騎手の過去の100のone" )
            dm.dn.append( t_instance, current_jockey["100"]["two"], "騎手の過去の100のtwo" )
            dm.dn.append( t_instance, current_jockey["100"]["three"], "騎手の過去の100のthree" )
            dm.dn.append( t_instance, current_jockey["100"]["time"], "騎手の過去の100のタイム" )
            dm.dn.append( t_instance, current_jockey["100"]["up"], "騎手の過去の100の上り" )
            dm.dn.append( t_instance, current_jockey["100"]["fhb"], "騎手の過去の100の平均passing" )
            #dm.dn.append( t_instance, past_horce_body, "過去の平均馬身" )
            #dm.dn.append( t_instance, ln_body, "馬番の平均馬身" )
            #win_rate_append( t_instance, win_rate_data, ri_list, key_data )
            #dm.dn.append( t_instance, hb, "最初の馬身" )

            if year == "2020":
                lib.dic_append( simu_data, race_id, {} )
                simu_data[race_id][key_horce_num] = {}
                simu_data[race_id][key_horce_num]["answer"] = first_horce_body
                simu_data[race_id][key_horce_num]["data"] = t_instance

            a_instance = [0] * 20
            rank = cd.rank()
            
            for r in range( 0, len( a_instance ) ):
                a_instance[r] = math.pow( 0.5, int( abs( rank - r ) ) ) * 2
            
            result["answer"].append( first_horce_body )
            result["teacher"].append( t_instance )
            result["year"].append( year )

        if not count + 1 == 0:
            result["query"].append( { "q": count + 1, "year": year } )
            
    for i in range( 0, len( result["answer"] ) ):
        result["answer"][i] = min( max( int( result["answer"][i] ), 0 ), 20 )

    print( len( result["answer"] ) , len( result["teacher"] ) )
    dm.dn.write( "first_horce_body.txt" )
    dm.pickle_upload( "first_horce_body_data.pickle", result )
    dm.pickle_upload( "first_horce_body_simu_data.pickle", simu_data )
    dm.dl.data_clear()
    
    return result, simu_data

