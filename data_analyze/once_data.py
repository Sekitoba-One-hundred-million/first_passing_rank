import math
import copy
import sklearn
from tqdm import tqdm
from mpi4py import MPI
from statistics import stdev

import sekitoba_library as lib
import sekitoba_data_manage as dm

from sekitoba_data_create.stride_ablity import StrideAblity
from sekitoba_data_create.time_index_get import TimeIndexGet
from sekitoba_data_create.jockey_data_get import JockeyData
from sekitoba_data_create.trainer_data_get import TrainerData
from sekitoba_data_create.high_level_data_get import RaceHighLevel

from common.name import Name

data_name = Name()

dm.dl.file_set( "race_data.pickle" )
dm.dl.file_set( "race_info_data.pickle" )
dm.dl.file_set( "horce_data_storage.pickle" )
dm.dl.file_set( "baba_index_data.pickle" )
dm.dl.file_set( "race_day.pickle" )
dm.dl.file_set( "horce_sex_data.pickle" )
dm.dl.file_set( "race_jockey_id_data.pickle" )
dm.dl.file_set( "race_trainer_id_data.pickle" )
dm.dl.file_set( "true_skill_data.pickle" )
dm.dl.file_set( "race_money_data.pickle" )
dm.dl.file_set( "corner_horce_body.pickle" )
dm.dl.file_set( "jockey_judgment_data.pickle" )
dm.dl.file_set( "jockey_judgment_rate_data.pickle" )
dm.dl.file_set( "trainer_judgment_data.pickle" )
dm.dl.file_set( "trainer_judgment_rate_data.pickle" )
dm.dl.file_set( "first_passing_true_skill_data.pickle" )
dm.dl.file_set( "last_passing_true_skill_data.pickle" )
dm.dl.file_set( "predict_train_score.pickle" )
dm.dl.file_set( "predict_pace_data.pickle" )
dm.dl.file_set( "predict_rough_race_data.pickle" )

class OnceData:
    def __init__( self ):
        self.race_data = dm.dl.data_get( "race_data.pickle" )
        self.race_info = dm.dl.data_get( "race_info_data.pickle" )
        self.horce_data = dm.dl.data_get( "horce_data_storage.pickle" )
        self.baba_index_data = dm.dl.data_get( "baba_index_data.pickle" )
        self.race_day = dm.dl.data_get( "race_day.pickle" )
        self.horce_sex_data = dm.dl.data_get( "horce_sex_data.pickle" )
        self.race_jockey_id_data = dm.dl.data_get( "race_jockey_id_data.pickle" )
        self.race_trainer_id_data = dm.dl.data_get( "race_trainer_id_data.pickle" )
        self.true_skill_data = dm.dl.data_get( "true_skill_data.pickle" )
        self.race_money_data = dm.dl.data_get( "race_money_data.pickle" )
        self.corner_horce_body = dm.dl.data_get( "corner_horce_body.pickle" )
        self.first_passing_true_skill_data = dm.dl.data_get( "first_passing_true_skill_data.pickle" )
        self.last_passing_true_skill_data = dm.dl.data_get( "last_passing_true_skill_data.pickle" )
        self.jockey_judgment_data = dm.dl.data_get( "jockey_judgment_data.pickle" )
        self.jockey_judgment_rate_data = dm.dl.data_get( "jockey_judgment_rate_data.pickle" )
        self.trainer_judgment_data = dm.dl.data_get( "trainer_judgment_data.pickle" )
        self.trainer_judgment_rate_data = dm.dl.data_get( "trainer_judgment_rate_data.pickle" )
        self.predict_train_score = dm.dl.data_get( "predict_train_score.pickle" )
        self.predict_pace_data = dm.dl.data_get( "predict_pace_data.pickle" )
        self.predict_rough_race_data = dm.dl.data_get( "predict_rough_race_data.pickle" )

        self.stride_ablity = StrideAblity()
        self.race_high_level = RaceHighLevel()
        self.time_index = TimeIndexGet()
        self.trainer_data = TrainerData()
        self.jockey_data = JockeyData()

        self.data_name_list = []
        self.write_data_list = []
        self.simu_data = {}
        self.jockey_judgement_param_list = [ "limb", "popular", "flame_num", "dist", "kind", "baba", "place" ]
        self.jockey_judgement_rate_param_list = [ "limb", "popular", "flame_num", "dist", "kind", "baba", "place" ]
        self.trainer_judgement_param_list = [ "limb", "popular", "flame_num", "dist", "kind", "baba", "place" ]
        self.trainer_judgement_rate_param_list = [ "limb", "popular", "flame_num", "dist", "kind", "baba", "place" ]
        self.result = { "answer": [], "teacher": [], "query": [], "year": [], "level": [], "diff": [], "horce_body": [] }
        self.data_name_read()

    def data_name_read( self ):
        f = open( "common/list.txt", "r" )
        str_data_list = f.readlines()

        for str_data in str_data_list:
            self.data_name_list.append( str_data.replace( "\n", "" ) )

    def score_write( self ):
        f = open( "common/rank_score_data.txt", "w" )

        for name in self.write_data_list:
            f.write( name + "\n" )

        f.close()

    def data_list_create( self, data_dict ):
        result = []
        write_instance = []
        
        for name in self.data_name_list:
            try:
                result.append( data_dict[name] )
                write_instance.append( name )
            except:
                continue

        if len( self.write_data_list ) == 0:
            self.write_data_list = copy.deepcopy( write_instance )

        return result

    def division( self, score, d ):
        if score < 0:
            score *= -1
            score /= d
            score *= -1
        else:
            score /= d

        return int( score )

    def clear( self ):
        dm.dl.data_clear()
    
    def create( self, k ):
        race_id = lib.id_get( k )
        year = race_id[0:4]
        race_place_num = race_id[4:6]
        day = race_id[9]
        num = race_id[7]
        key_place = str( self.race_info[race_id]["place"] )
        key_dist = str( self.race_info[race_id]["dist"] )
        key_kind = str( self.race_info[race_id]["kind"] )      
        key_baba = str( self.race_info[race_id]["baba"] )
        ymd = { "y": int( year ), "m": self.race_day[race_id]["month"], "d": self.race_day[race_id]["day"] }

        #芝かダートのみ
        if key_kind == "0" or key_kind == "3":
            return

        if not race_id in self.race_money_data:
            return

        if not race_id in self.corner_horce_body:
            return

        predict_pace = 0
        predict_rough_rate = -1

        if race_id in self.predict_pace_data:
            predict_pace = self.predict_pace_data[race_id]

        if race_id in self.predict_rough_race_data:
            predict_rough_rate = self.predict_rough_race_data[race_id]
        
        current_horce_body = self.corner_horce_body[race_id]
        min_corner_key = min( self.corner_horce_body[race_id] )
        key_race_money_class = str( int( lib.money_class_get( self.race_money_data[race_id] ) ) )
        teacher_data = []
        answer_data = []
        answer_horce_body = []
        diff_data = []
        horce_id_list = []
        race_limb = {}
        current_race_data = {}
        current_race_data[data_name.my_limb_count] = { "-1": -1 }

        for name in self.data_name_list:
            if name in current_race_data:
                continue

            current_race_data[name] = []

        escape_limb1_count = 0
        escape_limb2_count = 0
        one_popular_limb = -1
        two_popular_limb = -1
        one_popular_odds = -1
        two_popular_odds = -1
        
        for horce_id in self.race_data[k].keys():
            current_data, past_data = lib.race_check( self.horce_data[horce_id],
                                                     year, day, num, race_place_num )#今回と過去のデータに分ける
            cd = lib.current_data( current_data )
            pd = lib.past_data( past_data, current_data )

            if not cd.race_check():
                continue

            limb_math = lib.limb_search( pd )

            if not limb_math == -1:
                key_limb = str( int( limb_math ) )
                lib.dic_append( current_race_data[data_name.my_limb_count], key_limb, 0 )
                current_race_data[data_name.my_limb_count][key_limb] += 1
                
            race_limb[horce_id] = limb_math

            if limb_math == 1:
                escape_limb1_count += 1

            if limb_math == 2:
                escape_limb2_count += 1

            escape_within_rank = -1
            
            if limb_math == 1 or limb_math == 2:
                escape_within_rank = cd.horce_number()

            jockey_id = ""
            trainer_id = ""

            try:
                jockey_id = self.race_jockey_id_data[race_id][horce_id]
            except:
                pass

            try:
                trainer_id = self.race_trainer_id_data[race_id][horce_id]
            except:
                pass

            horce_true_skill = -1000
            jockey_true_skill = -1000
            trainer_true_skill = -1000
            horce_first_passing_true_skill = -1000
            jockey_first_passing_true_skill = -1000
            trainer_first_passing_true_skill = -1000
            horce_last_passing_true_skill = -1000
            jockey_last_passing_true_skill = -1000

            if race_id in self.true_skill_data["horce"] and \
              horce_id in self.true_skill_data["horce"][race_id]:
                horce_true_skill = self.true_skill_data["horce"][race_id][horce_id]

            if race_id in self.true_skill_data["jockey"] and \
              jockey_id in self.true_skill_data["jockey"][race_id]:
                jockey_true_skill = self.true_skill_data["jockey"][race_id][jockey_id]

            if race_id in self.true_skill_data["trainer"] and \
              trainer_id in self.true_skill_data["trainer"][race_id]:
                trainer_true_skill = self.true_skill_data["trainer"][race_id][trainer_id]
            
            if race_id in self.first_passing_true_skill_data["horce"] and \
              horce_id in self.first_passing_true_skill_data["horce"][race_id]:
                horce_first_passing_true_skill = self.first_passing_true_skill_data["horce"][race_id][horce_id]

            if race_id in self.first_passing_true_skill_data["jockey"] and \
              jockey_id in self.first_passing_true_skill_data["jockey"][race_id]:
                jockey_first_passing_true_skill = self.first_passing_true_skill_data["jockey"][race_id][jockey_id]
                
            if race_id in self.first_passing_true_skill_data["trainer"] and \
              trainer_id in self.first_passing_true_skill_data["trainer"][race_id]:
                trainer_first_passing_true_skill = self.first_passing_true_skill_data["trainer"][race_id][trainer_id]

            if race_id in self.last_passing_true_skill_data["horce"] and \
              horce_id in self.last_passing_true_skill_data["horce"][race_id]:
                horce_last_passing_true_skill = self.last_passing_true_skill_data["horce"][race_id][horce_id]

            if race_id in self.last_passing_true_skill_data["jockey"] and \
              jockey_id in self.last_passing_true_skill_data["jockey"][race_id]:
                jockey_last_passing_true_skill = self.last_passing_true_skill_data["jockey"][race_id][jockey_id]

            past_min_first_horce_body = -1000
            past_max_first_horce_body = -1000
            past_ave_first_horce_body = -1000
            past_std_first_horce_body = -1000
            past_first_horce_body_list = []

            for past_cd in  pd.past_cd_list():
                past_race_id = past_cd.race_id()
                past_key_horce_num = str( int( past_cd.horce_number() ) )

                if past_race_id in self.corner_horce_body:
                    past_min_corner_key = min( self.corner_horce_body[past_race_id] )

                    if past_key_horce_num in self.corner_horce_body[past_race_id][past_min_corner_key]:
                        past_first_horce_body_list.append( self.corner_horce_body[past_race_id][past_min_corner_key][past_key_horce_num] )

            if not len( past_first_horce_body_list ) == 0:
                past_min_first_horce_body = min( past_first_horce_body_list )
                past_max_first_horce_body = max( past_first_horce_body_list )
                past_ave_first_horce_body = sum( past_first_horce_body_list ) / len( past_first_horce_body_list )

                if len( past_first_horce_body_list ) > 1:
                    past_std_first_horce_body = stdev( past_first_horce_body_list )

            popular = cd.popular()
            odds = cd.odds()

            if popular == 1:
                one_popular_limb = limb_math
                one_popular_odds = odds
            elif popular == 2:
                two_popular_limb = limb_math
                two_popular_odds = odds

            train_score = -1000

            if race_id in self.predict_train_score and horce_id in self.predict_train_score[race_id]:
                train_score = self.predict_train_score[race_id][horce_id]

            horce_num = int( cd.horce_number() )
            stride_ablity_data = self.stride_ablity.ablity_create( cd, pd )
                
            current_year = cd.year()
            horce_birth_day = int( horce_id[0:4] )
            age = current_year - horce_birth_day
            current_time_index = self.time_index.main( horce_id, pd.past_day_list() )
            speed, up_speed, pace_speed = pd.speed_index( self.baba_index_data[horce_id] )
            corner_diff_rank_ave = pd.corner_diff_rank()
            
            for stride_data_key in stride_ablity_data.keys():
                for math_key in stride_ablity_data[stride_data_key].keys():
                    current_race_data[stride_data_key+"_"+math_key].append( stride_ablity_data[stride_data_key][math_key] )

            for param in self.jockey_judgement_param_list:
                jockey_judgment = -1000
                
                if race_id in self.jockey_judgment_data and horce_id in self.jockey_judgment_data[race_id]:
                    jockey_judgment = self.jockey_judgment_data[race_id][horce_id][param]
                    
                current_race_data["jockey_judgment_{}".format( param )].append( jockey_judgment )

            current_race_data[data_name.horce_true_skill].append( horce_true_skill )
            current_race_data[data_name.jockey_true_skill].append( jockey_true_skill )
            current_race_data[data_name.trainer_true_skill].append( trainer_true_skill )
            current_race_data[data_name.horce_first_passing_true_skill].append( horce_first_passing_true_skill )
            current_race_data[data_name.jockey_first_passing_true_skill].append( jockey_first_passing_true_skill )
            current_race_data[data_name.trainer_first_passing_true_skill].append( trainer_first_passing_true_skill )
            current_race_data[data_name.horce_last_passing_true_skill].append( horce_last_passing_true_skill )
            current_race_data[data_name.jockey_last_passing_true_skill].append( jockey_last_passing_true_skill )
            current_race_data[data_name.corner_diff_rank_ave].append( corner_diff_rank_ave )
            current_race_data[data_name.speed_index].append( lib.max_check( speed ) + current_time_index["max"] )
            current_race_data[data_name.up_rate].append( pd.up_rate( key_race_money_class ) )
            current_race_data[data_name.burden_weight].append( cd.burden_weight() )
            current_race_data[data_name.age].append( age )
            current_race_data[data_name.escape_within_rank].append( escape_within_rank )
            current_race_data[data_name.past_min_first_horce_body].append( past_min_first_horce_body )
            current_race_data[data_name.past_max_first_horce_body].append( past_max_first_horce_body )
            current_race_data[data_name.past_ave_first_horce_body].append( past_ave_first_horce_body )
            current_race_data[data_name.past_std_first_horce_body].append( past_std_first_horce_body )
            current_race_data[data_name.predict_train_score].append( train_score )
            current_race_data[data_name.first_result_rank_diff].append( pd.first_result_rank_diff() )
            current_race_data[data_name.last_result_rank_diff].append( pd.last_result_rank_diff() )
            current_race_data[data_name.best_first_passing_rank].append( pd.best_first_passing_rank() )
            current_race_data[data_name.best_second_passing_rank].append( pd.best_second_passing_rank() )
            current_race_data[data_name.passing_regression].append( pd.passing_regression() )
            current_race_data[data_name.diff_pace_first_passing].append( pd.diff_pace_first_passing() )
            current_race_data[data_name.diff_pace_time].append( pd.diff_pace_time() )
            current_race_data[data_name.max_time_point].append( pd.max_time_point() )
            horce_id_list.append( horce_id )

        if len( horce_id_list ) < 2:
            return

        current_key_list = []#list( current_race_data.keys() )

        for data_key in current_race_data.keys():
            if not type( current_race_data[data_key] ) is list or \
              len( current_race_data[data_key] ) == 0:
                continue

            current_key_list.append( data_key )

        for data_key in current_key_list:
            current_race_data[data_key+"_index"] = sorted( current_race_data[data_key], reverse = True )
            current_race_data[data_key+"_stand"] = lib.standardization( current_race_data[data_key] )
            current_race_data[data_key+"_devi"] = lib.deviation_value( current_race_data[data_key] )            

        ave_burden_weight = sum( current_race_data[data_name.burden_weight] ) / len( current_race_data[data_name.burden_weight] )
        std_race_horce_true_skill = stdev( current_race_data[data_name.horce_true_skill] )
        std_race_jockey_true_skill = stdev( current_race_data[data_name.jockey_true_skill] )
        std_race_trainer_true_skill = stdev( current_race_data[data_name.trainer_true_skill] )
        std_race_horce_first_passing_true_skill = stdev( current_race_data[data_name.horce_first_passing_true_skill] )
        std_race_jockey_first_passing_true_skill = stdev( current_race_data[data_name.jockey_first_passing_true_skill] )
        std_race_trainer_first_passing_true_skill = stdev( current_race_data[data_name.trainer_first_passing_true_skill] )        
        std_speed_index = stdev( current_race_data[data_name.speed_index] )
        std_up_rate = stdev( current_race_data[data_name.up_rate] )
        std_past_ave_first_horce_body = stdev( current_race_data[data_name.past_ave_first_horce_body] )
        std_past_max_first_horce_body = stdev( current_race_data[data_name.past_max_first_horce_body] )
        std_past_min_first_horce_body = stdev( current_race_data[data_name.past_min_first_horce_body] )

        min_race_horce_true_skill = min( current_race_data[data_name.horce_true_skill] )
        min_race_jockey_true_skill = min( current_race_data[data_name.jockey_true_skill] )
        min_race_trainer_true_skill = min( current_race_data[data_name.trainer_true_skill] )        
        min_race_horce_first_passing_true_skill = min( current_race_data[data_name.horce_first_passing_true_skill] )
        min_race_jockey_first_passing_true_skill = min( current_race_data[data_name.jockey_first_passing_true_skill] )
        min_race_trainer_first_passing_true_skill = min( current_race_data[data_name.trainer_first_passing_true_skill] )

        min_speed_index = min( current_race_data[data_name.speed_index] )
        min_up_rate = min( current_race_data[data_name.up_rate] )
        min_past_ave_first_horce_body = min( current_race_data[data_name.past_ave_first_horce_body] )
        min_past_max_first_horce_body = min( current_race_data[data_name.past_max_first_horce_body] )
        min_past_min_first_horce_body = min( current_race_data[data_name.past_min_first_horce_body] )

        max_race_horce_true_skill = max( current_race_data[data_name.horce_true_skill] )
        max_race_jockey_true_skill = max( current_race_data[data_name.jockey_true_skill] )
        max_race_trainer_true_skill = max( current_race_data[data_name.trainer_true_skill] )
        max_race_horce_first_passing_true_skill = max( current_race_data[data_name.horce_first_passing_true_skill] )
        max_race_jockey_first_passing_true_skill = max( current_race_data[data_name.jockey_first_passing_true_skill] )
        max_race_trainer_first_passing_true_skill = max( current_race_data[data_name.trainer_first_passing_true_skill] )

        max_speed_index = max( current_race_data[data_name.speed_index] )
        max_up_rate = max( current_race_data[data_name.up_rate] )
        max_past_ave_first_horce_body = max( current_race_data[data_name.past_ave_first_horce_body] )
        max_past_max_first_horce_body = max( current_race_data[data_name.past_max_first_horce_body] )
        max_past_min_first_horce_body = max( current_race_data[data_name.past_min_first_horce_body] )

        N = len( current_race_data[data_name.horce_true_skill] )
        ave_race_horce_true_skill = sum( current_race_data[data_name.horce_true_skill] ) / N 
        ave_race_jockey_true_skill = sum( current_race_data[data_name.jockey_true_skill] ) / N
        ave_race_trainer_true_skill = sum( current_race_data[data_name.trainer_true_skill] ) / N 
        ave_race_horce_first_passing_true_skill = sum( current_race_data[data_name.horce_first_passing_true_skill] ) / N 
        ave_race_jockey_first_passing_true_skill = sum( current_race_data[data_name.jockey_first_passing_true_skill] ) / N
        ave_race_trainer_first_passing_true_skill = sum( current_race_data[data_name.trainer_first_passing_true_skill] ) / N

        ave_speed_index = sum( current_race_data[data_name.speed_index] ) / N
        ave_up_rate = sum( current_race_data[data_name.up_rate] ) / N
        ave_past_ave_first_horce_body = sum( current_race_data[data_name.past_ave_first_horce_body] ) / N
        ave_past_max_first_horce_body = sum( current_race_data[data_name.past_max_first_horce_body] ) / N
        ave_past_min_first_horce_body = sum( current_race_data[data_name.past_min_first_horce_body] ) / N

        for count, horce_id in enumerate( horce_id_list ):
            current_data, past_data = lib.race_check( self.horce_data[horce_id],
                                                     year, day, num, race_place_num )#今回と過去のデータに分ける
            cd = lib.current_data( current_data )
            pd = lib.past_data( past_data, current_data )

            if not cd.race_check():
                continue
            
            before_cd = pd.before_cd()
            place_num = int( race_place_num )
            horce_num = int( cd.horce_number() )

            first_passing_rank = -1
            
            try:
                first_passing_rank = int( cd.passing_rank().split( "-" )[0] )
            except:
                pass

            key_horce_num = str( int( horce_num ) )
            min_current_corner_key = min( current_horce_body )

            if not key_horce_num in current_horce_body[min_corner_key]:
                continue

            answer_corner_horce_body = current_horce_body[min_corner_key][key_horce_num]

            before_speed_score = -1000
            before_diff_score = -1000
            before_id_weight_score = -1000
            before_popular = -1000
            before_passing_list = [ -1000, -1000, -1000, -1000 ]
            before_rank = -1000
            up3_standard_value = -1000
            diff_load_weight = -1000
            before_pace_up_diff = -1000
            popular_rank = -1000

            if not before_cd == None:
                before_speed_score = before_cd.speed()
                before_diff_score = max( before_cd.diff(), 0 ) * 10
                before_id_weight_score = self.division( min( max( before_cd.id_weight(), -10 ), 10 ), 2 )
                before_popular = before_cd.popular()
                before_passing_list = before_cd.passing_rank().split( "-" )
                before_rank = before_cd.rank()
                up3 = before_cd.up_time()
                p1, p2 = before_cd.pace()
                up3_standard_value = max( min( ( up3 - p2 ) * 5, 15 ), -10 )
                diff_load_weight = cd.burden_weight() - before_cd.burden_weight()
                popular_rank = abs( before_cd.rank() - before_cd.popular() )

            key_horce_num = str( int( cd.horce_number() ) )
            before_year = int( year ) - 1
            key_before_year = str( int( before_year ) )

            baba = cd.baba_status()
            limb_math = race_limb[horce_id]#lib.limb_search( pd )
            key_limb = str( int( limb_math ) )            
            weight_score = cd.weight() / 10
            escape_within_rank = -1

            if limb_math == 1 or limb_math == 2:
                escape_within_rank = current_race_data[data_name.escape_within_rank].index( horce_num )

            ave_burden_weight_diff = ave_burden_weight - cd.burden_weight()
            burden_weight_score = cd.burden_weight()
            dist_kind_count = pd.dist_kind_count()
            ave_first_passing_rank = pd.first_passing_rank()
            best_weight = pd.best_weight()
            average_speed = pd.average_speed()
            before_continue_not_three_rank = pd.before_continue_not_three_rank()
            
            try:
                before_last_passing_rank = int( before_passing_list[-1] )
            except:
                before_last_passing_rank = 0

            try:
                before_first_passing_rank = int( before_passing_list[0] )
            except:
                before_first_passing_rank = 0

            judgement_data = {}

            for param in self.jockey_judgement_param_list:
                jockey_judgment_rate = { "0": -1, "1": -1, "2": -1 }
                
                if race_id in self.jockey_judgment_rate_data and horce_id in self.jockey_judgment_rate_data[race_id]:
                    jockey_judgment_rate = self.jockey_judgment_rate_data[race_id][horce_id][param]

                for key_class in [ "0", "1", "2" ]:
                    try:
                        judgement_data["jockey_judgment_rate_{}_{}".format( param, key_class )] = jockey_judgment_rate[key_class]
                    except:
                        judgement_data["jockey_judgment_rate_{}_{}".format( param, key_class )] = -1

            for param in self.trainer_judgement_param_list:
                trainer_judgment = -1000
                
                if race_id in self.trainer_judgment_data and horce_id in self.trainer_judgment_data[race_id]:
                    trainer_judgment = self.trainer_judgment_data[race_id][horce_id][param]
                    
                judgement_data["trainer_judgment_{}".format( param )] = trainer_judgment

            t_instance = {}
            t_instance[data_name.age] = age
            t_instance[data_name.all_horce_num] = cd.all_horce_num()
            t_instance[data_name.ave_burden_weight_diff] = ave_burden_weight_diff
            t_instance[data_name.ave_first_passing_rank] = ave_first_passing_rank
            t_instance[data_name.ave_past_ave_first_horce_body] = ave_past_ave_first_horce_body - past_ave_first_horce_body
            t_instance[data_name.ave_past_max_first_horce_body] = ave_past_max_first_horce_body - past_max_first_horce_body
            t_instance[data_name.ave_past_min_first_horce_body] = ave_past_min_first_horce_body - past_min_first_horce_body
            t_instance[data_name.ave_race_horce_true_skill] = \
              ave_race_horce_true_skill - current_race_data[data_name.horce_true_skill][count]
            t_instance[data_name.ave_race_jockey_true_skill] = \
              ave_race_jockey_true_skill - current_race_data[data_name.jockey_true_skill][count]
            t_instance[data_name.ave_race_trainer_true_skill] = \
              ave_race_trainer_true_skill - current_race_data[data_name.trainer_true_skill][count]
            t_instance[data_name.ave_race_horce_first_passing_true_skill] = \
              ave_race_horce_first_passing_true_skill - current_race_data[data_name.horce_first_passing_true_skill][count]
            t_instance[data_name.ave_race_jockey_first_passing_true_skill] = \
              ave_race_jockey_first_passing_true_skill - current_race_data[data_name.jockey_first_passing_true_skill][count]
            t_instance[data_name.ave_race_trainer_first_passing_true_skill] = \
              ave_race_trainer_first_passing_true_skill - current_race_data[data_name.trainer_first_passing_true_skill][count]
            t_instance[data_name.ave_speed_index] = ave_speed_index - current_race_data[data_name.speed_index][count]
            t_instance[data_name.ave_up_rate] = ave_up_rate - current_race_data[data_name.up_rate][count]
            t_instance[data_name.average_speed] = average_speed            
            t_instance[data_name.baba] = cd.baba_status()
            t_instance[data_name.before_continue_not_three_rank] = before_continue_not_three_rank     
            t_instance[data_name.before_diff] = before_diff_score
            t_instance[data_name.before_first_passing_rank] = before_first_passing_rank
            t_instance[data_name.before_id_weight] = before_id_weight_score
            t_instance[data_name.before_last_passing_rank] = before_last_passing_rank
            t_instance[data_name.before_popular] = before_popular
            t_instance[data_name.before_rank] = before_rank
            t_instance[data_name.before_speed] = before_speed_score
            t_instance[data_name.best_weight] = best_weight
            t_instance[data_name.burden_weight] = burden_weight_score
            t_instance[data_name.diff_load_weight] = diff_load_weight
            t_instance[data_name.dist_kind] = cd.dist_kind()
            t_instance[data_name.dist_kind_count] = dist_kind_count
            t_instance[data_name.escape_limb1_count] = escape_limb1_count
            t_instance[data_name.escape_limb2_count] = escape_limb2_count
            t_instance[data_name.escape_within_rank] = escape_within_rank
            t_instance[data_name.high_level_score] = self.race_high_level.data_get( cd, pd, ymd )
            t_instance[data_name.horce_num] = cd.horce_number()
            t_instance[data_name.horce_sex] = self.horce_sex_data[horce_id]
            t_instance[data_name.jockey_rank] = self.jockey_data.rank( race_id, horce_id )
            t_instance[data_name.jockey_year_rank] = self.jockey_data.year_rank( race_id, horce_id, key_before_year )
            t_instance[data_name.level_score] = pd.level_score()
            t_instance[data_name.limb] = limb_math
            t_instance[data_name.match_rank] = pd.match_rank()
            t_instance[data_name.max_past_ave_first_horce_body] = \
              max_past_ave_first_horce_body - current_race_data[data_name.past_ave_first_horce_body][count]
            t_instance[data_name.max_past_max_first_horce_body] = \
              max_past_max_first_horce_body - current_race_data[data_name.past_max_first_horce_body][count]
            t_instance[data_name.max_past_min_first_horce_body] = \
              max_past_min_first_horce_body - current_race_data[data_name.past_min_first_horce_body][count]
            t_instance[data_name.max_race_horce_first_passing_true_skill] = \
              max_race_horce_first_passing_true_skill - current_race_data[data_name.horce_first_passing_true_skill][count]
            t_instance[data_name.max_race_horce_true_skill] = \
              max_race_horce_true_skill - current_race_data[data_name.horce_true_skill][count]
            t_instance[data_name.max_race_jockey_first_passing_true_skill] = \
              max_race_jockey_first_passing_true_skill - current_race_data[data_name.jockey_first_passing_true_skill][count]
            t_instance[data_name.max_race_jockey_true_skill] = \
              max_race_jockey_true_skill - current_race_data[data_name.jockey_true_skill][count]
            t_instance[data_name.max_race_trainer_first_passing_true_skill] = \
              max_race_trainer_first_passing_true_skill - current_race_data[data_name.trainer_first_passing_true_skill][count]
            t_instance[data_name.max_race_trainer_true_skill] = \
              max_race_trainer_true_skill - current_race_data[data_name.trainer_true_skill][count]
            t_instance[data_name.max_speed_index] = max_speed_index - current_race_data[data_name.speed_index][count]
            t_instance[data_name.max_up_rate] = max_up_rate - current_race_data[data_name.up_rate][count]
            t_instance[data_name.min_past_ave_first_horce_body] = \
              min_past_ave_first_horce_body - current_race_data[data_name.past_ave_first_horce_body][count]
            t_instance[data_name.min_past_max_first_horce_body] = \
              min_past_max_first_horce_body - current_race_data[data_name.past_max_first_horce_body][count]
            t_instance[data_name.min_past_min_first_horce_body] = \
              min_past_min_first_horce_body - current_race_data[data_name.past_min_first_horce_body][count]
            t_instance[data_name.min_race_horce_first_passing_true_skill] = \
              min_race_horce_first_passing_true_skill - current_race_data[data_name.horce_first_passing_true_skill][count]
            t_instance[data_name.min_race_horce_true_skill] = \
              min_race_horce_true_skill - current_race_data[data_name.horce_true_skill][count]
            t_instance[data_name.min_race_jockey_first_passing_true_skill] = \
              min_race_jockey_first_passing_true_skill - current_race_data[data_name.jockey_first_passing_true_skill][count]
            t_instance[data_name.min_race_jockey_true_skill] = \
              min_race_jockey_true_skill - current_race_data[data_name.jockey_true_skill][count]
            t_instance[data_name.min_race_trainer_first_passing_true_skill] = \
              min_race_trainer_first_passing_true_skill - current_race_data[data_name.trainer_first_passing_true_skill][count]
            t_instance[data_name.min_race_trainer_true_skill] = \
              min_race_trainer_true_skill - current_race_data[data_name.trainer_true_skill][count]
            t_instance[data_name.min_speed_index] = min_speed_index - current_race_data[data_name.speed_index][count]
            t_instance[data_name.min_up_rate] = min_up_rate - current_race_data[data_name.up_rate][count]
            t_instance[data_name.money] = pd.get_money()
            t_instance[data_name.my_limb_count] = current_race_data[data_name.my_limb_count][key_limb]
            t_instance[data_name.odds] = cd.odds()
            t_instance[data_name.one_popular_limb] = one_popular_limb
            t_instance[data_name.one_popular_odds] = one_popular_odds
            t_instance[data_name.one_rate] = pd.one_rate()
            t_instance[data_name.pace_up] = pd.pace_up_check()
            t_instance[data_name.past_std_first_horce_body] = past_std_first_horce_body
            t_instance[data_name.place] = place_num
            t_instance[data_name.popular] = cd.popular()
            t_instance[data_name.popular_rank] = popular_rank
            t_instance[data_name.predict_pace] = predict_pace
            t_instance[data_name.std_past_ave_first_horce_body] = std_past_ave_first_horce_body
            t_instance[data_name.std_past_max_first_horce_body] = std_past_max_first_horce_body
            t_instance[data_name.std_past_min_first_horce_body] = std_past_min_first_horce_body
            t_instance[data_name.std_race_horce_first_passing_true_skill] = std_race_horce_first_passing_true_skill
            t_instance[data_name.std_race_horce_true_skill] = std_race_horce_true_skill
            t_instance[data_name.std_race_jockey_first_passing_true_skill] = std_race_jockey_first_passing_true_skill
            t_instance[data_name.std_race_jockey_true_skill] = std_race_jockey_true_skill
            t_instance[data_name.std_race_trainer_first_passing_true_skill] = std_race_trainer_first_passing_true_skill
            t_instance[data_name.std_race_trainer_true_skill] = std_race_trainer_true_skill
            t_instance[data_name.std_speed_index] = std_speed_index
            t_instance[data_name.std_up_rate] = std_up_rate
            t_instance[data_name.three_average] = pd.three_average()
            t_instance[data_name.three_difference] = pd.three_difference()
            t_instance[data_name.three_rate] = pd.three_rate()
            t_instance[data_name.trainer_rank] = self.trainer_data.rank( race_id, horce_id )
            t_instance[data_name.two_popular_limb] = two_popular_limb
            t_instance[data_name.two_popular_odds] = two_popular_odds
            t_instance[data_name.two_rate] = pd.two_rate()
            t_instance[data_name.up3_standard_value] = up3_standard_value
            t_instance[data_name.weather] = cd.weather()
            t_instance[data_name.weight] = weight_score
            t_instance[data_name.speed_index] = current_race_data[data_name.speed_index][count]
            t_instance[data_name.speed_index_index] = \
              current_race_data[data_name.speed_index_index].index( current_race_data[data_name.speed_index][count] )
            t_instance[data_name.speed_index_stand] = current_race_data[data_name.speed_index_stand][count]
            t_instance[data_name.predict_rough_rate] = predict_rough_rate

            str_index = "_index"
            for data_key in current_race_data.keys():
                if len( current_race_data[data_key] ) == 0 or \
                  data_key in t_instance:
                    continue

                if str_index in data_key:
                    name = data_key.replace( str_index, "" )

                    if name in current_race_data:
                        t_instance[data_key] = current_race_data[data_key].index( current_race_data[name][count] )
                else:
                    t_instance[data_key] = current_race_data[data_key][count]

            for judge_key in judgement_data.keys():
                t_instance[judge_key] = judgement_data[judge_key]
                    
            t_list = self.data_list_create( t_instance )

            lib.dic_append( self.simu_data, race_id, {} )
            self.simu_data[race_id][horce_id] = {}
            self.simu_data[race_id][horce_id]["data"] = t_list
            self.simu_data[race_id][horce_id]["answer"] = { "first_passing_rank": first_passing_rank,
                                                           "odds": cd.odds(),
                                                           "popular": cd.popular(),
                                                           "horce_num": cd.horce_number() }

            answer_horce_body.append( answer_corner_horce_body )
            answer_data.append( first_passing_rank )
            teacher_data.append( t_list )
            #diff_data.append( cd.diff() )

        if not len( answer_data ) == 0:
            self.result["answer"].append( answer_data )
            self.result["teacher"].append( teacher_data )
            self.result["year"].append( year )
            self.result["horce_body"].append( answer_horce_body )
            self.result["query"].append( { "q": len( answer_data ), "year": year } )
