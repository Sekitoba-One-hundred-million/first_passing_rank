import math

import sekitoba_library as lib
import sekitoba_data_manage as dm

dm.dl.file_set( "train_time_data.pickle" )

class train_index_get:
    def __init__( self ):
        self.train_time_data = dm.dl.data_get( "train_time_data.pickle" )

    def train_index_create( self, race_id, key_horce_num ):
        result = {}
        result["a"] = 0
        result["b"] = 0
        
        time = 0
        count = 0

        try:
            wrap_time = self.train_time_data[race_id][key_horce_num]["wrap"]
        except:
            return result

        try:
            a, b = lib.regression_line( wrap_time )
        except:
            return result
            
        result["a"] = a
        result["b"] = b

        return result

    def main( self, race_data, horce_data, race_id ):
        result = {}
        fail_dic = { "time": 0, "a": 0, "b": 0 }
        t_instance = []
        year = race_id[0:4]
        race_place_num = race_id[4:6]
        day = race_id[9]
        num = race_id[7]
        train_index_data = []

        for k in race_data.keys():
            horce_id = k
            current_data, past_data = lib.race_check( horce_data[horce_id],
                                                      year, day, num, race_place_num )#今回と過去のデータに分ける

            cd = lib.current_data( current_data )
            pd = lib.past_data( past_data, current_data )

            if not cd.race_check():
                continue
    
            key_horce_num = str( int( cd.horce_number() ) )
            result[key_horce_num] = self.train_index_create( race_id, key_horce_num )

        return result
