import pandas as pd
import sqlite3
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity, sigmoid_kernel
import json

PATH_TO_DB = 'data.db'
REQ_COLS = [ 'front_wheel_size_in',
 'sae_net_horsepower_at_rpm',
 'displacement',
 'length_overall_in',
 'gas_mileage',
 'engine_type',
 'drivetrain',
 'passenger_capacity',
 'passenger_doors',
 ]
MAPPER = {0: 'FWD', 1:'AWD', 2:'RWD', 3:'4WD'}

class RecommendationEngine(object):

    def __init__(self):
        print('Recommendation Engine Instance Created')
        self.conn = sqlite3.connect(PATH_TO_DB)
        self.data = pd.read_sql("SELECT * FROM 'car_meta_data'", con=self.conn)
        self.out_list = []
        self.out = None
        self.conn.close()
        self.sc = StandardScaler()
        print('Connected to database !')

    def recommend(self, input_value):
        self.input_value = input_value
        if self.input_value in list(self.data['label2']): 
            self.car_type = self.data[self.data['label2'] == self.input_value]['body_style'].to_list()[0]
            self.recommend_df = self.data[self.data['body_style'] == self.car_type]
            self.label = self.recommend_df['label2'].to_list()
            self.recommend_df = self.recommend_df[REQ_COLS]
            self.recommend_df = self.sc.fit_transform(self.recommend_df)
            self.recommend_df = pd.DataFrame(cosine_similarity(self.recommend_df), columns=self.label, index=self.label)
            self.results = dict(self.recommend_df[self.input_value])
            self.results = list({k:v for k,v in sorted(self.results.items(), key=lambda item: item[1], reverse=True)})[:6]
            del self.recommend_df
            # self.data = self.data[self.data['label2'].isin(self.results)]
            # self.data = self.data.reset_index()
            # self.data.drop(['index','label', 'label2'], axis=1, inplace=True)
            # self.data['id'] = self.data.index
            # print(self.data)
            # print(self.results)
            for car in self.results:
                # print(car)
                self.out_list.append(self.data[self.data['label2'] == car])

            self.out = pd.concat(self.out_list)
            # self.data = self.data[self.data['label2'] == self.results[0]]
            self.out['drivetrain'] = self.out['drivetrain'].map(MAPPER)

            # print(self.out)
            return self.out.to_dict(orient='records')

    def getall(self):
        return self.data.to_json(orient='records', index=True)
