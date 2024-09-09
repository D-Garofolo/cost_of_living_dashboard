import pandas as pd
from sklearn.cluster import KMeans
from flask import Flask, jsonify
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS
from flask_cors import CORS
import re
import sys
import numpy
import math

states = ['AL', 'AK', '', 'AZ', 'AR', 'CA', '', 'CO', 'CT', 'DE', '', 'FL', 'GA', '', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', '', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', '', 'WA', 'WV', 'WI', 'WY']
fips = []
with open(r'data/fipscodes.txt', 'r') as f:
    for line in f:
        split_str = re.split(r'\s{8}', line.strip())[:2]
        split_str.append(states[math.floor(int(split_str[0])/1000) - 1])
        fips.append(split_str)

fips_df = pd.DataFrame(fips, columns=['fips', 'county', 'state'])

US_info = pd.read_csv(r'data/US_info_sample.csv')

merged_df = pd.merge(US_info, fips_df, on=['county', 'state'], how='left')
merged_info = pd.DataFrame(merged_df[['state', 'fips', 'family_member_count', 'housing_cost', 'food_cost', 'transportation_cost', 'healthcare_cost', 'other_necessities_cost', 'childcare_cost', 'taxes', 'total_cost', 'median_family_income', 'isMetro']])
merged_info['isMetro'] = merged_info['isMetro'].replace({True: '1', False: '0'})
merged_info['key'] = merged_info['fips'].astype(str) + merged_info['family_member_count'].astype(str)
merged_info[['parent_count', 'child_count', 'empty']] = merged_info['family_member_count'].str.split('[pc]', expand=True)
merged_info['parent_count'] = merged_info['parent_count'].astype(int)
merged_info['child_count'] = merged_info['child_count'].astype(int)
merged_info.drop(columns=['empty'], inplace=True)

full_info = pd.read_csv(r'data/US_info_full.csv')

full_merged_df = pd.merge(full_info, fips_df, on=['county', 'state'], how='left')
full_merged_info = pd.DataFrame(full_merged_df[['state', 'fips', 'family_member_count', 'housing_cost', 'food_cost', 'transportation_cost', 'healthcare_cost', 'other_necessities_cost', 'childcare_cost', 'taxes', 'total_cost', 'median_family_income', 'isMetro']])
full_merged_info['key'] = full_merged_info['fips'].astype(str) + full_merged_info['family_member_count'].astype(str)
full_merged_info['inSubset'] = full_merged_info['key'].isin(merged_info['key'])
full_merged_info['fullIndex'] = full_merged_info.index
full_merged_info['isMetro'] = full_merged_info['isMetro'].replace({True: '1', False: '0'})
full_merged_info[['parent_count', 'child_count', 'empty']] = full_merged_info['family_member_count'].str.split('[pc]', expand=True)
full_merged_info['parent_count'] = full_merged_info['parent_count'].astype(int)
full_merged_info['child_count'] = full_merged_info['child_count'].astype(int)
full_merged_info.drop(columns=['empty'], inplace=True)

key_to_idx_mapping = dict(zip(full_merged_info['key'], full_merged_info['fullIndex']))

merged_info['fullIndex'] = merged_info['key'].map(key_to_idx_mapping)

cluster_info = pd.DataFrame(full_merged_info)
cluster_info.drop(columns=['state', 'fips', 'family_member_count', 'key', 'inSubset', 'fullIndex'], inplace=True)
cluster_info = cluster_info.fillna(0)
cluster_info = StandardScaler().fit_transform(cluster_info)

MSE = []

for i in range(1, 11):
    kmeans_info = KMeans(n_clusters=i,init='random').fit(cluster_info)
    column_name = 'k_means_cluster_' + str(i)
    predict = kmeans_info.predict(cluster_info)
    full_merged_info[column_name] = pd.Series(predict, index= full_merged_info.index)
    MSE.append(kmeans_info.inertia_)

mds_info = pd.DataFrame(merged_info)
mds_info.drop(columns=['state', 'fips', 'family_member_count', 'fullIndex', 'key'], inplace=True)
mds = MDS()
mds_values = mds.fit_transform(mds_info)
merged_info[['mds_x', 'mds_y']] = pd.DataFrame(mds_values, index=merged_info.index)

for i in range(1, 11):
    column_name = 'k_means_cluster_' + str(i)
    merged_info[column_name] = pd.Series(dtype='int')

for i in range(0, 500):
    for j in range(1, 11):
        column_name = 'k_means_cluster_' + str(j)
        merged_info[column_name].iloc[i] = full_merged_info[column_name].iloc[merged_info['fullIndex'].iloc[i]]

full_merged_info.to_csv(r'data/test_full.csv')
merged_info.to_csv(r'data/test_sample.csv')

app = Flask(__name__)
CORS(app)

@app.route('/dashboarddata')
def get_data():
    return jsonify(merged_info.to_json(orient='records'))

@app.route('/fulldata')
def get_full_data():
    return jsonify(full_merged_info.to_json(orient='records'))

if __name__ == '__main__':
    app.run()
