import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import functools
import glob
import os
import math
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import Birch
from sklearn.decomposition import PCA
import itertools
import csv

'''
This script performs second level clustering for redundancy removal
It clusters news from filtered_one folder on themes and locations,
retains one news from each cluster,
and outputs per hour file to filtered_two folder

Birch Threshold for this script is set to a slightly larger value than the threshold in first_level_clustering.py
'''


def main():

    # parameters
    write_whole_cluster = True
    perform_pca = False
    birch_thresh = 2.2

    for year in range(2017,2018):
        for month in range(1,2):
            for day in range(1,6):
                for hour in range(0,24):

                    yearStr = str(year)
                    monthStr = str(month)
                    dayStr = str(day)
                    hourStr = str(hour)

                    if(month < 10):
                        monthStr = '0'+monthStr
                    if (day < 10):
                        dayStr = '0' + dayStr
                    if (hour < 10):
                        hourStr = '0' + hourStr

                    fileName = yearStr+monthStr+dayStr+hourStr+'.csv'
                    file_txt = yearStr+monthStr+dayStr+hourStr+'.txt'
                    df = pd.read_csv('filtered_one/'+fileName,header=None)

                    df.columns = ['record_id', 'date', 'url', 'counts', 'themes', 'locations', 'persons', 'organizations', 'tone']

                    row_dict = {}

                    df = df[pd.notnull(df['themes'])]
                    df = df[pd.notnull(df['locations'])]

                    df_locations = pd.DataFrame(df['locations'])
                    #df_persons = pd.DataFrame(df['persons'])

                    # i = 0
                    # for index, row in df.iterrows():
                    #     # if(pd.isnull(row['e'])==False):
                    #     #     row_dict[i]=row['url']
                    #     #     i+=1
                    #     # else:
                    #     #     print(row['a'])
                    #     row_dict[i] = row['url']
                    #     i += 1

                    row_dict = df.copy(deep=True)
                    row_dict.fillna('',inplace=True)
                    row_dict.index = range(len(row_dict))
                    row_dict = row_dict.to_dict('index') # dictionary that maps row number to row

                    df = df[df.columns[[4]]]
                    df.columns = ['themes']

                    #df['themes'].dropna()

                    df = pd.DataFrame(df['themes'].str.split(';'))    # splitting themes

                    #df_persons = pd.DataFrame(df_persons['persons'].str.split(';'))  # splitting persons

                    df_locations = pd.DataFrame(df_locations['locations'].str.split(';')) # splitting locations

                    for row in df_locations.itertuples():
                        # print(row.locations)
                        for i in range(0,len(row.locations)):
                            try:
                                row.locations[i] = (row.locations[i].split('#'))[3]    # for retaining only ADM1 Code
                                # print(row.locations[i])
                            except:
                                continue
                        merged = list(itertools.chain(*row.locations))
                        #print('Hello',merged)
                        #print(df_locations.loc[row.Index, 'locations'])
                        df_locations.loc[row.Index, 'locations'] = merged

                    #df_locations = df_locations[pd.notnull(df_locations['locations'])]

                    df = df[pd.notnull(df['themes'])]

                    #df['a'] = df['a'].apply(lambda x: x.pop())

                    # one hot encoding of themes
                    # print("hello")
                    mlb = MultiLabelBinarizer()
                    df = pd.DataFrame(mlb.fit_transform(df['themes']),columns=mlb.classes_,index=df.index)

                    # one hot encoding of locations
                    mlb2 = MultiLabelBinarizer()
                    df_locations = pd.DataFrame(mlb2.fit_transform(df_locations['locations']), columns=mlb2.classes_, index=df_locations.index)

                    #mlb3 = MultiLabelBinarizer()
                    #df_persons = pd.DataFrame(mlb3.fit_transform(df_persons['persons']), columns=mlb3.classes_,index=df_persons.index)

                    df=df.join(df_locations)
                    #df = df.join(df_persons)

                    # Reducing dimensions through principal component analysis

                    if perform_pca:
                        pca = PCA(n_components=None)
                        df = pd.DataFrame(pca.fit_transform(df))


                    # df.to_csv('one_hot_encoded_pca.csv', sep=',')
                    # print("hello1")
                    # return


                    print("Starting clustering")
                    brc = Birch(branching_factor=50, n_clusters=None, threshold=birch_thresh, compute_labels = True)
                    #brc.fit(df)
                    predicted_labels = brc.fit_predict(df)
                    # print(predicted_labels)

                    clusters = {}
                    n = 0
                    for item in predicted_labels:
                        # print("hello")
                        if item in clusters:
                            clusters[item].append(list((row_dict[n]).values()))   # since row_dict[n] is itself a dictionary
                        else:
                            clusters[item] = [list((row_dict[n]).values())]
                        n += 1

                    if write_whole_cluster:
                        with open('filtered_two/'+file_txt, 'w', encoding='utf-8') as file:
                            for item in clusters:
                                file.write("\n\nCluster "+str(item)+"\n")
                                # print("Cluster ", item)
                                for i in range(0,len(clusters[item])):
                                    # print(clusters[item][i][2])
                                    file.write(clusters[item][i][2]+'\n')           # appending url
                    else:
                        with open('filtered_two/'+fileName, 'w',newline='', encoding='utf-8') as file:
                            writer = csv.writer(file, delimiter=",")
                            for item in clusters:
                                # print("Cluster ", item)
                                writer.writerow(clusters[item][0])


if __name__ == "__main__":
    main()