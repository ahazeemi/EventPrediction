import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import Birch
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from scipy.sparse import csr_matrix
import itertools
import csv
import glob
import os
import numpy as np
import scipy
from scipy.sparse import  hstack
from operator import itemgetter
'''
This script performs the final clustering step of forming event chains
It clusters news from filtered_two folder on themes and locations,
and outputs per hour clusters to final folder

Birch Threshold for this script is set to a larger value than the threshold in second_level_clustering.py
'''


def main():

    # parameters
    perform_pca = False
    birch_thresh = 1

    for year in range(2017,2018):
        for month in range(1,2):
            for day in range(1,6):
                for hour in range(0, 3):

                    yearStr = str(year)
                    monthStr = str(month)
                    dayStr = str(day)
                    hourStr = str(hour)

                    if(month < 10):
                        monthStr = '0'+monthStr
                    if (day < 10):
                        dayStr = '0' + dayStr

                    fileName = yearStr+monthStr+dayStr+hourStr+'.csv'
                    file_prefix = yearStr+monthStr+dayStr+hourStr

                    path = r'C:\Users\lenovo\PycharmProjects\FYP\filtered_two'  # use your path
                    all_files = glob.glob(os.path.join(path, fileName))
                    print(all_files)
                    df_from_each_file = (pd.read_csv(f, header=None) for f in all_files)
                    df = pd.concat(df_from_each_file)

                    print(len(df))

                    df.columns = ['record_id', 'date', 'url', 'counts', 'themes', 'locations', 'persons', 'organizations', 'tone']

                    df = df[pd.notnull(df['themes'])]
                    df = df[pd.notnull(df['locations'])]

                    df_locations = pd.DataFrame(df['locations'])

                    row_dict = df.copy(deep=True)
                    row_dict.fillna('',inplace=True)
                    row_dict.index = range(len(row_dict))
                    row_dict = row_dict.to_dict('index') # dictionary that maps row number to row

                    df = df[df.columns[[4]]]
                    df.columns = ['themes']

                    df = pd.DataFrame(df['themes'].str.split(';'))    # splitting themes

                    df_locations = pd.DataFrame(df_locations['locations'].str.split(';')) # splitting locations

                    for row in df_locations.itertuples():
                        for i in range(0,len(row.locations)):
                            try:
                                temp = row.locations[i].split('#')
                                row.locations[i] = temp[4] + '#' + temp[5]    # for retaining only ADM1 Code
                            except:
                                continue

                    df = df[pd.notnull(df['themes'])]
                    for row in df.itertuples():
                        row.themes[:] = [x for x in row.themes if not x.startswith(('CRISISLEX'))]
                        if len(row.themes) == 1 and row.themes[0] == '':
                            row.themes.append('#')
                            row.themes.pop(0)
                        if row.themes[len(row.themes) - 1] == '':
                            row.themes.pop()

                    mlb = MultiLabelBinarizer(sparse_output=True)
                    sparse_themes = mlb.fit_transform(df['themes'])

                    mlb2 = MultiLabelBinarizer(sparse_output=True)
                    sparse_locations = mlb2.fit_transform(df_locations['locations'])

                    df = hstack([sparse_themes.astype(float), sparse_locations.astype(float)])
                    #df = hstack([sparse_locations.astype(float), sparse_persons.astype(float)])

                    # Reducing dimensions through principal component analysis
                    if perform_pca:
                        pca = PCA(n_components=None)
                        df = pd.DataFrame(pca.fit_transform(df))

                    print("Starting clustering")
                    brc = Birch(branching_factor=50, n_clusters=None, threshold=birch_thresh, compute_labels = True)
                    brc.fit(df)
                    predicted_labels = brc.predict(df)

                    clusters = {}
                    n = 0
                    for item in predicted_labels:
                        if item in clusters:
                            clusters[item].append(list((row_dict[n]).values()))   # since row_dict[n] is itself a dictionary
                        else:
                            clusters[item] = [list((row_dict[n]).values())]
                        n += 1

                    with open('final/' + file_prefix + '.txt', 'w', encoding='utf-8') as file:
                        for item in clusters:
                            file.write("\n\nCluster " + str(item) + "\n")
                            for i in range(0, len(clusters[item])):
                                file.write(clusters[item][i][2] + '\n')  # appending url

                    with open('final/' + file_prefix + '.csv', 'w', newline='', encoding='utf-8') as file:
                        writer = csv.writer(file, delimiter=",")
                        for item in clusters:
                            if len(clusters[item]) > 0:
                                clusters[item].sort(key=itemgetter(1))
                                for i in range(0, len(clusters[item])):
                                    writer.writerow(clusters[item][i])
                                writer.writerow('#')
                    return


if __name__ == "__main__":
    main()