import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import Birch
from sklearn.decomposition import PCA
import itertools
import csv
import glob
import os
from scipy.sparse import  hstack

'''
This script performs the final clustering step of forming event chains
It clusters news from filtered_two folder on themes and locations,
and outputs per hour clusters to final folder

Birch Threshold for this script is set to a larger value than the threshold in second_level_clustering.py
'''


def main():

    # parameters
    perform_pca = False
    birch_thresh = 3.2

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

                    fileName = yearStr+monthStr+dayStr+hourStr+'*.csv'
                    file_txt = yearStr+monthStr+dayStr+hourStr+'.txt'

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

                    #df_persons = pd.DataFrame(df_persons['persons'].str.split(';'))  # splitting persons

                    df_locations = pd.DataFrame(df_locations['locations'].str.split(';')) # splitting locations

                    for row in df_locations.itertuples():
                        for i in range(0,len(row.locations)):
                            try:
                                row.locations[i] = (row.locations[i].split('#'))[3]    # for retaining only ADM1 Code
                            except:
                                continue
                        #merged = list(itertools.chain(*row.locations))
                        #print(merged)

                        #df_locations.loc[row.Index, 'locations'] = merged

                    df = df[pd.notnull(df['themes'])]

                    # one hot encoding of themes
                    '''mlb = MultiLabelBinarizer()
                    df = pd.DataFrame(mlb.fit_transform(df['themes']),columns=mlb.classes_,index=df.index)

                    # one hot encoding of locations
                    mlb2 = MultiLabelBinarizer()
                    df_locations = pd.DataFrame(mlb2.fit_transform(df_locations['locations']), columns=mlb2.classes_, index=df_locations.index)

                    #mlb3 = MultiLabelBinarizer()
                    #df_persons = pd.DataFrame(mlb3.fit_transform(df_persons['persons']), columns=mlb3.classes_,index=df_persons.index)

                    df=df.join(df_locations)'''
                    #df = df.join(df_persons)

                    #df_locations.to_csv('temp2.csv', sep=',', index=0)

                    mlb = MultiLabelBinarizer(sparse_output=True)
                    sparse_themes = mlb.fit_transform(df['themes'])

                    mlb2 = MultiLabelBinarizer(sparse_output=True)
                    sparse_locations = mlb2.fit_transform(df_locations['locations'])

                    #df = df.join(df_locations['locations'])

                    #dfa = pd.DataFrame(df['themes'].tolist())
                    #dfa.to_csv('temp2.csv', sep=',', index=0)
                    #print(dfa.shape)

                    df = hstack([sparse_themes, sparse_locations])

                    #mlb = MultiLabelBinarizer(sparse_output=True)

                    #df = mlb.fit_transform(df)
                    #sparse_size = (sparse_dataset.data.nbytes + sparse_dataset.indptr.nbytes + sparse_dataset.indices.nbytes) / 1e6
                    #print(sparse_size)

                    # df = pd.DataFrame(mlb.fit_transform(df['themes']), columns=mlb.classes_, index=df.index)
                    # print(df.info())


                    # Reducing dimensions through principal component analysis
                    if perform_pca:
                        pca = PCA(n_components=None)
                        df = pd.DataFrame(pca.fit_transform(df))

                    print("Starting clustering")
                    brc = Birch(branching_factor=50, n_clusters=None, threshold=birch_thresh, compute_labels = True)
                    brc.fit(df)
                    predicted_labels = brc.predict(df)

                    #predicted_labels = brc.fit_predict(df)

                    clusters = {}
                    n = 0
                    for item in predicted_labels:
                        if item in clusters:
                            clusters[item].append(list((row_dict[n]).values()))   # since row_dict[n] is itself a dictionary
                        else:
                            clusters[item] = [list((row_dict[n]).values())]
                        n += 1

                    with open('final/a' + file_txt, 'w', encoding='utf-8') as file:
                        for item in clusters:
                            file.write("\n\nCluster " + str(item) + "\n")
                            for i in range(0, len(clusters[item])):
                                file.write(clusters[item][i][2] + '\n')  # appending url


if __name__ == "__main__":
    main()