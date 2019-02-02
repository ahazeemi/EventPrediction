
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import Birch
from sklearn.decomposition import PCA
import csv
from scipy.sparse import hstack
from sklearn import metrics
import numpy as np
from scipy.misc import comb


'''
This script evaluates the performance of clustering for redundancy removal
It clusters news from GDELT GKG on themes (at first level),
further clusters them on locations and counts (second level),
and outputs Precision, Recall, F1 Measure, NMI and Rand Index of the clustering
'''

# There is a comb function for Python which does 'n choose k'
# only you can't apply it to an array right away
# So here we vectorize it...


def myComb(a,b):
  return comb(a,b,exact=True)


vComb = np.vectorize(myComb)


def get_tp_fp_tn_fn(cooccurrence_matrix):
  tp_plus_fp = vComb(cooccurrence_matrix.sum(0, dtype=int),2).sum()
  tp_plus_fn = vComb(cooccurrence_matrix.sum(1, dtype=int),2).sum()
  tp = vComb(cooccurrence_matrix.astype(int), 2).sum()
  fp = tp_plus_fp - tp
  fn = tp_plus_fn - tp
  tn = comb(cooccurrence_matrix.sum(), 2) - tp - fp - fn

  return [tp, fp, tn, fn]


def precision_recall_fmeasure(cooccurrence_matrix):
    tp, fp, tn, fn = get_tp_fp_tn_fn(cooccurrence_matrix)

    # print ("TP: %d, FP: %d, TN: %d, FN: %d" % (tp, fp, tn, fn))

    # Print the measures:
    rand_index = (float(tp + tn) / (tp + fp + fn + tn))
    # print ("Rand index: %f" % rand_index)

    precision = float(tp) / (tp + fp)
    recall = float(tp) / (tp + fn)
    f1 = ((2.0 * precision * recall) / (precision + recall))

    # print ("Precision : %f" % precision)
    # print ("Recall    : %f" % recall)
    # print ("F1        : %f" % f1)

    return rand_index,precision,recall,f1


def main():

    # parameters
    write_whole_cluster = False
    perform_pca = False
    birch_thresh = 2.0
    count_thresh = 0.1

    eval_file_names = ['filtered_eval_one_event.csv', 'filtered_eval_three_event.csv', 'filtered_eval_five_event.csv']
    annotated_file_names = ['annotated_one_event.txt', 'annotated_three_event.txt', 'annotated_five_event.txt']

    for m in range(0,len(eval_file_names)):
        fileName = eval_file_names[m]
        file_prefix = 'output'
        print(fileName)

        for birch_thresh in np.arange(0.0, 4.1, 0.2):
            for count_thresh in np.arange(0.1, 1.1, 0.1):
                '''for i in range(1,179):
                    if(i not in temp):
                        print(i)
                '''

                df = pd.read_csv(fileName, header=None, encoding='latin-1')

                df.columns = ['record_id', 'date', 'url', 'counts', 'themes', 'locations', 'persons', 'organizations', 'tone']

                # Retaining only those news which have non-null themes and locations
                df = df[pd.notnull(df['themes'])]
                df = df[pd.notnull(df['locations'])]

                df_locations = pd.DataFrame(df['locations'])

                # Reading actual class labels assigned by expert human assessor
                class_labels = [None] * len(df)
                temp = {}
                with open(annotated_file_names[m], "r") as ins:
                    label = 1
                    for line in ins:
                        line = line.strip()
                        if line.startswith("#"):
                            continue
                        if line:
                            line = line.split(',')
                            # print(line)
                            for item in line:
                                class_labels[int(item) - 1]= label
                                temp[int(item)] = True
                            label += 1

                row_dict = df.copy(deep=True)
                row_dict.fillna('', inplace=True)
                row_dict.index = range(len(row_dict))
                row_dict = row_dict.to_dict('index')  # dictionary that maps row number to row

                identifier_dict = {}   # dictionary that maps GKG Record Id to Row Number
                i = 0
                for index, row in df.iterrows():
                    identifier_dict[row['record_id']] = i
                    i += 1

                df = df[df.columns[[4]]]
                df.columns = ['themes']

                df = pd.DataFrame(df['themes'].str.split(';'))  # splitting themes

                df_locations = pd.DataFrame(df_locations['locations'].str.split(';'))  # splitting locations

                for row in df_locations.itertuples():
                    for i in range(0, len(row.locations)):
                        try:
                            row.locations[i] = (row.locations[i].split('#'))[3]  # for retaining only ADM1 Code
                        except:
                            continue
                    # merged = list(itertools.chain(*row.locations))
                    # df_locations.loc[row.Index, 'locations'] = merged

                df = df[pd.notnull(df['themes'])]

                mlb = MultiLabelBinarizer(sparse_output=True)
                sparse_themes = mlb.fit_transform(df['themes'])
                df = sparse_themes

                # df = sparse_locations

                # Reducing dimensions through principal component analysis
                if perform_pca:
                    pca = PCA(n_components=None)
                    df = pd.DataFrame(pca.fit_transform(df))

                #print("Starting clustering")
                brc = Birch(branching_factor=50, n_clusters=None, threshold=birch_thresh, compute_labels=True)
                predicted_labels = brc.fit_predict(df)

                clusters = {}
                n = 0

                for item in predicted_labels:
                    if item in clusters:
                        clusters[item].append(list((row_dict[n]).values()))  # since row_dict[n] is itself a dictionary
                    else:
                        clusters[item] = [list((row_dict[n]).values())]
                    n += 1

                # clustering within each cluster, on counts
                count_clusters = {}  # dictionary which maps original_cluster_key to new clusters within that cluster
                for item in clusters:
                    count_clusters[item] = {}
                    cluster_df = pd.DataFrame(clusters[item])
                    cluster_row_dict = cluster_df.copy(deep=True)
                    cluster_row_dict.fillna('', inplace=True)
                    cluster_row_dict.index = range(len(cluster_row_dict))
                    cluster_row_dict = cluster_row_dict.to_dict('index')

                    df_counts = pd.DataFrame(cluster_df[cluster_df.columns[[3]]])
                    df_counts.columns = ['counts']
                    df_counts = pd.DataFrame(df_counts['counts'].str.split(';'))  # splitting counts

                    df_locations = pd.DataFrame(cluster_df[cluster_df.columns[[5]]])
                    df_locations.columns = ['locations']
                    df_locations = pd.DataFrame(df_locations['locations'].str.split(';'))

                    for row in df_locations.itertuples():
                        for i in range(0, len(row.locations)):
                            try:
                                row.locations[i] = (row.locations[i].split('#'))[3]  # for retaining only ADM1 Code
                            except:
                                continue

                    for row in df_counts.itertuples():
                        for i in range(0, len(row.counts)):
                            try:
                                temp_list = row.counts[i].split('#')
                                row.counts[i] = temp_list[0] + '#' + temp_list[1] + '#' + temp_list[5] # for retaining only COUNT_TYPE and QUANTITY and LOCATION ADM1 Code
                            except:
                                continue
                        if len(row.counts) == 1 and row.counts[0] == '':
                            row.counts.append('#')  # so that news with no counts are clustered together
                            row.counts.pop(0)

                        if row.counts[len(row.counts) - 1] == '':
                            row.counts.pop()

                        row.counts[:] = [x for x in row.counts if not x.startswith('CRISISLEX')]  # Removing CRISISLEX Entries due to elevated false positive rate

                    mlb4 = MultiLabelBinarizer(sparse_output=True)
                    sparse_counts = mlb4.fit_transform(df_counts['counts'])

                    mlb5 = MultiLabelBinarizer(sparse_output=True)
                    sparse_locations = mlb5.fit_transform(df_locations['locations'])

                    small_df = hstack([sparse_locations, sparse_counts])
                    #pca = PCA(n_components=2)
                    #df_counts = pd.DataFrame(pca.fit_transform(df_counts))

                    # print(df_counts.to_string())
                    # df_counts.to_csv('one_hot_encoded_counts.csv', sep=',')
                    # return

                    brc2 = Birch(branching_factor=50, n_clusters=None, threshold=count_thresh, compute_labels=True)
                    predicted_labels2 = brc2.fit_predict(small_df)

                    n2 = 0
                    for item2 in predicted_labels2:
                        if item2 in count_clusters[item]:
                            count_clusters[item][item2].append(
                                list((cluster_row_dict[n2]).values()))  # since cluster_row_dict[n2] is itself a dictionary
                        else:
                            count_clusters[item][item2] = [list((cluster_row_dict[n2]).values())]
                        n2 += 1

                # if write_whole_cluster:
                #     with open('filtered_one/'+file+'.txt', 'w', encoding='utf-8') as file:
                #         for item in count_clusters:
                #             for item2 in count_clusters[item]:
                #                 file.write("\n\nCluster "+str(item)+': ' + str(item2) + "\n")
                #                 for i in range(0, len(count_clusters[item][item2])):
                #                     file.write(count_clusters[item][item2][i][2] + '\n')  # appending url
                # else:
                #     with open('filtered_one/'+file+'.csv', 'w',newline='', encoding='utf-8') as file:
                #         writer = csv.writer(file, delimiter=",")
                #         for item in count_clusters:
                #             for item2 in count_clusters[item]:
                #                 writer.writerow(count_clusters[item][item2][0])

                test_dict = {}


                label = 1
                cluster_labels = [None] * n
                with open( file_prefix + '.txt', 'w', encoding='utf-8') as file:
                    for item in count_clusters:
                        for item2 in count_clusters[item]:
                            file.write("\n\nCluster " + str(item) + ': ' + str(item2) + "\n")
                            for i in range(0, len(count_clusters[item][item2])):
                                gkg_record_id = count_clusters[item][item2][i][0]
                                if(gkg_record_id in test_dict):
                                    print("yes")
                                    print(gkg_record_id)
                                    return
                                test_dict[gkg_record_id]=True
                                #file.write(str(identifier_dict[gkg_record_id]+1)+'\n'+count_clusters[item][item2][i][2]+ '\n' +count_clusters[item][item2][i][3]+ '\n\n')  # appending url
                                file.write(str(identifier_dict[gkg_record_id] + 1)+'\n')
                                cluster_labels[identifier_dict[gkg_record_id]]= label
                            label += 1

                # print(cluster_labels)

                matrix = metrics.cluster.contingency_matrix(class_labels, cluster_labels)
                rand_index,precision,recall,f1 = precision_recall_fmeasure(matrix)

                ari = metrics.cluster.adjusted_rand_score(class_labels, cluster_labels)
                #print("AdjustedRI:", ari)

                nmi = metrics.normalized_mutual_info_score(class_labels, cluster_labels)
                #print("NMI       :", nmi)

                print(birch_thresh, ",", count_thresh, ",", rand_index, ",", precision, ",",recall, ",", f1, ",", ari, ",", nmi)

                # with open(file_prefix + '.csv', 'w', newline='', encoding='utf-8') as file:
                #     writer = csv.writer(file, delimiter=",")
                #     for item in count_clusters:
                #         for item2 in count_clusters[item]:
                #             writer.writerow(count_clusters[item][item2][0])


if __name__ == "__main__":
    main()