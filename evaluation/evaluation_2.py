import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import Birch
from sklearn.decomposition import PCA
from scipy.sparse import hstack
from sklearn import metrics
import numpy as np
from scipy.misc import comb


'''
This script evaluates the performance of clustering for redundancy removal
It clusters news from GDELT GKG on themes, locations and counts (at first level),
and outputs Precision, Recall, F1 Measure, NMI and Rand Index of the clustering
'''


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

    eval_file_names = ['filtered_eval_one_event.csv', 'filtered_eval_three_event.csv', 'filtered_eval_five_event.csv']
    annotated_file_names = ['annotated_one_event.txt', 'annotated_three_event.txt', 'annotated_five_event.txt']

    '''for i in range(1,179):
        if(i not in temp):
            print(i)
    '''

    for m in range(0, len(eval_file_names)):

        fileName = eval_file_names[m]
        file_prefix = 'output'
        print(fileName)

        for birch_thresh in np.arange(0.0, 4.1, 0.2):
            df = pd.read_csv(fileName, header=None, encoding='latin-1')

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
                            class_labels[int(item) - 1] = label
                            temp[int(item)] = True
                        label += 1

            df.columns = ['record_id', 'date', 'url', 'counts', 'themes', 'locations', 'persons', 'organizations', 'tone']

            df = df[pd.notnull(df['themes'])]
            df = df[pd.notnull(df['locations'])]

            df_locations = pd.DataFrame(df['locations'])
            df_counts = pd.DataFrame(df['counts'])
            df_counts.fillna('#',inplace=True)

            df_counts = pd.DataFrame(df_counts['counts'].str.split(';'))  # splitting counts

            for row in df_counts.itertuples():
                for i in range(0, len(row.counts)):
                    try:
                        temp_list = row.counts[i].split('#')
                        row.counts[i] = temp_list[0] + '#' + temp_list[1]  # for retaining only COUNT_TYPE and QUANTITY
                        # print(row.locations[i])
                    except:
                        continue
                if len(row.counts) == 1 and row.counts[0] == '':
                    row.counts.append('#')  # so that news with no counts are clustered together
                    row.counts.pop(0)

                if row.counts[len(row.counts) - 1] == '':
                    row.counts.pop()

            # df_counts.to_csv('countsonly.csv', sep=',')


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

            mlb2 = MultiLabelBinarizer(sparse_output=True)
            sparse_locations = mlb2.fit_transform(df_locations['locations'])

            mlb3 = MultiLabelBinarizer(sparse_output=True)
            sparse_counts = mlb3.fit_transform(df_counts['counts'])

            df = hstack([sparse_themes, sparse_locations,sparse_counts])

            # Reducing dimensions through principal component analysis

            if perform_pca:
                pca = PCA(n_components=None)
                df = pd.DataFrame(pca.fit_transform(df))

            # print("Starting clustering")
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

            # print(n)
            label = 0
            cluster_labels = [None] * n
            with open(file_prefix + '.txt', 'w', encoding='utf-8') as file:
                for item in clusters:
                    file.write("\n\nCluster " + str(item) + "\n")
                    for i in range(0, len(clusters[item])):
                        gkg_record_id = clusters[item][i][0]
                        file.write(str(identifier_dict[gkg_record_id]+1)+'\n'+clusters[item][i][2]+ '\n' +clusters[item][i][3]+ '\n\n')  # appending url
                        cluster_labels[identifier_dict[gkg_record_id]] = label
                    label += 1

            #print(cluster_labels)
            # cluster_labels = predicted_labels

            matrix = metrics.cluster.contingency_matrix(class_labels, cluster_labels)
            rand_index, precision, recall, f1 = precision_recall_fmeasure(matrix)

            ari = metrics.cluster.adjusted_rand_score(class_labels, cluster_labels)
            # print("AdjustedRI:", ari)

            nmi = metrics.normalized_mutual_info_score(class_labels, cluster_labels)
            # print("NMI       :", nmi)

            print(birch_thresh, ",", rand_index, ",", precision, ",", recall, ",", f1, ",", ari, ",",
                  nmi)

if __name__ == "__main__":
    main()