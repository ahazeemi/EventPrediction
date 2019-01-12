
import pandas as pd
import glob
import os


'''
This script prepares a csv containing news only about a particular location
The csv is then used for manually selecting only the news about a particular news event at that location
And these news are manually labelled to evaluate the clustering quality
'''

def main():

    locationADMCode = 'TU34'   # selecting news containing Istanbul only

    path = r'D:\FYP\2017'
    fileName = '2017010100*.gkg.csv'
    all_files = glob.glob(os.path.join(path, fileName))


    df_from_each_file = (pd.read_csv(f,header=None) for f in all_files)
    df = pd.concat(df_from_each_file,ignore_index=True)

    df.columns = ['record_id', 'date', 'url', 'counts', 'themes', 'locations', 'persons', 'organizations', 'tone']

    df = df[pd.notnull(df['themes'])]
    df = df[pd.notnull(df['locations'])]

    df_locations = pd.DataFrame(df['locations'])

    df_locations = pd.DataFrame(df_locations['locations'].str.split(';')) # splitting locations

    indices_to_remove = []

    n = 0
    for row in df_locations.itertuples():
        contains_instanbul = False
        for i in range(0,len(row.locations)):
            try:
                row.locations[i] = (row.locations[i].split('#'))[3]    # for retaining only ADM1 Code
                if(row.locations[i]==locationADMCode):
                    contains_instanbul = True
                    print(n)
            except:
                continue

        if contains_instanbul == False:
            indices_to_remove.append(n)
        n += 1

    df.drop(df.index[indices_to_remove], inplace=True)
    df.to_csv('filtered_eval.csv', sep=',',index=False,header=False)

if __name__ == "__main__":
    main()