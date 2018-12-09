import pandas as pd
import urllib.request
import zipfile
import os
import shutil



'''
This script retains nine columns from GDELT GKG Files
and rewrites filtered files to "filtered" folder
'''


def process():

    for year in range(2016,2018):
        for month in range(1,13):
            for day in range(1,8):
                for hour in range(0,24):
                    for minute in range(0,60,15):
                        yearStr = str(year)
                        monthStr = str(month)
                        dayStr = str(day)
                        hourStr = str(hour)
                        minuteStr = str(minute)

                        if(month < 10):
                            monthStr = '0'+monthStr
                        if (day < 10):
                            dayStr = '0' + dayStr
                        if (hour < 10):
                            hourStr = '0' + hourStr
                        if (minute < 10):
                            minuteStr = '0' + minuteStr

                        file = yearStr+monthStr+dayStr+hourStr+minuteStr+'00.gkg.csv'
                        url = 'http://data.gdeltproject.org/gdeltv2/'+file+'.zip'
                        try:
                            zipped_file = 'zipdownload/' + file+'.zip'
                            extracted_file = 'extracted/' + file

                            urllib.request.urlretrieve(url,zipped_file)

                            with zipfile.ZipFile(zipped_file, "r") as zip_ref:
                                zip_ref.extractall(extracted_file)

                            data = pd.DataFrame(pd.read_csv(extracted_file+'/'+file,sep='\t'))
                            data = data[data.columns[[0, 1,4,5, 7, 9, 11, 13,15]]]  # keeping the required columns
                            data.to_csv('filtered/'+file, sep=',', index=0)

                            if os.path.isfile(zipped_file):
                                os.remove(zipped_file)

                            try:
                                shutil.rmtree(extracted_file)
                            except OSError as e:
                                print("Error removing folder ")

                            print(file)

                        except IOError:
                            print("IO Error: "+file)
                        except UnicodeDecodeError:
                            print("Decode Error: "+file)


def main():
    process()


if __name__ == "__main__":
    main()