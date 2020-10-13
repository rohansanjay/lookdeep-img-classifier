import wget
import pandas as pd
from zipfile import ZipFile

df = pd.DataFrame()

for tranch in range(1, 4):
    
    tranch_images = 'http://sampy.ml:499/persons-posture-tranch' + str(tranch) + '.zip'
    tranch_labels = 'http://sampy.ml:499/tranch' + str(tranch) + '_labels.csv'
    
    pictures_path = wget.download(tranch_images)
    labels_path = wget.download(tranch_labels)
    
    labels = pd.read_csv(labels_path)
    zip_file = ZipFile(pictures_path)

    file_list = [obj.filename for obj in zip_file.infolist()]
    file_list_simple = [name.split('/')[-1] for name in file_list]

    names = pd.DataFrame({'file_path': file_list, 'file_name': file_list_simple})
    names.head();

    tranch_df = pd.merge(names, labels, on = 'file_name')
    print(len(names), len(labels), len(tranch))
    df = df.append(tranch_df)
    zip_file.extractall()

df.to_csv('tranch_master.csv')