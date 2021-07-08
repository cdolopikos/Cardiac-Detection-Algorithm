import os
import glob
import pandas as pd
def concat():
    os.chdir("/Users/cmdgr/OneDrive - Imperial College London/pr_data/Preprocessed_data")
    cwd = os.getcwd()
    print(cwd)
    extension = 'csv'
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
    print(all_filenames)
    #combine all files in the list
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
    combined_csv=combined_csv.fillna(0)
    # for f in all_filenames:
    #     print(f)
    #     pd.read_csv(f)
    #export to csv
    combined_csv.to_csv( "/Users/cmdgr/OneDrive - Imperial College London/pr_data/Preprocessed_data/combined_csv.csv", index=False, encoding='utf-8-sig')

concat()