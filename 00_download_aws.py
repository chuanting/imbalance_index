# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: 00_download_aws.py
# This file is created by Chuanting Zhang
# Email: chuanting.zhang@kaust.edu.sa
# Date: 2021-03-15 (YYYY-MM-DD)
-----------------------------------------------
"""
import boto3
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config

if __name__ == '__main__':
    GSMA_file = 'data/MCI_Data_2020.xls'
    df_country_info = pd.read_excel(GSMA_file, skiprows=2, sheet_name=2)
    df_name = df_country_info.loc[df_country_info['Year'] == 2019]
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    i = 0
    for name in df_name['ISO Code']:
        print('Now downloading {:} data'.format(name))
        try:
            s3.download_file('dataforgood-fb-data',
                             'csv/month=2019-06/country={:}/type=total_population/{:}_total_population.csv.gz'.format(
                                 name,
                                 name),
                             'data/{:}.gz'.format(name))
        except Exception as e:
            print('File not found!')
            pass
        continue
