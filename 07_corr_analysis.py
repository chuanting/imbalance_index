# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: 07_corr_analysis.py
# This file is created by Chuanting Zhang
# Email: chuanting.zhang@kaust.edu.sa
# Date: 2021-03-21 (YYYY-MM-DD)
-----------------------------------------------
"""
import pandas as pd
import geopandas as gpd
import os
import numpy as np
import mobile_codes
import itertools


def main():
    folder = 'data/division/'
    final_results = []
    gsma_file = 'data/MCI_Data_2020.xls'
    gsma_data = pd.read_excel(gsma_file, skiprows=2, sheet_name=2)
    df_gsma = gsma_data.loc[gsma_data['Year'] == 2019]
    df_gsma = df_gsma[['Country', 'ISO Code', 'Index']]
    df_gsma.columns = ['Country', 'ISO Code', 'GSMA_index']
    inclusive_file = 'data/Inclusive_index.xls'
    inclusive_data = pd.read_excel(inclusive_file)
    inclusive_data.columns = ['Country', 'Inclusive_index']
    # find out countries in both index
    common_contry = pd.merge(left=inclusive_data, right=df_gsma, on='Country', how='left')
    common_contry.dropna(inplace=True)

    final_index = pd.read_csv(folder + 'final_index.csv')

    print(common_contry.head(10))
    for i, name in enumerate(common_contry['ISO Code']):
        whole_name = mobile_codes.alpha3(name).name
        print(i, name, whole_name)
        gsma_index = common_contry.loc[common_contry['ISO Code'] == name, 'GSMA_index'].values[0]
        inclusive_index = common_contry.loc[common_contry['ISO Code'] == name, 'Inclusive_index'].values[0]
        our_index = final_index.loc[final_index['ISO Code'] == name, 'index'].values[0]

        final_results.append((whole_name, gsma_index, inclusive_index, our_index))
    header = ['Country', 'GSMA_Index', 'Inclusive_Index', 'Inequality_Index']

    exclude_country = ['Latvia', 'Lithuania', 'Kuwait',
                       'Bahrain', 'Iran, Islamic Republic of', 'Myanmar']

    df = pd.DataFrame(final_results, columns=header)

    df.drop(df[df['Country'].isin(exclude_country)].index, inplace=True)
    df.to_csv(folder + 'common_country_index.csv', index=False)


if __name__ == '__main__':
    main()
