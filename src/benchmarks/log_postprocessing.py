#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import re
import os

# Hard coded logs to process
logs = [ [ '1b-r6a-24x-125MiB-5x-infinite-clang-2023-06-25-13-10.log', 'clang',  '[F]' ],
         [ '1b-r6a-24x-125MiB-5x-g++-2023-06-24-23-01.log',            'g++',    '[C]' ],
         [ '1b-c6a-16x-125MiB-g++-docker-2023-06-23-14-16.log',        'docker', '[C]' ],
         [ '1b-r6a-24x-125MiB-5x-clang-2023-06-24-13-57.log',          'clang',  '[F]' ]
       ]

header_pattern = r'^\s*\-\|\-\s*Algorithm\b.*$'
data_pattern = r'^\s*\-\|\-\s*qv_heap\b.*$'

qn = pd.DataFrame()
results_table = pd.DataFrame()
header = None

for logpair in logs:
    log = logpair[0]
    time_column = logpair[2]
            
    data_lines = []
    with open(log, 'r') as file:
        
        for line in file:
            if not header:
                if re.search(header_pattern, line):
                    header = line.strip()
            else:
                if re.search(data_pattern, line):
                    data_lines.append(line.strip())

    for count, config in enumerate(['finite', '10M', '1M']):

        # Create the DataFrame
        df = pd.DataFrame([header.split()] + [line.split() for line in data_lines])
        df.columns = df.iloc[0]
        df = df.iloc[1:].reset_index(drop=True)

        df.index = pd.RangeIndex(len(df)) 
        df_min = df.groupby(np.arange(len(df))//5)[time_column].min() # g++ is C, clang is F

        # print(df.shape[0])
        
        if (count*200 >= df.shape[0]):
            break
        
        df = df.iloc[count*200:(count+1)*200]


        if qn.empty:
            qn = df[['Queries', 'nprobe']].astype(int)
            qn = qn.groupby(np.arange(len(df))//5)[['Queries', 'nprobe']].mean().astype(int)
            results_table = qn
            results_table.columns=pd.MultiIndex.from_product([['Params'],['Queries', 'nprobe']])

        # print(logpair[0], logpair[1])
        # g++ is C, clang is F

        df = df[time_column].to_frame()

        # I don't know why this doesn't work, but it doesn't -- in a very very strange way
        # df.index = pd.RangeIndex(len(df)) 
        # grouped = df.groupby(np.arange(len(df))//5)
        # for group, indices in grouped.groups.items():
        #    print(f"Group: {group}")
        #    print(df.loc[indices])
        #    print()
        # df_min = df.groupby(np.arange(len(df))//5).min().astype(float)
        # print(df_min)
        
        # Compute the average of every five non-overlapping rows
        df_min = df.rolling(5).min().iloc[4::5]

        # Reset the index of the new DataFrame
        df_min = df_min.reset_index(drop=True)
        df_min.columns=['latency']
        # df.rename(columns={'[C]':'latency'})

        df_qps = (qn['Params','Queries'] / df_min['latency']).to_frame()
        df_qps.columns=['QPS']

        performance = pd.concat([df_min['latency'], df_qps['QPS']], axis=1)
        performance.columns = pd.MultiIndex.from_product([[logpair[1]],['latency', 'QPS']])
        results_table = pd.concat([results_table, performance], axis=1)

# Don't abbreviate when printing / displaying dataframe
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)        
        
# column_width = 12
# fixed_width_table = results_table.to_string(index=False, col_space=column_width)
# print(fixed_width_table)

# Save the table to a file
file_path = 'ivf_logs.csv'
results_table.to_csv(file_path, index=False)
    
# Display results table (Mostly useful in a jupyter notebook)
# results_table
