import pandas as pd
import numpy as np
import requests
import os
import re
from tqdm import tqdm

def get_exam_data(url_examdata):
    # get exam results data from polito webpage url_examdata
    url = requests.get(url_examdata)
    htmltext = url.text
    lines = htmltext.split('\n')

    for i,line in enumerate(lines):
        # use ReGex to search specific lines in the page
        if re.search('Totale iscritti: [0-9]+', line):
            # gets code, exams taken and exams passed on the considered session
            code = re.search('[0-9]{2}[A-Z]{5}', line).group()
            exams_taken = re.search('Totale iscritti: [0-9]+ ', line).group()
            exams_taken = int( re.search('[0-9]+', exams_taken).group() )
            exams_passed = re.search('Superi: [0-9]+ ', line).group()
            exams_passed = int( re.search('[0-9]+', exams_passed).group() )
        elif re.search("name: 'Iscritti',", line):
            # grades distribution from 15 to 30 is on the next line
            grades_count = [int(g) for g in re.findall('[0-9]+', lines[i+1])]
            grades_count = grades_count[3:] # remove count of 15-17 that are zero

    df = pd.DataFrame(
        [[code, exams_taken, exams_passed, *grades_count]],
        columns=['code','exams_taken','exams_passed', *[_ for _ in range(18,31)]]
    )
    return df

def get_program_data(urls_list, save_filename=None):
    # from urls_list containing the urls to pages with 
    # data from the courses of a degree program, returns the 
    # data from these courses in a single DataFrame and save it in .csv

    data = []
    for url in tqdm(urls_list, ncols=80):
        df = get_exam_data(url)
        data.append(df)
    # concatenate data from exams in a single df
    exams_df = pd.concat(data)
    exams_df.drop_duplicates(inplace=True, ignore_index=True)
    if save_filename:
        exams_df.to_csv(save_filename, index=True)
    return exams_df

if __name__ == '__main__':
    
    urls_filenames = [
        '.\\examsdata\\course01.txt',
        '.\\examsdata\\course02.txt',
        '.\\examsdata\\course03.txt'
    ]

    for urls_filename in urls_filenames:
        with open(urls_filename, 'r') as f:
            urls = f.read().split('\n')
        save_filename, extension = os.path.split(urls_filename)
        get_program_data(urls_filename, f'{save_filename}.csv')