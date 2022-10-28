# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from string import punctuation


# Get Data
def vectorize_data_no_norm(normalize = True):
    # Day videos were posted
    day_p = datetime.datetime.strptime('07-15-2012', '%m-%d-%Y') 
    day_k = datetime.datetime.strptime('09-05-2013', '%m-%d-%Y') 
    day_l = datetime.datetime.strptime('03-09-2011', '%m-%d-%Y') 
    day_e = datetime.datetime.strptime('08-05-2010', '%m-%d-%Y') 
    day_s = datetime.datetime.strptime('06-04-2010', '%m-%d-%Y')
    dates = [day_p,day_k,day_l,day_e,day_s]
    
    #func
    
    lst = ['1-Psy','2-KatyPerry','3-LMFAO','4-Eminem','5-Shakira']
    data_files = ['Youtube0' + l + '.csv' for l in lst]
    dat = pd.read_csv(data_files[0])
    dat['VIDEO_AUTHOR'] = ['Psy']*dat.shape[0]
    time_elapsed = []
    time_of_day = []
    for t in dat['DATE']:
        if type(t) != str:
            time_elapsed.append(0)
            time_of_day.append(0)
        else:
            date = datetime.datetime.strptime(t, '%Y-%m-%dT%H:%M:%S')
            time_elapsed.append((date - dates[0]).total_seconds())
            time_of_day.append(date.hour*60*60+date.minute*60+date.second)
    dat['TIME'] = time_of_day
    dat['TIME_ELAPSED'] = time_elapsed
    for i in range(1,5):
        df = pd.read_csv(data_files[i])
        df['VIDEO_AUTHOR'] = [lst[i][2:]]*df.shape[0]
        time_elapsed = []
        time_of_day = []
        for t in df['DATE']:
            if type(t) != str:
                time_elapsed.append(0)
                time_of_day.append(0)
            else:
                if len(t) == 19:
                    date = datetime.datetime.strptime(t, '%Y-%m-%dT%H:%M:%S')
                else:
                    date = datetime.datetime.strptime(t, '%Y-%m-%dT%H:%M:%S.%f')
                time_elapsed.append((date - dates[i]).total_seconds())
                time_of_day.append(date.hour*60*60+date.minute*60+date.second)
        # Deal with missing data        
        num_fails = sum([t == 0 for t in time_of_day])
        l = len(time_of_day)
        sum_t = sum(time_of_day)
        sum_t_e = sum(time_elapsed)
        avg_t = sum_t / (l - num_fails)
        avg_t_e = sum_t_e / (l - num_fails)
        def rep_missing_t(x):
            if x == 0:
                return avg_t
            return x
        def rep_missing_t_e(x):
            if x == 0:
                return avg_t_e
            return x
        time_of_day = list(map(rep_missing_t, time_of_day))
        time_elapsed = list(map(rep_missing_t_e, time_elapsed))
        df['TIME'] = time_of_day
        df['TIME_ELAPSED'] = time_elapsed
        dat = pd.concat([dat,df], ignore_index = True)
    
    # Vectorize data
    txt_data = dat['CONTENT']
    exclude = set(punctuation)
    for txt in txt_data:
        char_no_punct = [ char for char in txt if char not in exclude ]
        text_no_punct = "".join(char_no_punct)
        txt = text_no_punct.lower()
    vectorizer = CountVectorizer(stop_words = 'english', min_df = 10)
    fit = vectorizer.fit_transform(txt_data)
    arr = fit.toarray()
    words = vectorizer.get_feature_names()
    vec_dat = pd.DataFrame(arr)
    vec_dat.columns = words
    #vec_dat['TIME'] = dat['TIME']
    #vec_dat['TIME_ELAPSED'] = dat['TIME_ELAPSED']
    vec_dat['CLASS'] = dat['CLASS']
    
    # For doing analysis
    X = vec_dat.iloc[:,:-1]
    if normalize:
        X = normalize(X)
    Y = vec_dat.iloc[:,-1]
    
    return X, Y
