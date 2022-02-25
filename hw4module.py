#%%
import pandas as pd
from pathlib import Path
import os
import sys
import csv
import math
import re
import numpy as np

import warnings
warnings.filterwarnings('ignore')

#Path
readin = 'data/'

#Read in file
file="all_english"
my_file = Path(readin + file + ".csv")
corpus_data = pd.read_csv(my_file, delimiter=',', encoding='utf-8', converters = {"tweet_hashtags": lambda x: x.strip("[]").replace("'","").split(", ")})

corpus_data.head()
#%% 
# Auxiliary functions
def cleanTweets(s):
    #Line breaks
    s = s.replace(r'<lb>', "\n")
    s = re.sub(r'<br */*>', "\n", s)
    # Tabs
    s = s.replace(r'<tab>', "\i")
    #Symbols
    s = s.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    s = s.replace("&amp;", "&")
    # urls
    s = re.sub(r'\(https*://[^\)]*\)', "[url]", s)
    s = re.sub(r'https*://[^\s]*', "[url]", s)
    # Replace double instances of quotations with single instance
    s = re.sub(r'"+', '"', s)
    # custom removals
    s = re.sub(r'@[A-Za-z0-9_]+', "@usermention", s) # remove mentions
    #s = re.sub(r'#[A-Za-z0-9_]+', "#hashtag", s) # remove hashtags
    s = re.sub(r':[^:]+','[emoji]',s) # remove demojized text
    return str(s)


#%%
# Filter so just UK tweets
uk_data = corpus_data[corpus_data["group_country"]=="United Kingdom"]
uk_data = corpus_data[corpus_data["party_name"]=="Labour"]

#Clean text
uk_data['demojized_text'] = [cleanTweets(text) for text in uk_data['demojized_text']]
#len(uk_data)

#%%
#Extract hashtags
hashtags = [i for i in uk_data.tweet_hashtags[uk_data.tweet_hashtags.notnull()] for i in i if i != '']
unique, counts = np.unique(hashtags, return_counts=True)
hashtag_counts = dict(zip(unique, counts))
hashtag_counts = {k: v for k, v in sorted(hashtag_counts.items(), key=lambda item: item[1], reverse = True)}

#%%
from nltk.corpus import stopwords
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import text
np.seterr(divide='ignore', invalid='ignore')

start_time = datetime.now()

# Add custom stop words
add_stopwords = ['usermention','emoji','url'] # can add hashtag here
stop_words = text.ENGLISH_STOP_WORDS.union(add_stopwords)

# Vectorise word counts - only construct quad-grams
vec = CountVectorizer(ngram_range = (4,4), stop_words=stop_words, min_df=15, max_df=0.6)
#Fit vectoriser and convert to dense matrix
uk_vector = vec.fit_transform(uk_data.demojized_text).todense()

# Term frequencies
tf = np.array(uk_vector) # frequencies of each token in a numpy array
totaltf = tf.sum(axis=0) # sum of all frequencies for a particular token for all corpus (column)

#print("matrix has size", uk_vector.shape)

end_time = datetime.now()
#print('Duration: {}'.format(end_time - start_time))
# %%
# Frequencies of all terms
all_terms = dict(zip(vec.get_feature_names_out(), totaltf))
from heapq import nlargest
# DICTIONARY - top 100 terms (you can vary this)
N = 100
top100_terms = nlargest(N, all_terms , key = all_terms.get)
#print(top100_terms)

# %%
# Add column for filtering with text split
demojized_text_split = [i.split() for i in list(uk_data['demojized_text'])]
top100_terms_split = [i.split() for i in top100_terms]

# Example of tweets pushing same event - child forced to lie on hospital floor
eg_child_hospital = [any(item in i for item in ['pneumonia', 'coat', 'floor', 'lie']) for i in demojized_text_split]
tweets_child_floor = uk_data[eg_child_hospital]
list(tweets_child_floor['demojized_text'])

# Example of tweets pushing same event - brexit
eg_brexit = [any(item in i for item in ['brexit', 'botched', 'theresa']) for i in demojized_text_split]
tweets_brexit = uk_data[eg_brexit]
list(tweets_brexit['demojized_text'])

# %%
# Dictionary from the Top 4-grams
# Create a list of terms from the  top100_terms_split object. This will be the dictionary
dictionary = list(set([item for sublist in top100_terms_split for item in sublist]))
#print('Number of words in the dictionary: {}'.format(len(dictionary)))

# Variable with indicator whether the tweet contains at least one world of the dictionary
tweet_matches = [any(item in i for item in dictionary) for i in demojized_text_split]
uk_data_spin = uk_data.copy()
uk_data_spin['dictionary_spin'] = tweet_matches
uk_data_spin['dictionary_spin'] = uk_data_spin['dictionary_spin'].astype(int)

#print(uk_data_spin['dictionary_match'].value_counts())



# %%
# Label of spin from the use of media attached

# Load and merge the data
uk_md = pd.read_csv("data/uk_tweets.csv")
uk_md = uk_md[['id', 'created_at', 'entities.urls', 'attachments.media', 'author.public_metrics.followers_count']]

uk_md['created_at'] = pd.to_datetime(uk_md['created_at'], format="%Y-%m-%dT%H:%M:%S.000Z")

# Merge with the complete data
uk_data_spin = pd.merge(uk_data_spin, uk_md, on = 'id')
uk_data_spin['media_spin'] = uk_data_spin['attachments.media'].str.contains('video|photo', na=False).astype(int)



# %%
# Creation of some control variables

uk_data_spin['hashtags_counts'] = np.where(uk_data_spin['tweet_hashtags'].astype(str) == "['']",
                                          0,
                                          uk_data_spin['tweet_hashtags'].apply(lambda x: len(x)) )

# Length of the tweet
uk_data_spin['tweet_length'] = uk_data_spin['demojized_text'].apply(lambda x: len(x))

# decile of followers
uk_data_spin['decile_followers'] = pd.qcut(uk_data_spin['author.public_metrics.followers_count'], 10, labels=[i for i in range(1,11)])

# Day of the week
uk_data_spin['created_at_week_day'] = (uk_data_spin['created_at'].dt.weekday + 1).astype(object)

# Hour of the day
uk_data_spin['created_at_hour'] = (uk_data_spin['created_at'].dt.hour).astype(object)
uk_data_spin['created_at_hour'].replace([0,1,2,3,4,5,6], "0-6" , inplace=True)



# %%
# Exploratory analysis
#sns.boxplot(x = 'dictionary_match', y = 'public_metrics.like_count', data = uk_data_spin)

#sns.boxplot(x = 'dictionary_match', y = 'public_metrics.like_count', data = uk_like_counts)

#sns.kdeplot('public_metrics.like_count', data = uk_like_counts)



# %%
# Validation of the model

# White test 
#white_test = het_white(results.resid,  results.model.exog)
#labels = ['Test Statistic', 'Test Statistic p-value', 'F-Statistic', 'F-Test p-value']
#print results of White's test
#print(dict(zip(labels, white_test)))


#sns.scatterplot(x = y, y =  results.resid)

#sns.kdeplot(results.resid)


# Include # hashtags, # number of followers, length of tweets, # day, # hour of post
# 

# %% Preprocessing and linear regression function

import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_white


# The next
def run_cluster_Robust_OLS(df, target, treatment, controls, group_var):
    """
    This function takes a df and the target variable for running a preprocessing
    step and a cluster robust ols with a given set of regressors
    """
    
    # Preprocessing
    # Exclude tweets without reactions
    df_counts = df[df[target]>0]

    # Exclude atypical points by the method of 
    lk_q1 = df_counts['public_metrics.like_count'].quantile(0.25)
    lk_q3 = df_counts['public_metrics.like_count'].quantile(0.75)

    # lower bound
    lk_lw = lk_q1 - 1.5*(lk_q3 - lk_q1)
    lk_up = lk_q3 + 1.5*(lk_q3 - lk_q1)

    df_counts = df_counts[(df_counts[target]>lk_lw) & (df_counts[target]<lk_up)]

    # Regression
    
    X = df_counts[treatment + controls]
    X = pd.get_dummies(X,  drop_first=True)
    X = sm.add_constant(X)

    y = np.log(df_counts[target])

    group_var_ = df_counts[group_var]
    
    model = sm.OLS(y, X)
    results = model.fit(cov_type='cluster', cov_kwds={'groups': group_var_})


    print('Robust cluster OLS over {} observations using {} as target. On average the interactions increased {}% (p-value = {}). R2 = {}'.format(int(results.nobs), 
                                                                                                                         target, 
                                                                                                                         round(results.params[1]*100,2),
                                                                                                                         round(results.pvalues[1], 3),
                                                                                                                         round(results.rsquared_adj,3)))
    

    return int(results.nobs), target, round(results.params[1]*100,2), round(results.pvalues[1],3), round(results.rsquared_adj,3)


# %%
# Plot of some metrics for spined/not spined tweets

import seaborn as sns
sns.set_theme(style="whitegrid")
sns.color_palette("Paired")
import matplotlib.pyplot as plt

def metrics_plot(df, spin_tweet):
    """
    df: dataframe with the metrics variables and the label of spin/not spin tweet
    spin_tweet: var name that indicates weather a tweet has spin content (1) or not (0)
    """
    metrics = ['public_metrics.like_count', 'public_metrics.quote_count', 'public_metrics.reply_count', 'public_metrics.retweet_count']
    df2 = df.copy()
    df2[spin_tweet].replace({0:'No', 1:'Yes'}, inplace=True)
    reactions_agg = df2.groupby(spin_tweet)[metrics].median().reset_index()
    
    # Median number of reactions for tweets that match /don't match the dictionary
    reactions_agg_ = pd.melt(reactions_agg,
                             id_vars = spin_tweet,
                             var_name = 'metric',
                             value_name  = 'value')
    # Long data set
    reactions_agg_ ['metric'] = reactions_agg_ ['metric'].str.replace('(public_metrics.)|(\_count)', '', regex=True)
    
    # Plot 
    plt.figure(figsize=(12,7))
    sns.barplot(x="metric", y="value",
                hue=spin_tweet, 
                data=reactions_agg_)
    plt.title('Median number of reactions', fontsize=12, style='italic')
    plt.xlabel('Type of interaction')
    plt.ylabel('Value')
    plt.legend(title='Spin content')
    plt.show()

#metrics_plot(uk_data_spin, 'dictionary_spin')
    
# %%
# using quadgrams as dictionary terms

#create a new column of tweets without stop words
stop_words = list(stop_words)
uk_data['demojized_text_lower_no_stopwords'] = uk_data['demojized_text'].str.lower()
uk_data['demojized_text_lower_no_stopwords'] = uk_data['demojized_text_lower_no_stopwords'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
uk_data['demojized_text_lower_no_stopwords']

# Our dictionary would be composed of the top 100 quadgrams we got in the previous step
dictionary_1 = top100_terms
#print('Number of terms in the dictionary: {}'.format(len(dictionary_1)))
#print(dictionary_1[0:10])

# Variable with indicator whether the tweet contains at one set of quad-gram in the dictionary
tweet_matches_1 = [any(item in i for item in dictionary_1) for i in uk_data['demojized_text_lower_no_stopwords']]
uk_data['tweet_matches_1'] = tweet_matches_1 
uk_data_spin = uk_data_spin.merge(uk_data[['id', 'tweet_matches_1']], on = ['id'], how = 'left')
uk_data_spin['dictionary_spin_quad'] = uk_data_spin['tweet_matches_1']
uk_data_spin['dictionary_spin_quad'] = uk_data_spin['dictionary_spin_quad'].astype(int)

#print(uk_data_spin['dictionary_spin_quad'].value_counts())

# %%
# Dictionary from the Top 4-grams, excluding terms that are also used by Conservative
dictionary_labour = open("labour_dictionary.txt").read().split()
#print('Number of words in the dictionary: {}'.format(len(dictionary_labour)))

# Variable with indicator whether the tweet contains at least one world of the dictionary
uk_data['dictionary_labour_spin'] = [any(item in i for item in dictionary_labour) for i in demojized_text_split]
uk_data_spin = uk_data_spin.merge(uk_data[['id', 'dictionary_labour_spin']], on = ['id'], how = 'left')
uk_data_spin['dictionary_labour_spin'] = uk_data_spin['dictionary_labour_spin'].astype(int)

#print(uk_data_spin['dictionary_labour_spin'].value_counts())

# %% 
# Full regression results for one model
def run_cluster_Robust_OLS_full(df, target, treatment, controls, group_var):
    """
    This function takes a df and the target variable for running a preprocessing
    step and a cluster robust ols with a given set of regressors, shows the full results
    """
    
    # Preprocessing
    # Exclude tweets without reactions
    df_counts = df[df[target]>0]

    # Exclude atypical points by the method of 
    lk_q1 = df_counts['public_metrics.like_count'].quantile(0.25)
    lk_q3 = df_counts['public_metrics.like_count'].quantile(0.75)

    # lower bound
    lk_lw = lk_q1 - 1.5*(lk_q3 - lk_q1)
    lk_up = lk_q3 + 1.5*(lk_q3 - lk_q1)

    df_counts = df_counts[(df_counts[target]>lk_lw) & (df_counts[target]<lk_up)]

    # Regression
    
    X = df_counts[treatment + controls]
    X = pd.get_dummies(X,  drop_first=True)
    X = sm.add_constant(X)

    y = np.log(df_counts[target])

    group_var_ = df_counts[group_var]
    
    model = sm.OLS(y, X)
    results = model.fit(cov_type='cluster', cov_kwds={'groups': group_var_})
    
    print(results.summary())
                                                               