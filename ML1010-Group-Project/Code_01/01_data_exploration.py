# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 10:24:25 2021

@author: oanad
"""

import pandas as pd
import numpy as np
import seaborn 
import matplotlib.pyplot 


XXX_All = pd.read_pickle(r"C:\Users_Folders\Cursuri_toate\YORK_MLcertificate\Course_02\Project/trimmed_cellphone.pkl")


XXX_All.dtypes
XXX_All.columns

################################################################################################
## retain only essentia fields and add other relevant fields
XXX_All=XXX_All[['overall', 'verified', 'reviewTime', 'reviewerID', 'asin', 'style',
        'reviewText', 'summary', 'vote','category',  'description', 'title','brand', 'feature',  'main_cat','price']]

#XXX_All['years_2_today']=(XXX_All['reviewTime']-pd.to_datetime("now"))/pd.TimeDelta('1D')
XXX_All["reviewTime"]=pd.to_datetime(XXX_All["reviewTime"])
XXX_All["year"]=XXX_All["reviewTime"].dt.year




################################################################################################
### remove tablets
temp_loc=XXX_All["main_cat"]!="Computers"
XXX_All=XXX_All[temp_loc]

### remove ipods
temp_loc=XXX_All["main_cat"]!="Apple Products"
zzz=XXX_All[temp_loc]

### check what is in the "All Electronics" category
temp_loc=XXX_All["main_cat"]=="All Electronics"
zzz=XXX_All[temp_loc]




################################################################################################
## looking for potential duplicate entries
grouped1 = XXX_All.groupby(by=[ 'reviewerID', 'asin']).agg(No_scores=('reviewerID','count'))
print(grouped1.size)
grouped2 = XXX_All.groupby(by=[ 'reviewerID', 'asin','title']).agg(No_scores=('reviewerID','count'))
print(grouped2.size)
grouped3 = XXX_All.groupby(by=[ 'reviewerID', 'asin','title','reviewText']).agg(No_scores=('reviewerID','count'))
print(grouped3.size)





################################################################################################
### group into one category the periferic brands
def category_brands(x):
    if x in ['Samsung','BLU','Apple',"LG","Motorola",'Huawei','BlackBerry','HTC','Nokia','Asus','Sony','ZTE','Tracfone']:
        return x
    else:
        return 'Other'
XXX_All['main_brands'] = XXX_All['brand'].apply(category_brands)

# XXX_All['main_cat'] = pd.Categorical(XXX_All.main_cat)
# XXX_All["main_cat"].describe()
# XXX_All["main_cat"].unique()
# XXX_All["category"].unique()
# zzz_phones=grouped = XXX_All.groupby(by=["asin","title","brand"]).count(["reviewTime"])

  ### group into one category non cell phones 
def category_groups(x):
    if x =='''['Cell Phones & Accessories', 'Cell Phones', 'Carrier Cell Phones']''':
        return 'Carrier Cell Phones'
    elif x=='''['Cell Phones & Accessories', 'Cell Phones', 'Unlocked Cell Phones']''':
        return 'Unlocked Cell Phones'
    elif x=='''['Cell Phones & Accessories', 'Cell Phones']''':
        return 'Cell Phones'
    else:
        return 'Other'    
XXX_All['category'] = XXX_All['category'].apply(category_groups)


### count the review length
def review_length(x):
    if isinstance(x, str):
        return len(x)
    else:
        return -1
XXX_All['reviewText_length'] = XXX_All['reviewText'].apply(review_length)
 


################################################################################################
### descriptive stats: unique phone specs
zzz_phones=XXX_All[['brand','asin','title','reviewTime','overall']].groupby(by=['brand','asin','title']).\
               agg(No_scores=('overall','count'),Score_min=('overall','min'),Score_avg=('overall','mean'),Score_std=('overall','std'),Score_med=('overall','median'),Score_max=('overall','max'),Date_min=('reviewTime','min'),Date_max=('reviewTime','max'))


################################################################################################
### descriptive stats: brands
zzz_brand = XXX_All.groupby(by=['brand','overall']).agg(No_reviews=('overall','count'))





################################################################################################
### descriptive stats: count of unique brands per year
zzz_brands_per_year=XXX_All[['brand','year']].groupby(by=['brand','year']).agg(No_reviews_per_phone_unique_specs=('year','count')).reset_index()
seaborn.catplot(x="year",
                data=zzz_brands_per_year, kind="count",sharey=False,
                height=2.5, aspect=5,palette="crest").set(title='No of unique brands per year')


### count of distinct phones per year
zzz_phones_per_year=XXX_All[['brand','year','asin','title']].groupby(by=['brand','asin','title','year']).agg(No_reviews_per_phone_unique_specs=('asin','count')).reset_index()
seaborn.catplot(x="year",
                data=zzz_phones_per_year, kind="count",sharey=False,
                height=2.5, aspect=5,palette="crest").set(title='No of unique phone specifications per year')


### reviews per year
seaborn.catplot(x="year",
                data=XXX_All, kind="count",sharey=False,
                height=2.5, aspect=5,palette="crest").set(title='No of reviews per year')





### average reviews per phone spec
f, ax = matplotlib.pyplot.subplots(figsize=(15, 3))
matplotlib.pyplot.title('Distribution of no of reviews per unique phone specification (optimized number of quantiles, see https://vita.had.co.nz/papers/letter-value-plot.pdf)')
matplotlib.pyplot.yscale("log")
matplotlib.pyplot.grid(axis = 'y',linestyle = '--')
seaborn.despine(f)
seaborn.boxenplot(
    x="year", y="No_reviews_per_phone_unique_specs",k_depth=4,
               palette="crest",
              scale="linear", data=zzz_phones_per_year,
)




################################################################################################
## descriptive stats: reviews overall rating per brand
#g = seaborn.catplot(x="overall",col="main_brands",
#                data=XXX_All[XXX_All["year"]>2010], kind="count",sharey=False,
#                height=4, aspect=.7,palette="Blues_d")

f, ax = matplotlib.pyplot.subplots(figsize=(15, 2.5))
matplotlib.pyplot.title('Overall ratings distribution, all data (optimized number of quantiles, see https://vita.had.co.nz/papers/letter-value-plot.pdf)')
matplotlib.pyplot.grid(axis = 'y',linestyle = '--')
seaborn.despine(f)
seaborn.boxenplot(
    x="main_brands", y="overall",k_depth=4,
              palette="viridis", order= ['Samsung','BLU','Apple',"LG","Motorola",'Huawei','BlackBerry','HTC','Nokia','Asus','Sony','ZTE','Tracfone','Other'],
              scale="linear", data=XXX_All,
)

f, ax = matplotlib.pyplot.subplots(figsize=(15, 2.5))
matplotlib.pyplot.title('Overall ratings distribution, 2013 & before (optimized number of quantiles, see https://vita.had.co.nz/papers/letter-value-plot.pdf)')
matplotlib.pyplot.grid(axis = 'y',linestyle = '--')
seaborn.despine(f)
seaborn.boxenplot(
    x="main_brands", y="overall",k_depth=4,
              palette="viridis", order= ['Samsung','BLU','Apple',"LG","Motorola",'Huawei','BlackBerry','HTC','Nokia','Asus','Sony','ZTE','Tracfone','Other'],
              scale="linear", data=XXX_All[(XXX_All["year"]<=2013)],
)

f, ax = matplotlib.pyplot.subplots(figsize=(15, 2.5))
matplotlib.pyplot.title('Overall ratings distribution, 2014-2016 (optimized number of quantiles, see https://vita.had.co.nz/papers/letter-value-plot.pdf)')
matplotlib.pyplot.grid(axis = 'y',linestyle = '--')
seaborn.despine(f)
seaborn.boxenplot(
    x="main_brands", y="overall",k_depth=4,
              palette="viridis",order= ['Samsung','BLU','Apple',"LG","Motorola",'Huawei','BlackBerry','HTC','Nokia','Asus','Sony','ZTE','Tracfone','Other'],
              scale="linear", data=XXX_All[(XXX_All["year"]>=2014) & (XXX_All["year"]<=2016)],
)


f, ax = matplotlib.pyplot.subplots(figsize=(15, 2.5))
matplotlib.pyplot.title('Overall ratings distribution, 2017 & after (optimized number of quantiles, see https://vita.had.co.nz/papers/letter-value-plot.pdf)')
matplotlib.pyplot.grid(axis = 'y',linestyle = '--')
seaborn.despine(f)
seaborn.boxenplot(
    x="main_brands", y="overall",k_depth=4,
              palette="viridis",order= ['Samsung','BLU','Apple',"LG","Motorola",'Huawei','BlackBerry','HTC','Nokia','Asus','Sony','ZTE','Tracfone','Other'],
              scale="linear", data=XXX_All[XXX_All["year"]>2016],
)



################################################################################################
## descriptive stats: reviews length
f, ax = matplotlib.pyplot.subplots(figsize=(15, 3))
matplotlib.pyplot.title('Overall review length distribution, all data (optimized number of quantiles, see https://vita.had.co.nz/papers/letter-value-plot.pdf)')
matplotlib.pyplot.yscale("log")
matplotlib.pyplot.grid(axis = 'y',linestyle = '--')
seaborn.despine(f)
seaborn.boxenplot(
    x="main_brands", y="reviewText_length",k_depth=4,
              palette="viridis", order= ['Samsung','BLU','Apple',"LG","Motorola",'Huawei','BlackBerry','HTC','Nokia','Asus','Sony','ZTE','Tracfone','Other'],
              scale="linear", data=XXX_All[(XXX_All["reviewText_length"]>=1)],
)

f, ax = matplotlib.pyplot.subplots(figsize=(15, 3))
matplotlib.pyplot.title('Overall review length  distribution, 2013 & before (optimized number of quantiles, see https://vita.had.co.nz/papers/letter-value-plot.pdf)')
matplotlib.pyplot.yscale("log")
matplotlib.pyplot.grid(axis = 'y',linestyle = '--')
seaborn.despine(f)
seaborn.boxenplot(
    x="main_brands", y="reviewText_length",k_depth=4,
              palette="viridis", order= ['Samsung','BLU','Apple',"LG","Motorola",'Huawei','BlackBerry','HTC','Nokia','Asus','Sony','ZTE','Tracfone','Other'],
              scale="linear", data=XXX_All[(XXX_All["year"]<=2013)&(XXX_All["reviewText_length"]>=1)],
)

f, ax = matplotlib.pyplot.subplots(figsize=(15, 3))
matplotlib.pyplot.title('Overall review length  distribution, 2014-2016 (optimized number of quantiles, see https://vita.had.co.nz/papers/letter-value-plot.pdf)')
matplotlib.pyplot.yscale("log")
matplotlib.pyplot.grid(axis = 'y',linestyle = '--')
seaborn.despine(f)
seaborn.boxenplot(
    x="main_brands", y="reviewText_length",k_depth=4,
              palette="viridis",order= ['Samsung','BLU','Apple',"LG","Motorola",'Huawei','BlackBerry','HTC','Nokia','Asus','Sony','ZTE','Tracfone','Other'],
              scale="linear", data=XXX_All[(XXX_All["year"]>=2014) & (XXX_All["year"]<=2016)&(XXX_All["reviewText_length"]>=1)],
)


f, ax = matplotlib.pyplot.subplots(figsize=(15, 3))
matplotlib.pyplot.title('Overall review length distribution, 2017 & after (optimized number of quantiles, see https://vita.had.co.nz/papers/letter-value-plot.pdf)')
matplotlib.pyplot.yscale("log")
matplotlib.pyplot.grid(axis = 'y',linestyle = '--')
seaborn.despine(f)
seaborn.boxenplot(
    x="main_brands", y="reviewText_length",k_depth=4,
              palette="viridis",order= ['Samsung','BLU','Apple',"LG","Motorola",'Huawei','BlackBerry','HTC','Nokia','Asus','Sony','ZTE','Tracfone','Other'],
              scale="linear", data=XXX_All[(XXX_All["year"]>2016)&(XXX_All["reviewText_length"]>=1)],
)

################################################################################################
## descriptive stats: reviews length by vote and overall rating
### count of reviews per length and score
zzz_reviews_by_overall_vote=XXX_All[(XXX_All["year"]>2016)&(XXX_All["reviewText_length"]>=1)][['reviewText_length','overall','vote','asin']].groupby(by=['reviewText_length','overall','vote']).agg(No_reviews=('asin','count')).reset_index()


f=seaborn.relplot(
    data=zzz_reviews_by_overall_vote,
    x="reviewText_length", y="vote",
    hue="overall", size="No_reviews",
    palette="viridis", sizes=(10, 200),
).set(title='No of reviews per year')

