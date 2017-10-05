import sys
sys.path.append('R:\\Users\\4126694\\Python\\Modules')
import blpfunctions as blp
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.cluster import KMeans
#from sklearn.metrics import classification_report
import random
import matplotlib.gridspec as gridspec
from dateutil import rrule
from pandas.tseries.offsets import *
from holidays_jp import CountryHolidays
import dateutil.rrule as RR

def bbg_volcurve(ind, event, edate, numdays, interval,fld_lst):
    #sec_list = blp.get_index(ind)
    sec_list = ind
    volcurves = pd.DataFrame()
    fmt = "%Y-%m-%d" + 'T' + "%H:%M:%S"  #Assumes no milliseconds
    endDateTime = dt.datetime.strptime(edate, fmt)
   
    #skip SQ and holidays
    day1=dt.datetime(2017,1,1)
    sq = list(RR.rrule(RR.MONTHLY,byweekday=RR.FR,bysetpos=2,dtstart=day1,until=endDateTime))
    hols = list(zip(*CountryHolidays.get('JP', int(edate[0:4])))[0])
    skipdays = sq + hols
    bday_jp = CustomBusinessDay(holidays=skipdays)
    
    startDateTime = endDateTime.replace(hour=9) - numdays*bday_jp
    timedelta = pd.date_range(startDateTime, endDateTime, freq=bday_jp).nunique()
    sdate = startDateTime.strftime(fmt)
    
    for stock in sec_list:
        stock = str(stock) + ' JP Equity'
        output=blp.get_Bars(stock, event, sdate, edate, interval, fld_lst)
        output.rename(columns={'VOLUME':stock},inplace=True)
        volcurves = volcurves.join(output,how="outer")

    #process the raw data into historical averages
    volcurves.rename(columns=lambda x: x[:4], inplace=True)
    timevect = pd.Series(volcurves.index.values)
    timeframet = timevect.to_frame()
    timeframet.columns =['date']
    timeframet.set_index(timevect,inplace="True")
    volcurves.set_index(timevect,inplace="True")#timezone hack
    timeframet['bucket'] = timeframet['date'].apply(lambda x: dt.datetime.strftime(x, '%H:%M:%S'))
    timeframet=timeframet.join(volcurves)
    volcurvesum=timeframet.groupby(['bucket']).sum()
    adv = volcurvesum.sum()/timedelta
    volcurves = volcurvesum / volcurvesum.sum()
    volcurves = volcurves.cumsum()
    volcurves = volcurves.interpolate()
    volcurvesum = volcurvesum.interpolate()
    volcurvesum = volcurvesum.dropna(axis=1,how='all')
    
    return (adv, volcurvesum.fillna(method='bfill'), volcurves.fillna(method='bfill'))
    
    
def DTWDistance(s1, s2, w):
    DTW={}

    w = max(w, abs(len(s1)-len(s2)))

    for i in range(-1,len(s1)):
        for j in range(-1,len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(max(0, i-w), min(len(s2), i+w)):
            dist= (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])

    return np.sqrt(DTW[len(s1)-1, len(s2)-1])

def LB_Keogh(s1,s2,r):
    LB_sum=0
    for ind,i in enumerate(s1):

        lower_bound=min(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
        upper_bound=max(s2[(ind-r if ind-r>=0 else 0):(ind+r)])

        if i>upper_bound:
            LB_sum=LB_sum+(i-upper_bound)**2
        elif i<lower_bound:
            LB_sum=LB_sum+(i-lower_bound)**2

    return np.sqrt(LB_sum)
    
#Return centroids of clusters
def k_means_clust(data,num_clust,num_iter,w=4):
    centroid_list=random.sample(data.columns.values,num_clust)
    centroids = data[centroid_list]
    counter=0
    for n in range(num_iter):
        counter+=1
        #print "Iteration: ", counter
        asses={}
        #assign data points to clusters
        for ind,i in enumerate(data):
            min_dist=float('inf')
            closest_clust=None
            for c_ind,j in enumerate(centroids):
                if LB_Keogh(data[i],centroids[j],5)<min_dist:
                    cur_dist=DTWDistance(data[i],centroids[j],w)
                    #cur_dist=Euclid(data[i],centroids[j])
                    if cur_dist<min_dist:
                        min_dist=cur_dist
                        closest_clust=c_ind
            
            if closest_clust in asses:
                asses[closest_clust].append(i)
            else:
                asses[closest_clust]=[]
                
        asses = {i:j for i,j in asses.items() if i is not None}
        #recalculate centroids of clusters
        for key in asses:
            clust_sum=np.zeros(shape=len(data))
            if asses[key]:
                for k in asses[key]:
                    clust_sum = clust_sum + np.transpose(data[k].values)  
                centroids.columns = asses.keys()
                centroids[key]=clust_sum/len(asses[key])
                
    centroids.reindex(index=data.index.values) 
    
    return centroids

#Assign test data to centroids
def assignments(test, centro, w):
    asses={}
    dists = []
    for ind, i in enumerate(test):
        min_dist=float('inf')
        cur_dist=float('inf')
        closest_clust=None    
        for c_ind,j in enumerate(centro):
            if LB_Keogh(test[i],centro[j],5)<min_dist:
                cur_dist=DTWDistance(test[i],centro[j],w)
                #cur_dist=Euclid(data[i],centroids[j])
                if cur_dist<min_dist:
                    min_dist=cur_dist
                    closest_clust=c_ind
        
        if asses.has_key(closest_clust):       
            asses[closest_clust].append(i)
        else:
            asses[closest_clust]=[]
            asses[closest_clust].append(i)

        dists.append((i,min_dist))
        
    return (asses, dists)              

#create the volume curves
#ind = "NKY Index"
fld = ["VOLUME"]
event = ["TRADE"]
sd = "2017-02-01T09:00:00"
ed = "2017-03-01T15:00:00"
iv = 5

sd1 = "2016-09-08T09:00:00"
ed1 = "2016-09-08T15:00:00"

df = pd.DataFrame.from_csv("guchi3.csv")
datacols = ['Side','Strategy','CapG','IntervalReturn','Duration','ExcldIndexRtn','NearPct','Spread','pctMkt','pctPTS','pctDark']
ids = df["Code"].unique()

adv20s, rawcurve, volcurve = bbg_volcurve(ids,event,ed,20,iv,fld)

#Find and return the best centroids and assignments
def find_min(curv, windmax, clustmin, clustmax, iters=3):
    result =[]
    minima =float('inf')
    for wind in range(5,windmax):
        for numclusts in range(clustmin,clustmax):
            print " Window: ", wind, " Clusts: ", numclusts
            roids = k_means_clust(curv,numclusts,iters,wind)
            ass, distas = assignments(curv,roids,wind)
            foo = pd.DataFrame(distas,columns=['sym','dist'])
            avedist = foo['dist'].mean()
            result.append([wind, numclusts, avedist])
            if avedist < minima:
                minima = avedist
                bestroids = roids
                bestass = ass
    return (result, bestroids, bestass)

maxwindsize = 7
minclusts = 15
maxclusts = 20
iterations = 10

results, centroids, asss = find_min(volcurve,maxwindsize,minclusts,maxclusts,iterations)
inv_ass = {value: key for key in asss for value in asss[key]}
df['Cluster'] = df['Code'].astype(str).map(inv_ass)