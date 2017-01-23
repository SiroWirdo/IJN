######################################################################33
#encoding=UTF-8

import sklearn
from sklearn import metrics
import numpy as np
from numpy import dot, array, sum, zeros, outer, any
from numpy.random import random
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import unicodedata
import nltk 
from nltk.collocations import *
import StringIO
import sys
import skfuzzy as fuzz
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from functools import partial
import math
from sklearn import datasets
from array import *
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_distances
from joblib import Parallel, delayed
import multiprocessing
from scipy.spatial.distance import cosine

py3k = sys.version_info >= (3, 0)

if sys.version_info >= (2, 7):
    import unittest
else:
    import unittest2 as unittest


import morfeusz

if py3k:
    def u(s):
        return s
else:
    def u(s):
        return s.decode('UTF-8')


sgjp = 'SGJP' in morfeusz.about()

corpus=dict()
m=2
wor=dict()
wor['MRY']=2
big=dict()

def processInput(filename,w,b):
    with open(filename, 'r') as myfile:
        source=myfile.read()
		
    analysed=analyse(source,filename)
    analysedForBigrams=analyseForBigrams(source,filename)  
    mapa_1=createMapFromText(analysed,w)
           
    list_2=bigramsFromText(analysedForBigrams,b)
    wor[filename]=mapa_1
    big[filename]=list_2

    return [wor,big]



def analyse(text2,filename):
    analysedSource='';  
    target = open('/home/olusiak/Pobrane/analiza2/'+filename, 'w+') 
    for text in text2.split('\n'):
        for line in text.split(' '): #word
            if (len(line)<3): 
                target.write(':')
                analysedSource=analysedSource+' :'
                target.write(' '); 
                continue
            
            if ('n.p.m.' in line):
                continue
               
            try:
                interps = morfeusz.analyse(line)
            except(KeyError):
                continue
            
            for i in interps[0]:
                if (len(i[0])<3):
                    continue

                if ('subst' not in i[2]):
                    continue 
                target.write(i[1].encode('UTF-8'))
                analysedSource=analysedSource+' '+i[1].encode('UTF-8')
                target.write(' ');
    target.close()
        
    return analysedSource

def analyseForBigrams(text2,filename):
    dic=dict()
    analysedSource='';
	
    for text in text2.split('\n'):
        for line in text.split(' '): #word
            if (len(line)<3): 
              
                analysedSource=analysedSource+' :'
                continue
           
            if ('n.p.m.' in line):
                continue
           
            try:
                interps = morfeusz.analyse(line)
            except(KeyError):
                continue
            
            for i in interps[0]: 
                analysedSource=analysedSource+' '+i[1].encode('UTF-8')
                dic[i[1].encode('UTF-8')]=i[2]
    dic[':']='null'
    return [analysedSource,dic]




def createMapFromText(text,WORDS_G):
    mapa=dict()
    for word in text.split(" "):
        word=word.lower()  
        if (word not in mapa):
            mapa[word]=1
        else:
            mapa[word]=mapa[word]+1
    mapa[':']=-1
    maxVal=0
    chosenKey=''
    for k in mapa:
        if mapa[k]>=maxVal:
            chosenKey=k
            maxVal=mapa[k]
    mapa2=dict()
    maxVal=1*maxVal
    minVal=0.9 
    tuples=list()
    for k in mapa:
        if mapa[k]>=minVal and mapa[k]<=maxVal: 
            tuples.append([k,mapa[k]])
       
    tuples.sort(key=lambda tup: tup[1]) 
        
    c=0
    for t in reversed(tuples):
       mapa2[t[0]]=t[1]
       c=c+1
       if (c==WORDS_G):
           break 
        
    return mapa2


def bigramsFromText(text,BIGRAMS_G):
       
    bigram_list=list()
    dic=text[1]  
    line = ""
    for val in text[0]:
        line += val
    tokens = line.split()
        

    bigram_measures = nltk.collocations.BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(tokens)
    finder.apply_freq_filter(1)
        
    counter=0 
    for ngram in finder.score_ngrams (bigram_measures.pmi):
           
        if (('subst' in dic[ngram[0][0]] and 'adj' in dic[ngram[0][1]])):
        
            if (ngram[1]<10.5):
                bigram_list.append(ngram[0]) 
                counter=counter+1   
                if (counter==BIGRAMS_G):
                    break           
    return bigram_list 


################################################################################

class test_analyse:

    WORDS=35
    BIGRAMS=5
    CLUSTERS=5
    RAND=True
    RANDOM=False 

    omega=0.5 
    phi_p=0.5
    phi_g=0.5
    
    def Jfit(self,p):
         
        data=p[0]
        m=p[1]
        cntr=p[2]  

        d = self.getDistance(data, cntr)
        unew = d ** (- 2. / (m - 1))
        unew = np.fmax(unew, np.finfo(np.float64).eps) 
        prawdopKlastrow=unew.sum(axis=0)  
          
        for l in unew:
            cnter=0
            for elem in range(len(l)):
                l[elem]=l[elem]/prawdopKlastrow[cnter]
                cnter+=1
       
        jm=(unew * d ** 2).sum() 
        return jm

    

    def pso(self, fcmResult, data, m, cntr, size=100, maxiter=500):
        
        min_difference=1e-5
        min_difference_fx=1e-5   
        
      
        SWARM = size
        DIMENSIONS = self.CLUSTERS*len(data[0])  # wymiar cząstki
       
        x = np.random.rand(SWARM, DIMENSIONS)  # pozycje cząstek

        
        v = np.zeros_like(x)  # szybkości cząstek
        p = np.zeros_like(x)  # najlepsze pozycje 
        fx = np.zeros(SWARM)  # aktualne wartości Jfit
        
        fp = np.ones(SWARM)*np.inf  # najlepsze Jfit
        g = []  # najlepsza pozycja roju
        fg = np.inf  # najlepsza pozycja (inicjalizacja)
         
    # inicjalizacja pozycji
       
        if fcmResult is not None:
            for i in range(SWARM): 
                x[i]=list(fcmResult)
                l=len(data[0])
                c=int(math.floor(l/10))    
               
                change=np.random.choice(l,c,replace=False)
                
                for cc in change:
                    x[i][cc]=np.random.rand()    
                    
            x[0]=list(fcmResult)  
        else:
            if self.RANDOM:
                r='nothing'
            else: 
                for i in range(SWARM):
                
                    vec=np.random.choice(len(data),self.CLUSTERS,replace=False) #wybór wektorów danych jako cząstek
                    l=len(data[0])
                    c=int(math.floor(l/10))    
                    vectors=list()
                    for index in vec:
                        v=list(data[index])
                        change=np.random.choice(l,c,replace=False)
                        for cc in change:
                            v[cc]=np.random.rand()    #zaszumianie danych
                        vectors.append(v)
                    x[i]=[item for sublist in vectors for item in sublist]   
        
      
    # inicjalizacja fx (oblicz Jfit)
        
        for i in range(SWARM):
            fx[i] = self.Jfit((data,m,x[i, :]))
           
       
    # inicjalizacja p i fp (dzięki x i fx)
       
        for i in range(SWARM):
            p[i, :] = x[i, :].copy()
            fp[i] = fx[i]

    # inicjalizacja g, fg (dzięki p i fp)
        min_index = np.argmin(fp)
        fg = fp[min_index]
        g = p[min_index, :].copy()
       
    # inicjalizacja v
        v =  np.random.rand(SWARM, DIMENSIONS)
       
        it = 0
        while it < maxiter:
            r_p = np.random.uniform(size=(SWARM, DIMENSIONS))
            r_g = np.random.uniform(size=(SWARM, DIMENSIONS))
            v = self.omega*v + self.phi_p*r_p*(p - x) + self.phi_g*r_g*(g - x)
            x = x + v
              
            for i in range(len(x)):      #naprawa ograniczeń wartości 
                for j in range(len(x[i])): 
                    if x[i][j]<0:
                        x[i][j]=x[i][j]*(-1)
                    if x[i][j]>1:
                        x[i][j]=2-x[i][j]  
             
            for i in range(SWARM):
                fx[i] = self.Jfit((data,m,x[i, :]))
                

        # kopiuj najlepsze pozycje
            better_index = (fx < fp)
            p[better_index, :] = x[better_index, :].copy()
            fp[better_index] = fx[better_index]
           
        # wybierz najlepszy z roju
            min_index = np.argmin(fp)
            if fp[min_index] < fg:
                p_min = p[min_index, :].copy()
                step_difference = np.linalg.norm(g - p_min)

                if np.abs(fg - fp[min_index]) < min_difference_fx:
                    return p_min, fp[min_index]
                if step_difference < min_difference:
                    return p_min, fp[min_index]
                
                g = p_min.copy()
                fg = fp[min_index]
            it += 1
    
        return g, fg

















################################  F u z z y   C   M e a n s    ########################################################3

    def fuzzyStep(self, data, u_old, c, m):
    
        u_old = np.fmax(u_old, np.finfo(np.float64).eps) #eliminuje zerowe wartości do minimalnych
        
        #2. Obliczanie centrów
        um = u_old ** m
       
        cntr = um.dot(data)
        prawdopPrzynaleznosiElementuDoKlastrow=um.sum(axis=1)  
        cnter=0  
        for l in cntr:
            for elem in range(len(l)):
                l[elem]=l[elem]/prawdopPrzynaleznosiElementuDoKlastrow[cnter]
            cnter+=1
        
        #3. partition matrix
        d = cosine_distances(data, cntr).T
        d = np.fmax(d, np.finfo(np.float64).eps) #eliminuje zerowe wartości do minimalnych
        jm = (um * d ** 2).sum()
        unew = d ** (- 2. / (m - 1))
       
        prawdopKlastrow=unew.sum(axis=0)  
          
        for l in unew:
            cnter=0
            for elem in range(len(l)):
                l[elem]=l[elem]/prawdopKlastrow[cnter]
                cnter+=1
        jm = (unew * d ** 2).sum()  
        return cntr, unew, jm

    def getDistance(self, data, centers, one=False):
                
        files=len(data)
        counter=0
        row=list() 
        cntrs=np.ndarray(shape=(self.CLUSTERS,len(data[0])))
       
        cou=0
        for cnt in centers:
            
            row.append(cnt)
            counter+=1
            if (counter==len(data[0])):
                counter=0
                cn=list() 
                for r in row:
                    cn.append(r)
                cntrs[cou]=cn
                cou+=1 
                row=list()
       
        return cosine_distances(data, cntrs).T#

    def fuzzycmeans(self, data, c, m, error, maxiter, cntr, start=None,seed=None):
   
        if seed is not None:
            np.random.seed(seed=seed)
    
        if start is None:         
            n = data.shape[0]
            u_start = np.random.rand(c, n)
            u_start /= np.ones(#
                (c, 1)).dot(np.atleast_2d(u_start.sum(axis=0)))
            start = u_start.copy()
        
        u_start = start
        
        if cntr is None and not self.RAND:
            cntr=list()   
            v=np.random.choice(len(data),self.CLUSTERS,replace=False) #wybór wektorów danych jako centrów  
            for ind in v:
                cntr.append(data[ind])
        if cntr is not None:
            
            d = self.getDistance(data, cntr)
            unew = d ** (- 2. / (m - 1))
            unew = np.fmax(unew, np.finfo(np.float64).eps) 
            prawdopKlastrow=unew.sum(axis=0)  
        #  
            for l in unew:
                cnter=0
                for elem in range(len(l)):
                    l[elem]=l[elem]/prawdopKlastrow[cnter]
                    cnter+=1
            u_start=unew 
        
        u = np.fmax(u_start, np.finfo(np.float64).eps)
       
        iter = 0

        u_old = u.copy() 
    
        while iter < maxiter - 1:
            u_old = u.copy()
            [cntr, u, jm] = self.fuzzyStep(data, u_old, c, m)
            iter += 1
            if np.linalg.norm(u - u_old) < error: #pierwiastek z sumy kwadratów
                break

   
        return cntr, u, u_start

###########################    C o r p u s   a n d   V e c t o r i z a t i o n   #####################33

    def analyse(self,text,filename):
        err=False
        analysedSource='';  
        
        target = open('/home/olusiak/Pobrane/analiza2/'+filename, 'w+') 
        for line in text.split(' '):
            if (len(line)<3): 
                target.write(':')
                analysedSource=analysedSource+' :'
                target.write(' '); 
                continue
            
            if ('n.p.m.' in line):
                continue
            
            try:
                interps = morfeusz.analyse(line)
            except(KeyError):
                continue
            
            for i in interps[0]:
                if (len(i[0])<3):
                    continue

                if ('subst' not in i[2]):
                    continue 
                target.write(i[1].encode('UTF-8'))
                analysedSource=analysedSource+' '+i[1].encode('UTF-8')
                target.write(' ');
        target.close()
        
        return analysedSource

    def analyseForBigrams(self,text,filename):
        err=False
        analysedSource='';  
        dic=dict()

        for line in text.split(' '):
            if (len(line)<3): 
                
                analysedSource=analysedSource+' :'
                continue
            
            if ('n.p.m.' in line):
                continue
           
            try:
                interps = morfeusz.analyse(line)
            except(KeyError):
                continue
            
            for i in interps[0]: 
                analysedSource=analysedSource+' '+i[1].encode('UTF-8')
                dic[i[1].encode('UTF-8')]=i[2]
        return [analysedSource,dic]


    


    def openFiles(self,filenames,result):
        corpus=dict()
        counter=1
        vecs=list()
        arrCount=0
        c=0

        for filename in filenames:
            processInput(filename,self.WORDS,self.BIGRAMS)
 
        for filename in filenames:
            mapa_1=wor[filename]
            
            list_2=big[filename]
            for k in mapa_1:
                if (k not in corpus):
                    corpus[k]=counter
                    counter=counter+1 
            for k in list_2:
                k=(k[0].lower(),k[1].lower())
                if (k not in corpus):
                    corpus[k]=counter
                    counter=counter+1
        arrs=np.ndarray(shape=(len(filenames),len(corpus)), dtype=float, order='F')
        for filename in filenames:
            vector=list() 
            mapa_1=wor[filename]
            
            list_2=big[filename]
             
            for k in corpus:
                
                if k in mapa_1 or k in list_2:
                    
                    vector.append(1)
                else:
                    
                    vector.append(0) 
            
            vecs.append(vector)
            
            my_array=np.array('i')
            
            my_array=[xi for xi in vector] 
           
            arrs[arrCount]=my_array
            arrCount=arrCount+1
             
        return corpus, arrs

    def cmeans(self,alldata,centers=None,u_start=None):
          
        cntr, u_orig, u_start = self.fuzzycmeans(alldata, self.CLUSTERS, m, maxiter=2000, error=0.005, cntr=centers, start=u_start)  
        cluster_membership = np.argmax(u_orig, axis=0)
        
        return cntr, u_orig, alldata,m
       
    def createMapFromText(self,text):
        mapa=dict()
        for word in text.split(" "):
            word=word.lower()  
            if (word not in mapa):
                mapa[word]=1
            else:
                mapa[word]=mapa[word]+1
        mapa[':']=-1
        maxVal=0
        chosenKey=''
        for k in mapa:
            if mapa[k]>=maxVal:
                chosenKey=k
                maxVal=mapa[k]
        mapa2=dict()
        maxVal=1*maxVal
        minVal=0.9
        tuples=list()
        for k in mapa:
            if mapa[k]>=minVal and mapa[k]<=maxVal: 
                tuples.append([k,mapa[k]])
        
        tuples.sort(key=lambda tup: tup[1]) 
        
        c=0
        for t in reversed(tuples):
           mapa2[t[0]]=t[1]
           c=c+1
           if (c==self.WORDS):
               break 
        
        return mapa2


    def bigramsFromText(self,text):
        
        bigram_list=list()
        dic=text[1]  
        line = ""
        for val in text[0]:
            line += val
        tokens = line.split()
        

        bigram_measures = nltk.collocations.BigramAssocMeasures()
        finder = BigramCollocationFinder.from_words(tokens)
        finder.apply_freq_filter(1)
        
        counter=0 
        for ngram in finder.score_ngrams (bigram_measures.pmi):
            
            if (('subst' in dic[ngram[0][0]] and 'adj' in dic[ngram[0][1]])):
          
                if (ngram[1]<10.5):
                    bigram_list.append(ngram[0]) 
                    counter=counter+1   
                    if (counter==self.BIGRAMS):
                        break           
        return bigram_list 



############################################################################

# -*- coding: utf-8 -*-
#__author__ = "Joaquim Viegas"
#""" JQM_CV - Python implementations of Dunn and Davis Bouldin clustering validity indices


    def delta(ck, cl):
        values = np.ones([len(ck), len(cl)])*10000
        for i in range(0, len(ck)):
            for j in range(0, len(cl)):
                values[i, j] = np.linalg.norm(ck[i]-cl[j])
        return np.min(values)
    def big_delta(ci):
        values = np.zeros([len(ci), len(ci)])
        for i in range(0, len(ci)):
            for j in range(0, len(ci)):
                values[i, j] = np.linalg.norm(ci[i]-ci[j])
        return np.max(values)

    def dunn(k_list):

        deltas = np.ones([len(k_list), len(k_list)])*1000000
        big_deltas = np.zeros([len(k_list), 1])
        l_range = list(range(0, len(k_list)))
        for k in l_range:
            for l in (l_range[0:k]+l_range[k+1:]):
                deltas[k, l] = delta(k_list[k], k_list[l])
            big_deltas[k] = big_delta(k_list[k])
        di = np.min(deltas)/np.max(big_deltas)
        return di

    def delta_fast(self,ck, cl, distances):
        values = distances[np.where(ck)][:, np.where(cl)]
        values = values[np.nonzero(values)]
        return np.min(values)
    def big_delta_fast(self,ci, distances):
        values = distances[np.where(ci)][:, np.where(ci)]
        return np.max(values)


    def dunn_fast(self,points, labels):

        distances = euclidean_distances(points)
        ks = np.sort(np.unique(labels))
        deltas = np.ones([len(ks), len(ks)])*1000000
        big_deltas = np.zeros([len(ks), 1])
        l_range = list(range(0, len(ks)))
        for k in l_range:
            for l in (l_range[0:k]+l_range[k+1:]):
                deltas[k, l] = self.delta_fast((labels == ks[k]), (labels == ks[l]), distances)
            big_deltas[k] = self.big_delta_fast((labels == ks[k]), distances)
        di = np.min(deltas)/np.max(big_deltas)
        return di

    def big_s(self,x, center):
        len_x = len(x)
        total = 0
        for i in range(len_x):
            total += np.linalg.norm(x[i]-center)
        if len_x==0: #
            return 0 #
        return total/len_x

    def davisbouldin(self,k_list, k_centers):

        len_k_list = len(k_list)
        big_ss = np.zeros([len_k_list], dtype=np.float64)
        d_eucs = np.zeros([len_k_list, len_k_list], dtype=np.float64)
        db = 0
        for k in range(len_k_list):
            big_ss[k] = self.big_s(k_list[k], k_centers[k])

        for k in range(len_k_list):
            for l in range(0, len_k_list):
                d_eucs[k, l] = np.linalg.norm(k_centers[k]-k_centers[l])

        for k in range(len_k_list):
            values = np.zeros([len_k_list-1], dtype=np.float64)
            for l in range(0, k):
                values[l] = (big_ss[k] + big_ss[l])/d_eucs[k, l]
            for l in range(k+1, len_k_list):
                values[l-1] = (big_ss[k] + big_ss[l])/d_eucs[k, l]
            db += np.max(values)
        res = db/len_k_list
        return res
         
def start():
    xy=list()
    labels_true=list()

  #  xy.append("wind")
   # labels_true.append(0)
   # xy.append("linux")
   # labels_true.append(0)
   # xy.append("wind2")
   # labels_true.append(0)
   # xy.append("wojna1")
  #  labels_true.append(1)
 #   xy.append("wojna2")
 #   labels_true.append(1)
  #  xy.append("wojna3")
  #  labels_true.append(1)
  #  xy.append("wojna4")
  #  labels_true.append(1)
#xy.append("panw") 
  #  xy.append("historia1")
  #  labels_true.append(1)
  #  xy.append("wojnapolniem")
  #  labels_true.append(1)
  #  xy.append("historia2")
  #  labels_true.append(1)
  #  xy.append("historia3")
  #  labels_true.append(1)
  #  xy.append("bitwa1")
  #  labels_true.append(1)
					
    xy.append("alzheimer.txt") 
    labels_true.append(3)
    xy.append("astma.txt")
    labels_true.append(3)
    xy.append("budka_suflera.txt")
    labels_true.append(4)
    xy.append("dream_theater.txt")
    labels_true.append(4)
    xy.append("dzem.txt")
    labels_true.append(4)
    xy.append("enej.txt")
    labels_true.append(4)
    xy.append("hokej na lodzie.txt")
    labels_true.append(5)
    xy.append("infekcyjne_zapalenie_wsierdzia.txt")
    labels_true.append(3)
    xy.append("kombi.txt")
    labels_true.append(4)
    xy.append("korn.txt")
    labels_true.append(4)
    xy.append("lady_pank.txt")
    labels_true.append(4)
    xy.append("maraton.txt")
    labels_true.append(5)

    xy.append("metallica.txt")
    labels_true.append(4)
    xy.append("padaczka.txt")
    labels_true.append(3)
    xy.append("parkinson.txt")
    labels_true.append(3)
    xy.append("perfect.txt")
    labels_true.append(4)
    xy.append("pilka_nozna.txt")
    labels_true.append(5)
    xy.append("saturn.txt")
    labels_true.append(11)
    xy.append("slonce.txt")
    xy.append("pilka_reczna.txt")
    labels_true.append(5)
    xy.append("plyanie.txt")
    labels_true.append(5)
    xy.append("riverside.txt")
    labels_true.append(4)
    xy.append("schizofrenia.txt")
    labels_true.append(3)
    xy.append("siatkowka.txt")
    labels_true.append(5)
    xy.append("skok_wzwyz.txt")
    labels_true.append(5)
    xy.append("tenis.txt")
    labels_true.append(5)
    xy.append("pluton.txt")
    labels_true.append(11)

    labels_true.append(11)
    xy.append("uran.txt")
    labels_true.append(11)
    xy.append("wenus.txt")
    labels_true.append(11)


    xy.append("tenis_stolowy.txt")
    labels_true.append(5)
    xy.append("unihokej.txt")
    labels_true.append(5)
    xy.append("zapalenie_opon_mozgowych.txt")
    labels_true.append(3)
    xy.append("zapalenie_pluc.txt")
    labels_true.append(3)
    xy.append("zapalenie_zatok_przynosowych.txt")
    labels_true.append(3)
    xy.append("zawal_serca.txt")
    labels_true.append(3)

#xy.append("jangielski.txt")
 #   labels_true.append(10)
  #  xy.append("jarabski.txt")
  #  labels_true.append(10)
  #  xy.append("jchinski.txt")
  #  labels_true.append(10)
  #  xy.append("jfinski.txt")
  #  labels_true.append(10)
  #  xy.append("jfrancuski.txt")
  #  labels_true.append(10)
   # xy.append("jhiszpanski.txt")
   # labels_true.append(10)
   # xy.append("jniemiecki.txt")
   # labels_true.append(10)
    xy.append("jowisz.txt")
    labels_true.append(11)
    #xy.append("jpolski.txt")
    #labels_true.append(10)
    #xy.append("jportugalski.txt")
    #labels_true.append(10)
    xy.append("mars.txt")
    labels_true.append(11)
    xy.append("merkury.txt")
    labels_true.append(11)
    xy.append("neptun.txt")
    labels_true.append(11)

    #xy.append("wloski.txt")
    #labels_true.append(11)
    xy.append("ziemia.txt")
    labels_true.append(11)   

    xy.append("austria.txt")  
    labels_true.append(6)
    xy.append("bach.txt")
    labels_true.append(7)
    xy.append("bmw.txt")
    labels_true.append(8)
    xy.append("brahms.txt")
    labels_true.append(7)
    xy.append("c.txt")
    labels_true.append(9)
    xy.append("c#.txt")
    labels_true.append(9)
    xy.append("c++.txt")
    labels_true.append(9)
    xy.append("chopin.txt")
    labels_true.append(7)
    xy.append("czechy.txt")
    labels_true.append(6)
    xy.append("ferarri.txt")
    labels_true.append(8)
    xy.append("fiat.txt")
    labels_true.append(8)
    xy.append("ford.txt")
    labels_true.append(8)
    xy.append("fordmustang.txt")
    labels_true.append(8)
    xy.append("francja.txt")
    labels_true.append(6)
    xy.append("hiszpania.txt")
    labels_true.append(6)
    xy.append("java.txt")
    labels_true.append(9)
    xy.append("javascript.txt")
    labels_true.append(9)
    xy.append("liszt.txt")
    labels_true.append(7)
    xy.append("litwa.txt")
    labels_true.append(6)
    xy.append("mercedes.txt")
    labels_true.append(8)
    xy.append("moniuszko.txt")
    labels_true.append(7)
    xy.append("mozart.txt")
    labels_true.append(7)
    xy.append("niemcy.txt")
    labels_true.append(6)
    xy.append("perl.txt")
    labels_true.append(9)
    xy.append("peugeot.txt")
    labels_true.append(8)
    xy.append("php.txt")
    labels_true.append(9)
    xy.append("polonez.txt")
    labels_true.append(8)
    xy.append("polska.txt")
    labels_true.append(6)
    xy.append("python.txt")
    labels_true.append(9)
    xy.append("rachmaninow.txt")
    labels_true.append(7)
    xy.append("renault.txt")
    labels_true.append(8)
    xy.append("rossini.txt")
    labels_true.append(7)
    xy.append("ruby.txt")
    labels_true.append(9)
    xy.append("schumann.txt")
    labels_true.append(7)
    xy.append("slowacja.txt")
    labels_true.append(6)
    xy.append("szpilman.txt")
    labels_true.append(7)
    xy.append("ukraina.txt")
    labels_true.append(6)
    xy.append("usa.txt")
    labels_true.append(6)
    xy.append("visualbasic.txt")
    labels_true.append(9)
    xy.append("volkswagen.txt")
    labels_true.append(8)  

    rrand=list()
    rrand.append(True)
    rrand.append(False)

    cclusters=range(2,11)
    cclusters=list()
    cclusters.append(8)
    pparameters=np.arange(0.3,1.01,0.1)
    pparameters=list()
    pparameters.append(0)
    wwords=range(5,21,5)
    bbigs=range(5,21,5)
    wwords=list()
    wwords.append(35)
    bbigs=list()
    bbigs.append(5)
    #bbigs.append(25)
    #bbigs.append(25) 
    
    num_cores = multiprocessing.cpu_count()
    for c in cclusters:
        for w in wwords:
            for p1 in pparameters:
                for p2 in pparameters:
                   for om in pparameters: 
                       Parallel(n_jobs=num_cores)(delayed(forLoopClusters)(c,xy,labels_true) for b in bbigs) 

def forLoopClusters(c,xy,labels_true):
    print '>;',c,';'
    wor=dict()
    big=dict()
    x = test_analyse()
    x.CLUSTERS=c  
    m=2
    corpus, arrs=x.openFiles(xy,"resultFile")
    best, _ = x.pso(None, arrs, m, None)
    cntr,u_old,data,m=x.cmeans(arrs,centers=best)
    print 'u ',u_old
    labels_pred = np.argmax(u_old, axis=0)     
    try:
        silh=metrics.silhouette_score(data,labels_pred)       
    except:
        silh=-2 
    db=list()
    k_list=list()
    for i in range(c):         
        db.append(list())
    counter=0
    for i in labels_pred:
        db[i].append(data[counter])
        counter+=1
    for i in range(len(db)):
        listaKlastra=db[i]
        arrs=np.ndarray(shape=(len(listaKlastra),len(data[0])), dtype=float, order='F')
        count=0
        for elem in listaKlastra:
            arr=np.array('i')
            arr=[xi for xi in elem]
            arrs[count]=arr
            count+=1
        k_list.append(arrs) 
    davis=x.davisbouldin(k_list,cntr)            
    dunn=x.dunn_fast(data,labels_pred)
    score=metrics.adjusted_rand_score(labels_true, labels_pred)
    print '>;',c,';',score,' ... ' ,silh,'; ',dunn,';',davis,' -> ',labels_pred

def forLoop(w,b,c,xy,labels_true,p1,p2,om):

    m=2
    wor=dict()
    big=dict()
    x = test_analyse()
    x.WORDS=w
    x.BIGRAMS=b
    x.CLUSTERS=c
    x.phi_p=p1
    x.phi_g=p2
    x.omega=om   
    corpus, arrs=x.openFiles(xy,"resultFile")
    best, _ = x.pso(None, arrs, m, None)
    cntr,u_old,data,m=x.cmeans(arrs,centers=best)
    labels_pred = np.argmax(u_old, axis=0)     
    try:
        silh=metrics.silhouette_score(data,labels_pred)       
    except:
        silh=-2 
    score=metrics.adjusted_rand_score(labels_true, labels_pred)
    score2=metrics.normalized_mutual_info_score(labels_true,labels_pred)
    score3=metrics.mutual_info_score(labels_true,labels_pred)
    score4=metrics.v_measure_score(labels_true, labels_pred)
                
    dunn=x.dunn_fast(data,labels_pred)
    print '>',w,'; ',b,'; ',p1,', ',p2,', ',om,' = ',score,'; ',score2,'; ',score3,'; ',silh,'; ',dunn,' -> ',labels_pred

def read(filename):
    filename='/home/olusiak/Pobrane/python-morfeusz-0.3300/'+filename
    with open(filename, 'r') as myfile:
        source=myfile.read()
    sort=dict()
    for line in source.split('\n'):
        if len(line)>0:
            first_char = line[0] 
            if first_char=='>':
                try:
                    spl=line.split(';')
                    spl=float(spl[2]) 
                    sort[line]=spl
                except:
                    continue
    tuples=list()
    for k in sort:
        tuples.append([k,sort[k]])
       
    tuples.sort(key=lambda tup: tup[1])

start() 
