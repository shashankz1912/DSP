

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


movies = pd.read_csv('tmdb_5000_movies.csv')  
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[3]:


movies.head()


# In[4]:


credits.head(1)


# In[5]:


movies.merge(credits,on='title')


# In[6]:


movies = movies.merge(credits,on='title')


# In[7]:


movies.head(1)


# In[8]:


movies = movies[['movie_id','title','overview','cast','crew','keywords','genres']]


# In[9]:


movies.head()


# In[10]:


movies.isnull().sum()


# In[11]:


movies.dropna(inplace= True) #dropping the empty column#


# In[12]:


movies.duplicated().sum() 


# In[13]:


def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
           L.append(i['name'])
    return L
           


# In[14]:


import ast


# In[15]:


movies['keywords'] = movies['keywords'].apply(convert)


# In[16]:


movies['keywords']


# In[17]:


def convert3(obj):
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L


# In[18]:


movies['cast']=movies['cast'].apply(convert3)


# In[19]:


movies['cast']


# In[20]:


movies.head()


# In[21]:


movies['crew'][0]


# In[22]:


def fetch_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name'])
            break
    return L


# In[23]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[24]:


movies.head()


# In[25]:


movies['genres']=movies['genres'].apply(convert)


# In[26]:


movies.head()


# In[27]:


movies['overview'] = movies['overview'].apply(lambda x:x.split()) #to convert string to list


# In[28]:


movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","")for i in x])
movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","")for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","")for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","")for i in x])


# In[29]:


movies.head()


# In[30]:


movies['tags']= movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[31]:


movies.head()


# In[32]:


new_df = movies[['movie_id','title','tags']]


# In[33]:


new_df


# In[34]:


new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))


# In[35]:


new_df.head()


# In[36]:


new_df['tags'][0]


# In[37]:


new_df.head()


# In[38]:


import nltk


# In[39]:


get_ipython().system('pip install nltk')


# In[40]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[41]:


new_df['tags']=new_df['tags'].apply(stem)
new_df['tags']


# In[42]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words = 'english')


# In[43]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[44]:


vectors


# In[45]:


cv.get_feature_names()


# In[ ]:


stem('In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. Action Adventure Fantasy ScienceFiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d SamWorthington ZoeSaldana SigourneyWeaver JamesCameron')


# In[49]:


pip install scikit-learn


# In[50]:


from sklearn import metrics


# In[51]:


from sklearn.metrics.pairwise import cosine_similarity


# In[52]:


similarity = cosine_similarity(vectors)


# In[53]:


sorted(list(enumerate(similarity[0])),reverse = True,key = lambda x:x[1])[1:6]


# In[54]:


similarity


# In[55]:


def recommend(movie):
    movie_index = new_df[new_df['title']==movie].index[0]
    distances = similarity[movie_index]
    movies_list= sorted(list(enumerate(distances)),reverse = True,key = lambda x:x[1])[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
        


# In[56]:


recommend('Avatar')


# In[57]:


import pickle


# In[58]:


pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))


# In[59]:


new_df['title'].values


# In[60]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:




