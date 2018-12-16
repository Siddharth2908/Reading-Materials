
# coding: utf-8

# In[14]:

import pandas as pd
df = pd.read_csv('jan2017.csv',encoding='latin-1')
df.head()


# In[16]:

from io import StringIO
col = ['Label', 'Text']
df = df[col]
df = df[pd.notnull(df['Text'])]
df.columns = ['Label', 'Text']
df['category_id'] = df['Label'].factorize()[0]
category_id_df = df[['Label', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'Label']].values)
df.head()




# In[38]:

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(4,4))
df.groupby('Label').Text.count().plot.bar(ylim=0)
plt.draw()


# In[39]:

#
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.Text).toarray()
labels = df.category_id
features.shape


# In[54]:

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['Label'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)



# In[55]:

NBclf = MultinomialNB().fit(X_train_tfidf, y_train)


# In[56]:

print(NBclf.predict(count_vect.transform(["REFERENCE YOUR SWIFT MSG FIN-199 DT.31.01.17 REGARDING OUR SWIFT PAYMENT REF LORMT/ST/06/17 DTD.09.01.2017 FOR USD13361.00. IFO: 5300796613, GEORGETOWN UNIVERSITY PLEASE READ IN FIELD 79 IN SERIAL NO. 1 AS FOLLOWS: THE NAME OF THE ORDERING PARTY: ''DR. ABDUL MOYEEN KHAN''. . BEST REGARDS RMT.SEC. -}{5:{CHK:2148A27B1B0D}}?"])))


# In[57]:

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
models = [RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),MultinomialNB()]
CV = 10
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
import seaborn as sns
sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=8, jitter=True, edgecolor="blue", linewidth=2)
plt.draw()


# In[58]:

cv_df.groupby('model_name').accuracy.mean()


# In[59]:

#selecting the Linear SVC model


# In[60]:

model = LinearSVC()
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.3, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.Label.values, yticklabels=category_id_df.Label.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


# In[61]:

#check all the incorrect predictions
from IPython.display import display
for predicted in category_id_df.category_id:
  for actual in category_id_df.category_id:
    if predicted != actual and conf_mat[actual, predicted] >= 10:
      print("'{}' predicted as '{}' : {} examples.".format(id_to_category[actual], id_to_category[predicted], conf_mat[actual, predicted]))
      display(df.loc[indices_test[(y_test == actual) & (y_pred == predicted)]][['Label', 'Text']])
      print('')


# In[108]:

from sklearn import metrics
print(metrics.classification_report(y_test, y_pred, target_names=df['Label'].unique()))


# In[63]:

model.fit(features, labels)


# In[98]:

df1 = pd.read_csv('Rpadat.csv',encoding='utf-8')
df1.head()


# In[101]:

df1=pd.DataFrame(df1.iloc[:,0:2])
df1.head()


# In[102]:

df2=df1['Text']
df2=pd.DataFrame(df2)
df2_features = tfidf.transform(df2)
df2.head()


# In[106]:

df1['category_id'] = df1['Label'].factorize()[0]
from io import StringIO
category_id_df = df[['Label', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'Label']].values)


# In[107]:

predictions = model.predict(df2_features)
for text, predicted in zip(df1, predictions):
  print('"{}"'.format(df1))
  print("  - Predicted as: '{}'".format(id_to_category[predicted]))
  print("")

