from pathlib import Path
import json, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
ROOT=Path(__file__).resolve().parents[1]
df=pd.read_csv(ROOT/'data/input/product_reviews_sample.csv')
required={'review_text','sentiment'}
if not required.issubset(df.columns): raise ValueError(f'CSV must contain columns: {required}')
df=df.dropna(subset=['review_text','sentiment']); X_train,X_test,y_train,y_test=train_test_split(df.review_text.astype(str),df.sentiment.astype(str),test_size=.2,random_state=42,stratify=df.sentiment)
model=Pipeline([('tfidf',TfidfVectorizer(lowercase=True,ngram_range=(1,2),max_features=12000,sublinear_tf=True)),('classifier',LogisticRegression(max_iter=1000,class_weight='balanced',random_state=42))])
model.fit(X_train,y_train); pred=model.predict(X_test)
(ROOT/'models').mkdir(exist_ok=True); (ROOT/'data/output').mkdir(parents=True,exist_ok=True)
joblib.dump(model,ROOT/'models/sentiment_model.joblib')
pd.DataFrame({'review_text':X_test.values,'actual_sentiment':y_test.values,'predicted_sentiment':pred}).to_csv(ROOT/'data/output/test_predictions.csv',index=False)
cm=confusion_matrix(y_test,pred,labels=sorted(df.sentiment.unique()))
pd.DataFrame(cm,index=['actual_'+x for x in sorted(df.sentiment.unique())],columns=['pred_'+x for x in sorted(df.sentiment.unique())]).to_csv(ROOT/'data/output/confusion_matrix.csv')
metrics={'accuracy':accuracy_score(y_test,pred),'classification_report':classification_report(y_test,pred,output_dict=True,zero_division=0),'training_samples':len(y_train),'test_samples':len(y_test)}
(ROOT/'data/output/metrics.json').write_text(json.dumps(metrics,indent=2)); print(json.dumps(metrics,indent=2))
