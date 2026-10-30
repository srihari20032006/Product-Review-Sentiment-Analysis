from pathlib import Path
import json, joblib, pandas as pd, streamlit as st
ROOT=Path(__file__).resolve().parent
MODEL=ROOT/'models/sentiment_model.joblib'
st.set_page_config(page_title='Product Review Sentiment Analyzer',page_icon='📊',layout='wide')
st.title('📊 Product Review Sentiment Analyzer')
st.caption('Classifies electronics product reviews as positive, negative or neutral using TF-IDF + Logistic Regression.')
if not MODEL.exists(): st.error('Model not found. Run: python src/train.py'); st.stop()
model=joblib.load(MODEL)
tab1,tab2=st.tabs(['Analyze a review','Explore dataset'])
with tab1:
 review=st.text_area('Enter a product review',placeholder='Example: Battery life is excellent and the sound quality is clear.')
 if st.button('Analyze sentiment',type='primary'):
  if review.strip():
   label=model.predict([review])[0]; probs=model.predict_proba([review])[0]; classes=model.classes_
   st.subheader('Prediction: '+label.title())
   st.write('Model confidence (largest class probability):',f'{max(probs):.1%}')
   st.bar_chart(pd.DataFrame({'probability':probs},index=classes))
  else: st.warning('Please enter a review first.')
with tab2:
 data_path=ROOT/'data/input/product_reviews_sample.csv'
 if data_path.exists():
  df=pd.read_csv(data_path); c1,c2,c3=st.columns(3); c1.metric('Reviews',len(df)); c2.metric('Product categories',df.product_category.nunique()); c3.metric('Sentiment classes',df.sentiment.nunique())
  st.bar_chart(df.sentiment.value_counts()); st.dataframe(df.head(30),use_container_width=True)
  st.info('The bundled dataset is synthetic demonstration data, not real customer reviews. Replace it with a properly sourced dataset for final evaluation.')
