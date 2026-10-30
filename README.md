# Product Review Sentiment Analysis

A small end-to-end ML project for classifying electronics product reviews into positive, negative and neutral sentiment.

## Important dataset disclosure
The included 300-row CSV is **synthetic demonstration data** created to make the project runnable immediately. It is not scraped from Amazon/Flipkart and must not be described as genuine customer feedback. For a final academic evaluation, replace `data/input/product_reviews_sample.csv` with a permitted, labelled real-world review dataset (retain columns `review_text`, `sentiment`; optional `product_category`). Re-run training and update the report with the new measured results.

## Stack
Python, Pandas, scikit-learn, TF-IDF, Logistic Regression, Streamlit.

## Run locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/train.py
streamlit run app.py
```

## Dataset format
CSV columns: `review_text`, `sentiment` (positive/negative/neutral), and optionally `product_category`.

## Project structure
- `data/input/`: original labelled CSV
- `data/intermediate/`: preprocessed CSV
- `data/output/`: test predictions, confusion matrix, metrics
- `models/`: serialized trained pipeline
- `src/train.py`: reproducible training/evaluation
- `app.py`: interactive Streamlit demo
- `docs/`: architecture and submission material

## Evaluation
The included metrics are calculated on a held-out split of the synthetic demonstration dataset. They demonstrate pipeline execution only; they do not estimate performance on genuine customer reviews.

## GitHub collaborator
Create a GitHub repository and invite the faculty account `gundimedavenugopal` under Settings → Collaborators. Do not publish private credentials or tokens.
