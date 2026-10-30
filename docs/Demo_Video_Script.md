# Demo Video Script (3–4 minutes)

## Scene 1 — Introduction (0:00–0:25)
**On-screen:** Project title slide.

**Voice-over:** “Hello. This project is Product Review Sentiment Analysis Using Machine Learning. It analyzes electronics and computer-product review text and classifies each review as positive, negative, or neutral.”

## Scene 2 — Problem and approach (0:25–0:55)
**On-screen:** Show `docs/architecture.png`.

**Voice-over:** “Reading large numbers of product reviews manually takes time. Our system uses text preprocessing, TF-IDF feature extraction, and a Logistic Regression classifier. The trained model is then used through a Streamlit dashboard.”

## Scene 3 — Dataset disclosure (0:55–1:20)
**On-screen:** Open `data/input/product_reviews_sample.csv`.

**Voice-over:** “The repository currently includes a small synthetic demonstration dataset with three balanced sentiment classes. It is included so the complete pipeline can run immediately. It is not scraped customer data. For final real-world evaluation, this file should be replaced with a permitted, labelled product-review dataset.”

## Scene 4 — Run the application (1:20–1:45)
**On-screen:** Terminal commands.

```bash
pip install -r requirements.txt
python src/train.py
streamlit run app.py
```

**Voice-over:** “First, we install the dependencies and train the model. The training script saves the model, test predictions, confusion matrix, and evaluation metrics. Then we launch the Streamlit application.”

## Scene 5 — Live prediction (1:45–2:40)
**On-screen:** Enter these example reviews in the app:

- “Excellent battery life and great sound quality. I am very happy.”
- “The device stopped working after two days. Very disappointing.”
- “It works as expected and has average features.”

**Voice-over:** “Here I enter a product review and click Analyze sentiment. The application displays the predicted sentiment and the probability scores for each class.”

## Scene 6 — Dataset dashboard and outputs (2:40–3:15)
**On-screen:** Explore dataset tab; show `data/output/metrics.json`, `test_predictions.csv`, and `confusion_matrix.csv`.

**Voice-over:** “The Explore Dataset tab shows the sample records and sentiment distribution. The output folder stores the held-out predictions, confusion matrix, and evaluation metrics.”

## Scene 7 — Conclusion and limitations (3:15–3:40)
**On-screen:** Final slide.

**Voice-over:** “The project demonstrates an end-to-end sentiment classification workflow. The current dataset is synthetic, so the reported score only verifies the demonstration pipeline. The next step is to retrain and evaluate the model on authentic, permitted product reviews. Thank you.”

---

## Recording checklist
- Record the screen at 1080p if possible.
- Keep terminal commands readable and hide personal folders/usernames if desired.
- Use subtitles or the supplied narration; ensure the synthetic-data disclosure remains visible.
- Do not present the synthetic-data accuracy as real-world model performance.
