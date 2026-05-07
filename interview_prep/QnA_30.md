# ChurnLens — 30 Interview Q&A
### Ready-to-speak answers. No jargon dumps. Just clear, confident responses.

---

## 🎯 PROJECT OVERVIEW

---

**Q1. Tell me about this project in 30 seconds.**

ChurnLens is a churn prediction system I built for the fitness industry. It uses a fine-tuned BERT model trained on 391K real Yelp gym reviews to predict member sentiment and churn risk from review text. But I didn't stop at just NLP — I added a behavioral risk engine that scores things like visit frequency and class bookings to catch the "silent churners" who never write reviews but quietly stop showing up. Both signals are blended into a hybrid churn score. The whole thing is served through a FastAPI backend and a Streamlit dashboard where gym operators can upload CSVs and see per-location analytics.

---

**Q2. Why did you build this? What's the business problem?**

Gym churn runs at about 30 to 50 percent annually — that's massive revenue loss. The problem is that gyms usually find out a member is leaving only after they've already stopped coming. By then it's too late to intervene. I wanted to flip that — detect churn signals early from what members are saying in reviews, and also from how they're behaving (visit patterns, class engagement). If a gym operator sees a location with 35% churn risk and the top complaint is "equipment," they know exactly what to fix and who to reach out to.

---

**Q3. Who would actually use this in the real world?**

Three personas. A gym manager would paste in a single review to instantly check if that member is at risk. An operations team would upload a batch CSV of hundreds of reviews and get a risk report they can act on. And for a gym chain with multiple locations, the dashboard shows which branches are healthy and which are bleeding members — so HQ can prioritize where to invest.

---

**Q4. What makes this project different from a typical sentiment analysis project?**

Two things. First, it's multi-task — one BERT model predicts both sentiment and churn risk simultaneously using shared representations, not two separate models. Second, it's multi-signal — I don't just rely on what people write. The behavioral risk engine catches members who are silently disengaging. A member could write "great gym!" but if they've only visited once in the last month and haven't booked any classes, that's a red flag the text alone would never catch.

---

## 📊 DATA & LABELING

---

**Q5. Where did you get the training data?**

The Yelp Open Dataset — it's publicly available for academic and research use. It contains 6.9 million reviews across 150K businesses. I filtered it down to 18,857 fitness-related businesses like gyms, yoga studios, CrossFit boxes — which gave me 391,357 fitness-specific reviews. That's a large, real-world, domain-specific dataset with authentic messy language.

---

**Q6. How did you create labels without manually annotating 391K reviews?**

I used a technique called weak supervision. For sentiment, Yelp already provides star ratings — a 5-star review is almost always positive, and 1-star is almost always negative. So I mapped 4-5 stars to positive, 3 stars to neutral, and 1-2 stars to negative. For churn risk, I built a keyword lexicon — if a review contains phrases like "cancelling my membership," "switching gyms," or "never coming back," it gets flagged as a churn signal. At 391K samples, the noise from edge cases averages out.

---

**Q7. Isn't keyword-based churn labeling noisy? What about false positives?**

Absolutely, it's not perfect. A review like "I'd never cancel, this place is amazing" would be a false positive. That's roughly 20% label noise. But here's the thing — at 391K samples, statistical noise washes out. We also use class weights during training so the model doesn't just predict the majority class. And honestly, this limitation is exactly why I built the behavioral risk engine — to complement the text signal with actual usage patterns. In production, you'd replace keyword labels with real CRM cancellation records as ground truth.

---

**Q8. What's the class distribution? How did you handle imbalance?**

Positive reviews make up 68%, negative 26%, and neutral only 6%. For churn, only about 10% of reviews signal churn risk. If you train naively, the model would just predict "positive" for everything and still get 68% accuracy. So I used inverse-frequency class weights in the loss function — rare classes like neutral get a 5.5x higher penalty when misclassified. This forces the model to pay equal attention to all classes rather than gaming the accuracy metric.

---

## 🧠 THE MODEL

---

**Q9. Why did you choose BERT? Why not something simpler like TF-IDF with Logistic Regression?**

Simpler models treat text as a bag of words — they lose word order and context completely. A phrase like "not bad" would confuse them because "not" and "bad" are both negative words individually. BERT understands context bidirectionally — it reads left-to-right AND right-to-left simultaneously. For domain-specific text like gym reviews where people use sarcasm, negation, and industry jargon, BERT significantly outperforms bag-of-words approaches. The tradeoff is compute cost, which I handled by using Google Colab's free T4 GPU.

---

**Q10. What does "dual-head" mean in your architecture?**

It means one shared BERT encoder feeds into two separate classification heads. The first head predicts sentiment across 3 classes — positive, neutral, negative. The second head predicts churn risk as a binary yes or no. Both heads share the same BERT backbone, which means features learned for sentiment also help churn prediction. Negative reviews tend to correlate with churn, so the shared layers capture that relationship naturally. It's also 2x more efficient than running two separate models.

---

**Q11. What is the CLS token and why do you use it?**

BERT adds a special CLS token at the very beginning of every input. After the text passes through all 12 transformer layers, this token's hidden state becomes a 768-dimensional vector that summarizes the meaning of the entire review. Think of it as a compressed fingerprint of the full text. I feed this single vector into both classification heads to make predictions.

---

**Q12. What is fine-tuning? Why not train BERT from scratch?**

Fine-tuning means starting from pre-trained weights — BERT was already trained on billions of words from Wikipedia and BookCorpus, so it already understands English grammar, syntax, semantics. I just continue training on my gym reviews to teach it domain-specific patterns. Training from scratch would need massive compute — we're talking weeks on 64 TPUs, costing tens of thousands of dollars. Fine-tuning took about 3.5 hours on a single free GPU. The analogy I use is: BERT already speaks fluent English, I'm just teaching it gym industry vocabulary.

---

**Q13. Explain your loss function. Why the 60/40 split?**

The total loss is a weighted combination — 60% from sentiment cross-entropy loss and 40% from churn binary cross-entropy loss. Sentiment gets higher weight because those labels are more reliable — they come from star ratings, which are a strong signal. Churn labels are noisier because they're keyword-based. By weighting sentiment higher, the model's shared layers are anchored by the more reliable task, and churn benefits from the learned representations without dominating the training.

---

**Q14. What training setup did you use?**

Google Colab free tier with a Tesla T4 GPU, 16 gigs of VRAM. I used the AdamW optimizer at a learning rate of 2e-5 with linear warmup for the first 10% of steps, then linear decay. Mixed precision training with FP16 to halve memory usage and double speed. Batch size 32, max sequence length 256 tokens, trained for 3 epochs with early stopping. Total training time about 3.5 hours on 313K training samples.

---

**Q15. Why such a tiny learning rate? What happens if you use a bigger one?**

BERT's pre-trained weights are already excellent — they encode deep language understanding from billions of words. A large learning rate would make aggressive updates that destroy these pre-trained weights. This is called catastrophic forgetting. The learning rate 2e-5 is the sweet spot — large enough to adapt to our gym domain but small enough to preserve what BERT already knows. The warmup phase gradually ramps up the learning rate so the randomly initialized classification heads can stabilize before we start making big updates.

---

**Q16. What results did you achieve?**

93.25% F1 score for sentiment classification and 86.41% F1 for churn detection, both measured on a held-out test set of 39K samples that the model never saw during training. I use F1 for churn instead of accuracy because accuracy is misleading with imbalanced data — a model that predicts "no churn" for everything would get 90% accuracy but catch zero actual churners. F1 balances precision and recall, which is what matters for churn detection.

---

## 🔀 MULTI-SIGNAL ARCHITECTURE

---

**Q17. Explain the behavioral risk engine. How does it work?**

It's a rule-based scoring system that evaluates three behavioral signals from the CSV data. First, visit frequency — if a member visited only once or less in the last month, that's a 45% risk contribution because they've essentially stopped coming. Second, class engagement — zero classes booked adds 25% risk because engaged members book classes. Third, membership tenure combined with visit drop — a member who's been around for over a year but suddenly stopped visiting gets an extra 20% risk bump because that's the "veteran burnout" pattern. These scores are summed and capped at 1.0.

---

**Q18. Why rules instead of another ML model for behavioral scoring?**

Three practical reasons. First, we don't have real cancellation labels to train on — without ground truth, you can't train a supervised model. Second, the risk factors are well-understood business heuristics — "low visits equals high risk" doesn't need a neural network to figure out. Third, interpretability — a gym manager can understand "this member only visited once this month" much better than "the model output 0.73." In production with real CRM cancellation data, I'd replace this with a gradient boosted model like XGBoost trained on actual historical cancellations.

---

**Q19. How does the hybrid blend work?**

I take the BERT text churn confidence and the behavioral risk score, weight them 50/50, and produce a hybrid churn score between 0 and 1. If the hybrid score exceeds 0.35, the member is flagged as at-risk. The power of this approach is in the edge cases — a member who writes "love this gym!" but hasn't visited in three weeks would be missed by text-only analysis. The behavioral signal catches them. Conversely, a member who says "thinking about cancelling" but still visits regularly gets a moderate score, not an extreme one. The blend captures reality better than either signal alone.

---

**Q20. What's the biggest technical decision you made in this project?**

The dual-signal architecture. I could have shipped a perfectly fine BERT-based sentiment and churn detector and called it done. But I recognized the fundamental limitation — most churners are silent. They don't write reviews, they just leave. Adding the behavioral engine transformed this from a text classification demo into something that could actually work in production. It also makes a much stronger interview talking point because it shows I think about real-world applicability, not just model accuracy on a test set.

---

## ⚙️ ENGINEERING & TECH STACK

---

**Q21. Walk me through your tech stack and justify each choice.**

PyTorch for the model because it's the standard for research and production NLP, and HuggingFace Transformers for the BERT tokenizer and pre-trained weights. FastAPI for the backend because it's async-first, auto-generates OpenAPI docs, and has built-in Pydantic validation — it's basically Flask but faster and with less boilerplate. Streamlit for the dashboard because it lets you build interactive data apps in pure Python without writing any frontend code. Plotly for interactive charts. Pandas for data wrangling. Docker for reproducible deployments. And Parquet for data storage because it's 5-10x smaller than CSV with typed columns.

---

**Q22. Why FastAPI over Flask?**

FastAPI is async by default, meaning it handles concurrent requests without blocking. It automatically generates interactive API docs at the slash-docs endpoint. It uses Pydantic for request validation — if someone sends wrong data types, the error response is detailed and automatic. Benchmarks show FastAPI handles 2-3x more requests per second than Flask. And type hints are enforced at the framework level, catching bugs at development time rather than production.

---

**Q23. How does your API handle model loading efficiently?**

I use the singleton pattern. The BERT model is about 1.5 GB and takes roughly 5 seconds to load from disk. Without optimization, every API request would re-load the model — that's 5 seconds of latency per request plus memory explosion. Instead, I load the model once on the first request and store it in a module-level variable. Every subsequent request reuses the same loaded model instance. This gives sub-200ms response times after the initial load.

---

**Q24. How do you handle large batch requests without crashing?**

Chunked processing. When someone uploads 500 reviews, I don't feed all 500 into BERT at once — that would exceed memory limits. Instead, I split the batch into chunks of 16, process each chunk through the model, and concatenate the results. This keeps memory usage predictable regardless of batch size. The API accepts up to 500 reviews per batch request, which is validated through Pydantic schema constraints.

---

**Q25. What's Demo Mode and why did you build it?**

Demo Mode is a rule-based fallback that activates when no trained BERT model checkpoint exists on disk. It uses keyword matching and simple heuristics to generate predictions — not as accurate as BERT, but good enough to demonstrate the full workflow. I built it so that anyone can clone the repo and immediately run the entire system — API, dashboard, everything — without needing to train a model first. It also prevents the application from crashing with a FileNotFoundError during development.

---

## 📈 DASHBOARD & UX

---

**Q26. How does the dashboard work?**

Two tabs. The Review Analyzer tab is for single reviews — paste one review, hit analyze, and you instantly see sentiment with confidence, churn risk flag, and detected complaint themes. The Batch Analysis tab is the main product — upload a CSV with review text and optional location and behavioral columns. The system runs BERT inference, scores behavioral risk, blends them, and renders everything on one page: KPI cards, churn-by-location bar charts, sentiment stacked bars, a theme heatmap showing which complaints dominate at which location, per-gym summary cards, and a downloadable results CSV. All charts are interactive Plotly visualizations.

---

**Q27. What's the CSV auto-detection feature?**

Users can either upload a CSV file or paste CSV text directly into a text area. The app automatically detects if the pasted text is CSV format by checking if the first line contains commas and the word "text." If it detects CSV, it parses it with pandas including any location and behavioral columns. If not, it treats each line as a plain text review. This removes friction for quick testing — you don't need to create a file just to try 10 reviews.

---

## 🔍 TOUGH QUESTIONS

---

**Q28. What are the honest limitations of this system?**

Three main ones. First, churn labels are keyword-based with about 20% noise — a review saying "I'd never cancel, it's perfect" would be a false positive. In production you'd want real cancellation records from a CRM. Second, the behavioral data in our demo is simulated — the rules are sound but they haven't been validated against actual gym cancellation outcomes. Third, the 50/50 blend weight between text and behavioral signals is a heuristic, not optimized. With real data you'd cross-validate to find the ideal weighting.

---

**Q29. How would you improve this for production deployment?**

Four concrete steps. First, integrate with a real gym CRM like Mindbody or ClubReady to get actual visit logs and cancellation records as ground truth labels. Second, retrain BERT on those real labels instead of keyword proxies. Third, replace the rule-based behavioral scoring with a trained XGBoost model on actual cancellation data — it would learn nonlinear patterns that simple rules miss. Fourth, build an A/B testing framework to measure whether acting on ChurnLens predictions actually reduces churn rates. Close the feedback loop.

---

**Q30. Why didn't you use GPT-4 or a large language model for this?**

Cost and latency. GPT-4 charges about 3 cents per thousand tokens. With 391K reviews averaging 150 tokens each, that's roughly 1,800 dollars just for one pass through the data — and that's before any inference costs in production. BERT runs locally, costs nothing after training, processes a review in under 200 milliseconds on CPU, and can be fully self-hosted. For a focused classification task with domain-specific training data, a fine-tuned smaller model like BERT consistently matches or beats larger models. You don't need a 175-billion parameter model to classify gym reviews.

---

> **Pro tip:** Don't memorize these word-for-word. Read each answer, understand the *why*, and then explain it in your own words. Interviewers can tell when someone is reciting vs. genuinely understanding.
