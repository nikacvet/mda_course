from keybert import KeyBERT
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
# from sklearn.cluster import KMeans
# from sentence_transformers import SentenceTransformer
# from sklearn.feature_extraction.text import CountVectorizer
# from collections import Counter

# def cluster_objectives_with_bert(projects_preprocessed: pd.DataFrame) -> None:

#     text_data = projects_preprocessed["objective"].dropna().astype(str).tolist()
#     if len(text_data) < 5:
#         print("Not enough objectives to cluster meaningfully.")
#         return

#     model = SentenceTransformer("all-MiniLM-L6-v2")
#     embeddings = model.encode(text_data, show_progress_bar=True)

#     # Determine number of clusters using heuristic
#     k = min(8, max(2, len(text_data) // 25))
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     labels = kmeans.fit_predict(embeddings)

#     # Assign cluster labels to original texts
#     clustered_texts = pd.DataFrame({"text": text_data, "cluster": labels})

#     # Extract top words per cluster
#     vectorizer = CountVectorizer(stop_words="english")
#     counts = vectorizer.fit_transform(clustered_texts["text"])
#     terms = vectorizer.get_feature_names_out()

#     wordcloud_data = []
#     for cluster_id in sorted(clustered_texts["cluster"].unique()):
#         row_idx = clustered_texts[clustered_texts["cluster"] == cluster_id].index
#         cluster_counts = counts[row_idx]
#         word_freq = cluster_counts.sum(axis=0).A1
#         top_words = [terms[i] for i in word_freq.argsort()[::-1][:15]]
#         for word in top_words:
#             wordcloud_data.append({"topic": f"topic_{cluster_id}", "word": word})

#     with open("objective_topics_wordcloud.json", "w") as f:
#         json.dump(wordcloud_data, f, indent=2)

#     print("Exported BERT-based word cloud topic data to objective_topics_wordcloud.json")



def extract_keyphrases(projects_preprocessed: pd.DataFrame) -> dict:
    unique_df = (
        projects_preprocessed.dropna(subset=["objective"])
        .drop_duplicates(subset=["projectID"])
        .sort_values("projectID")[["projectID", "objective"]]
        .astype(str)
    )
    objectives = unique_df["objective"].tolist()[0:20] #! THIS NEEDS TO BE DELETED!!!
    project_ids = unique_df["projectID"].tolist()[0:20]
    model = KeyBERT("all-MiniLM-L6-v2")
    climate_keywords = [
        "climate", "carbon", "emission", "neutral", "warming", "co2", "sustainab",
        "net zero", "decarbon", "biodiversity", "resilience"
    ]
    project_keyphrases = defaultdict(list)
    for pid, obj in tqdm(zip(project_ids, objectives), total=len(objectives), desc="Extracting filtered keyphrases"):
        phrases = model.extract_keywords(
            obj,
            keyphrase_ngram_range=(1, 3),
            stop_words='english',
            top_n=5,
            use_maxsum=True
        )
        for phrase, _ in phrases:
            phrase_lower = phrase.lower()
            if any(k in phrase_lower for k in climate_keywords):
                project_keyphrases[pid].append(phrase_lower)

                
    return dict(project_keyphrases)