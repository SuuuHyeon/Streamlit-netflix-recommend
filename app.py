import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from googletrans import Translator

translator = Translator()

@st.cache_data
def load_data():
    df = pd.read_csv("netflix_titles.csv")
    df = df[df['type'] == 'Movie'].dropna(subset=['title', 'description'])
    return df.reset_index(drop=True)

@st.cache_data
def compute_similarity(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['description'])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend(title, df, similarity_matrix, n=5):
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    idx = indices.get(title)
    if idx is None:
        return []
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    rec_indices = [i[0] for i in sim_scores]
    return df.iloc[rec_indices]

# ---------------------- UI 시작 ----------------------

st.title("🎬 넷플릭스 영화 추천기")
st.write("선택한 영화의 설명과 비슷한 넷플릭스 영화를 추천해드려요!")

df = load_data()
similarity_matrix = compute_similarity(df)

user_input = st.text_input("🎞️ 영화 제목을 입력하세요 (예: Inception)", "")

if user_input:
    matched_titles = df['title'][df['title'].str.contains(user_input, case=False, na=False)]
    if matched_titles.empty:
        st.warning("입력한 제목과 일치하는 영화를 찾을 수 없습니다.")
    else:
        selected_title = matched_titles.values[0]
        st.subheader("🔎 추천 결과")
        results = recommend(selected_title, df, similarity_matrix)
        for _, row in results.iterrows():
            translated_desc = translator.translate(row['description'], dest='ko').text
            translated_genre = translator.translate(row['listed_in'], dest='ko').text
            st.markdown(f"**🎬 {row['title']}**")
            st.caption(f"장르: {translated_genre}")
            st.write(translated_desc)
            st.markdown("---")
