import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from deep_translator import GoogleTranslator

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
        return pd.DataFrame()
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    rec_indices = [i[0] for i in sim_scores]
    return df.iloc[rec_indices]

# ---------------------- UI 시작 ----------------------

st.title("🎬 넷플릭스 영화 추천기 (한글 번역 지원)")
st.write("🎯 영화 제목을 입력하면, 설명이 비슷한 영화를 추천하고 한국어로 보여줍니다.")

df = load_data()
similarity_matrix = compute_similarity(df)

user_input = st.text_input("🎞️ 영화 제목을 영어로 입력하세요 (예: Inception)", "")

if user_input:
    matched_titles = df['title'][df['title'].str.contains(user_input, case=False, na=False)]
    if matched_titles.empty:
        st.warning("일치하는 영화가 없습니다. 다른 제목을 시도해보세요.")
    else:
        selected_title = matched_titles.values[0]
        st.success(f"'{selected_title}'와 비슷한 영화 추천 결과:")

        results = recommend(selected_title, df, similarity_matrix)

        for _, row in results.iterrows():
            desc = row['description']
            genre = row['listed_in']

            # 번역 수행
            translated_desc = GoogleTranslator(source='auto', target='ko').translate(desc)
            translated_genre = GoogleTranslator(source='auto', target='ko').translate(genre)

            st.markdown(f"**🎬 {row['title']}**")
            st.caption(f"장르: {translated_genre}")
            st.write(translated_desc)
            st.markdown("---")
