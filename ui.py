import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from tg_parser import TelegramParser
import seaborn as sns
from geotracker import GeoAnalyzer
import folium
from streamlit_folium import st_folium


SEARCH_ROOTS = []

st.set_page_config(page_title="Telegram City Detector", layout="wide")

st.title("Telegram data analyzer")
st.write("Анализ сообщений и извлечение локаций")

mode = st.radio(
    "Источник данных",
    ["Обработать raw JSON", "Загрузить уже обработанный CSV"]
)

if "df" not in st.session_state:
    st.session_state.df = None

if "filtered_df" not in st.session_state:
    st.session_state.filtered_df = None

if "places" not in st.session_state:
    st.session_state.places = None

if "stage" not in st.session_state:
    st.session_state.stage = "raw"   # raw → loaded → lemmatized → filtered

if mode == "Обработать raw JSON":

    uploaded_file = st.file_uploader("Загрузите result.json", type=["json"])

    if uploaded_file:

        with open("temp.json", "wb") as f:
            f.write(uploaded_file.getbuffer())

        parser = TelegramParser("temp.json")

        with st.spinner("Загрузка сообщений..."):
            st.session_state.df = parser.load_data()

        st.session_state.stage = "loaded"

        st.success(f"Сообщений загружено: {len(st.session_state.df)}")


if mode == "Загрузить уже обработанный CSV":

    uploaded_file = st.file_uploader("Загрузите parsed_df.csv", type=["csv"])

    if uploaded_file:

        df = pd.read_csv(uploaded_file)
        df["date"] = pd.to_datetime(df["date"])

        st.session_state.df = df

        st.session_state.stage = "lemmatized"

        st.success(f"CSV загружен. Сообщений: {len(df)}")


if st.session_state.df is not None:

    st.subheader("Тепловая карта активности")
    df_heat = st.session_state.df.copy()
    df_heat['hour'] = df_heat['date'].dt.hour
    df_heat['weekday'] = df_heat['date'].dt.day_name()

    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    heat_data = df_heat.groupby(['weekday', 'hour']).size().unstack(fill_value=0)
    heat_data = heat_data.reindex(days_order)

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(heat_data, cmap="YlGnBu", ax=ax)
    st.pyplot(fig)

    st.subheader("Ключевые слова")
    SEARCH_ROOTS = st.text_input(
        "Введите корни ключевых слов через запятую",
        "каф,в ,во ,на ,из ,у ,к ,с ,рест,бар,пойт,идт,хоч,зайт,кофей",
        key="2"
    ).split(",")
    keywords_input = "кафе, в , ресторан, бар, пойти, идти, хочу, зайти, кофейня"



if st.session_state.df is not None and st.session_state.stage == "loaded":


    if st.button("Запустить лемматизацию"):

        parser = TelegramParser("dummy")
        parser.df = st.session_state.df

        progress = st.progress(0)
        status = st.empty()

        with st.spinner("Лемматизация..."):

            st.session_state.df = parser.lemmatize_parallel(
                batch_size=500,
                progress_bar=progress,
                status=status,
                roots=SEARCH_ROOTS
            )

        #st.session_state.stage = "lemmatized"

        st.success("Данные лемматизированы.")
        st.status("Данные фильтруются")

        keywords = [k.strip() for k in keywords_input.split(",")]
        with st.spinner("Фильтрация..."):
            parser = TelegramParser("dummy")
            parser.df = st.session_state.df

            filtered_df = parser.filter_by_lemmas(keywords)

            st.session_state.filtered_df = filtered_df
            st.session_state.places = parser.extract_entities_ner(filtered_df)

        st.session_state.stage = "filtered"


if st.session_state.stage in ["lemmatized", "filtered"]:

    #st.subheader("Ключевые слова")

    #keywords_input = st.text_input(
    #    "Введите ключевые слова через запятую",
    #    "кафе, ресторан, бар, пойти, идти, хочу, зайти, кофейня, я в",
    #    key="1"
    #)

    keywords = [k.strip() for k in keywords_input.split(",")]

    if st.button("Фильтровать сообщения"):
        keywords = [k.strip() for k in keywords_input.split(",")]
        with st.spinner("Фильтрация..."):
            parser = TelegramParser("dummy")
            parser.df = st.session_state.df

            filtered_df = parser.filter_by_lemmas(keywords)

            st.session_state.filtered_df = filtered_df
            st.session_state.places = parser.extract_entities_ner(filtered_df)

        st.session_state.stage = "filtered"


if st.session_state.filtered_df is not None:

    filtered_df = st.session_state.filtered_df

    st.write(f"Сообщений после фильтрации: {len(filtered_df)}")

    max_rows = st.slider("Сколько сообщений показать", 10, 500, 100)

    text_output = "\n".join(
        f"[{row['date']}] ({row['chat']}) {row['text'][:120]}"
        for _, row in filtered_df.head(max_rows).iterrows()
    )

    st.text(text_output)


if st.session_state.places:

    st.subheader("Найденные места")

    places_text = "\n".join(
        f"{place} — {count}"
        for place, count in list(st.session_state.places.items())[:100]
    )

    st.text(places_text)

if "df_coords" not in st.session_state:
    st.session_state.df_coords = None
if "geo_prediction" not in st.session_state:
    st.session_state.geo_prediction = None

if st.session_state.get("places"):
    st.divider()
    st.subheader(" Прогноз местоположения")

    if st.button("Запустить гео-анализ"):
        analyzer = GeoAnalyzer()

        with st.spinner("Связываемся с картами..."):
            coords_res = analyzer.get_coords_list(st.session_state.places)
            st.session_state.df_coords = coords_res

            if not coords_res.empty:
                city, country = analyzer.predict_location(coords_res)
                st.session_state.geo_prediction = f"{city}, {country}"
            else:
                st.session_state.geo_prediction = "Не удалось определить"

if st.session_state.df_coords is not None:
    df_coords = st.session_state.df_coords

    st.subheader("📊 Лог найденных локаций")
    cols_to_show = ['name', 'weight']
    if 'address' in df_coords.columns:
        cols_to_show.append('address')

    st.dataframe(df_coords[cols_to_show].sort_values(by='weight', ascending=False), use_container_width=True)

    st.subheader("🗺 Интерактивная карта")

    m = folium.Map(location=[df_coords.lat.mean(), df_coords.lon.mean()], zoom_start=6)

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen']

    for _, row in df_coords.iterrows():
        cluster_id = int(row['cluster'])

        point_color = colors[cluster_id % len(colors)] if cluster_id != -1 else 'black'

        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=5 + (min(row['weight'], 20) * 0.5),
            color=point_color,
            fill=True,
            fill_color=point_color,
            popup=f"Кластер: {cluster_id} | {row['name']}. Упоминаний - {row['weight']}"
        ).add_to(m)

    st_folium(m, width=1000, height=500, key="geo_map")

    if st.session_state.geo_prediction:
        st.success(f"📍 **Итоговый прогноз:** {st.session_state.geo_prediction}")

if st.session_state.filtered_df is not None:

    csv = st.session_state.filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Скачать CSV",
        csv,
        "filtered_messages.csv",
        "text/csv"
    )