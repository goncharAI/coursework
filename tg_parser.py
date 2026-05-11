import json
import re
from collections import Counter
from multiprocessing import Pool, cpu_count
import pandas as pd
from nlp_utils import init_worker, lemmatize_batch, ner_batch


class TelegramParser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_data(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        messages = []

        def extract_text(msg_raw):
            text_raw = msg_raw.get("text", "")
            if isinstance(text_raw, list):
                text_raw = "".join(t if isinstance(t, str) else t.get("text", "") for t in text_raw)
            return text_raw

        chats = data.get("chats", {}).get("list", []) if "chats" in data else [data]
        for chat in chats:
            chat_name = chat.get("name", chat.get("id", "Unknown"))
            for msg in chat.get("messages", []):
                t = extract_text(msg)
                if t:
                    messages.append({"date": msg.get("date"), "text": t, "chat": chat_name})

        self.df = pd.DataFrame(messages)
        self.df["date"] = pd.to_datetime(self.df["date"])
        return self.df

    def lemmatize_parallel(self, batch_size=500, progress_bar=None, status=None, roots=None):
        if roots is None: roots = ['каф', 'рест', 'бар', 'пойт', 'идт']

        pattern = '|'.join(roots)
        mask = self.df['text'].str.contains(pattern, case=False, na=False)
        to_process_df = self.df[mask].copy()

        unique_texts = to_process_df['text'].unique().tolist()
        total_unique = len(unique_texts)

        if status: status.text(f"Лемматизация {total_unique} уникальных фраз...")

        batches = [unique_texts[i:i + batch_size] for i in range(0, total_unique, batch_size)]

        results_map = {}
        with Pool(processes=cpu_count(), initializer=init_worker) as pool:
            for i, res in enumerate(pool.imap(lemmatize_batch, batches)):
                start_idx = i * batch_size
                for j, lem_text in enumerate(res):
                    results_map[unique_texts[start_idx + j]] = lem_text
                if progress_bar: progress_bar.progress((i + 1) / len(batches))

        self.df['lemmas'] = self.df['text'].map(results_map).fillna("")
        self.df.to_csv("lemmatized.csv", index=False)
        return self.df

    def filter_by_lemmas(self, keywords):
        keywords = [k.strip().lower() for k in keywords if k.strip()]
        if not keywords: return self.df
        pattern = '|'.join([re.escape(k) for k in keywords])
        mask = self.df["lemmas"].str.contains(pattern, case=False, na=False, regex=True)
        return self.df[mask]

    def extract_entities_ner(self, df, batch_size=200):
        texts = df["text"].tolist()
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

        all_locations = []
        with Pool(processes=cpu_count(), initializer=init_worker) as pool:
            for res in pool.imap(ner_batch, batches):
                all_locations.extend(res)

        return Counter(all_locations)
