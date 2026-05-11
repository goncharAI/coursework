from natasha import (
    Segmenter, MorphVocab, NewsEmbedding,
    NewsMorphTagger, NewsNERTagger, Doc
)

nlp = {}


def init_worker():
    emb = NewsEmbedding()
    nlp['segmenter'] = Segmenter()
    nlp['morph_vocab'] = MorphVocab()
    nlp['morph_tagger'] = NewsMorphTagger(emb)
    nlp['ner_tagger'] = NewsNERTagger(emb)


def lemmatize_batch(texts):
    results = []
    for text in texts:
        if not text:
            results.append("")
            continue
        doc = Doc(text.lower())
        doc.segment(nlp['segmenter'])
        doc.tag_morph(nlp['morph_tagger'])

        lemmas = []
        for token in doc.tokens:
            token.lemmatize(nlp['morph_vocab'])
            lemmas.append(token.lemma)
        results.append(" ".join(lemmas))
    return results


def ner_batch(texts):
    batch_locations = []
    for text in texts:
        doc = Doc(text)
        doc.segment(nlp['segmenter'])
        doc.tag_ner(nlp['ner_tagger'])
        for span in doc.spans:
            if span.type == 'LOC':
                span.normalize(nlp['morph_vocab'])
                loc = span.normal.title().strip()
                if len(loc) > 2:
                    batch_locations.append(loc)
    return batch_locations
