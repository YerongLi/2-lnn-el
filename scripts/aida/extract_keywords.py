import pytextrank
import spacy
# import mongodb

# example text
text = "Port conditions update - Syria - Lloyds Shipping . Port conditions from Lloyds Shipping Intelligence Service -- LATTAKIA , Aug 10 - waiting time at Lattakia and Tartous presently 24 hours ."

# load a spaCy model, depending on language, scale, etc.
nlp = spacy.load("en_core_web_lg")

# add PyTextRank to the spaCy pipeline
nlp.add_pipe("textrank")
doc = nlp(text)

# examine the top-ranked phrases in the document
for phrase in doc._.phrases:
    print(phrase.text)
    print(phrase.rank, phrase.count)
    print(phrase.chunks)

import yake
kw_extractor = yake.KeywordExtractor()
keywords = kw_extractor.extract_keywords(text)
print(keywords)