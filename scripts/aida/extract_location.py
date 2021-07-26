import spacy
from spacy import displacy 
nlp = spacy.load('en_core_web_lg')# Text with nlp
doc = nlp(" Multiple tornado warnings were issued for parts of New York on Sunday night.The first warning, which expired at 9 p.m., covered the Bronx, Yonkers and New Rochelle. More than 2 million people live in the impacted area.")# Display Entities
displacy.render(doc, style="ent")