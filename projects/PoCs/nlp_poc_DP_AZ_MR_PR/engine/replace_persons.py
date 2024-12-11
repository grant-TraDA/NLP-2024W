import spacy

nlp = spacy.load("en_core_web_lg")

PLACEHOLDER = "John"


def replace_ner(text: str) -> str:
    new_text = text

    doc = nlp(text)
    names_ent = filter(lambda ent: ent.label_ == "PERSON", doc.ents)
    names = map(lambda ent: ent.text, names_ent)
    for name in names:
        new_text = new_text.replace(name, PLACEHOLDER)

    return new_text
