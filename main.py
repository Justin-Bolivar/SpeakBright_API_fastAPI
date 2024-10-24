import subprocess
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import spacy
import uvicorn
import nltk
from nltk.corpus import wordnet

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

nltk.download('wordnet')

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # If the model is not found, download it
    print("Downloading the 'en_core_web_sm' model...")
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

class TextInput(BaseModel):
    text: str

class TextOutput(BaseModel):
    sentence: str
    pos_tags: list
    ner_tags: list
    dependency: list

def arrange_words_by_order(doc):
    pronoun = None
    adjective = None
    verb = None
    noun = None

    # Iterate over tokens and classify them based on their POS tags
    for token in doc:
        if token.pos_ == "PRON" and pronoun is None:
            pronoun = token.text
        elif token.pos_ == "ADJ" and adjective is None:
            adjective = token.text
        elif token.pos_ == "VERB" and verb is None:
            verb = token.text
        elif token.pos_ == "NOUN" and noun is None:
            noun = token.text

    # Create a list of the words in the correct order
    ordered_sentence = []
    if pronoun:
        ordered_sentence.append(pronoun)
    if adjective:
        ordered_sentence.append(adjective)
    if verb:
        ordered_sentence.append(verb)
    if noun:
        ordered_sentence.append(noun)

    return ' '.join(ordered_sentence).strip()

def find_suitable_verb(noun):
    synsets = wordnet.synsets(noun, pos=wordnet.NOUN)
    if synsets:
        for synset in synsets:
            for lemma in synset.lemmas():
                related_forms = lemma.derivationally_related_forms()
                for form in related_forms:
                    if form.synset().pos() == 'v':  # Check if it's a verb
                        return form.name()
    return None

def generate_sentence(input_words):
    rough_sentence = ' '.join(input_words)
    doc = nlp(rough_sentence)

    # Extract tokens from the doc for further processing
    subject = ""
    verb = ""
    obj = ""
    adjective = ""
    aux_verb = ""
    named_entities = []

    has_noun = False
    has_verb = False

    # Process POS and dependencies as in the original logic
    for token in doc:
        if token.pos_ == 'PRON':
            subject = token.text
            has_noun = True
        elif token.pos_ == 'VERB':
            verb = token.text
            has_verb = True
        elif token.dep_ == 'dobj' or token.pos_ == 'NOUN' or token.pos_ == 'PROPN':
            obj = token.text
            has_noun = True 
        elif token.dep_ == 'advmod' or token.dep_ == 'acomp' or token.dep_ == 'amod':  # Adjective
            adjective = token.text
        elif token.ent_type_:
            named_entities.append(token.text)
            has_noun = True

    # If no verb is found, use WordNet to find a suitable verb for the noun
    if not has_verb and has_noun:
        verb = find_suitable_verb(obj if obj else subject)
        if verb:
            has_verb = True

    # Determine auxiliary verb based on the subject
    if subject.lower() == "i":
        aux_verb = "am"
    elif subject.lower() in ["he", "she", "it"]:
        aux_verb = "is"
    elif subject.lower() in ["you", "we", "they"]:
        aux_verb = "are"

    # Final sentence formation, combining ordered words and other logic
    sentence = ""

    # If the sentence has a subject and auxiliary verb with an adjective
    if adjective and aux_verb and subject:
        sentence = f"{subject.capitalize()} {aux_verb} {adjective}"
    
    # Add the verb and object
    if verb:
        if sentence:
            sentence += f", {subject.capitalize()} want to {verb}"
        else:
            sentence = f"{subject.capitalize()} {verb}"
        if obj:
            sentence += f" {obj}"

    # Ensure proper sentence ending
    sentence = sentence.strip()
    if sentence and not sentence.endswith('.'):
        sentence += '.'

    # Append additional sentence if object exists and verb is found
    if obj and verb:
        sentence += f" I want to {verb} {obj}."

    return sentence

@app.post("/complete_sentence", response_model=TextOutput)
def generate_sentence_endpoint(text_input: TextInput):
    input_words = text_input.text.split()
    generated_sentence = generate_sentence(input_words)
    doc = nlp(text_input.text)

    pos_tags = [{"word": token.text, "pos": token.pos_} for token in doc]
    ner_tags = [{"word": ent.text, "label": ent.label_} for ent in doc.ents]
    dependencies = [{"word": token.text, "dependency": token.dep_, "head": token.head.text} for token in doc]

    return {
        "sentence": generated_sentence,
        "pos_tags": pos_tags,
        "ner_tags": ner_tags,
        "dependency": dependencies
    }

#if __name__ == "__main__":
   #uvicorn.run("main:app", host="192.168.1.21", port=5724)