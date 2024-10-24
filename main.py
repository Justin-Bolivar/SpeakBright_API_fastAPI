import subprocess
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import spacy
import uvicorn
import verbnet  # Added for using VerbNet

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

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

def find_suitable_verb_verbnet(noun):
    # Access verb classes related to the noun
    verb_classes = verbnet.classids(lemma=noun)
    if verb_classes:
        # Get the verbs associated with those classes
        verbs = []
        for v_class in verb_classes:
            verbs.extend(verbnet.members(v_class))
        return verbs[0] if verbs else None
    return None

def generate_sentence(input_words):
    rough_sentence = ' '.join(input_words)
    doc = nlp(rough_sentence)

    # Initialize variables
    subject = ""
    verb = ""
    obj = ""
    adjective = ""
    aux_verb = ""
    named_entities = []
    has_noun = False
    has_verb = False

    # Extract details using POS and dependency parsing
    for token in doc:
        if token.pos_ == 'PRON':
            subject = token.text
            has_noun = True
        elif token.pos_ == 'VERB':
            verb = token.text
            has_verb = True
        elif token.dep_ in ['dobj', 'attr', 'nsubj'] and token.pos_ in ['NOUN', 'PROPN']:
            obj = token.text
            has_noun = True
        elif token.dep_ in ['acomp', 'amod']:  # Adjective
            adjective = token.text
        elif token.ent_type_:
            named_entities.append(token.text)
            has_noun = True

    # Attempt to find a suitable verb if not already present
    if not has_verb and has_noun:
        if adjective:
            verb = find_suitable_verb_verbnet(adjective)
        if not verb and obj:
            verb = find_suitable_verb_verbnet(obj)

    # Auxiliary verb determination
    if subject.lower() == "i":
        aux_verb = "am"
    elif subject.lower() in ["he", "she", "it"]:
        aux_verb = "is"
    elif subject.lower() in ["you", "we", "they"]:
        aux_verb = "are"

    # Constructing the sentence
    sentence = f"{subject.capitalize()} {aux_verb} {adjective}".strip()

    # Add verbs and objects if needed
    if verb and obj:
        sentence += f", {subject.capitalize()} want to {verb} {obj}."

    # Sentence finalization
    if not sentence.endswith('.'):
        sentence += '.'

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

# if __name__ == "__main__":
#     uvicorn.run("main:app", host="192.168.1.21", port=5724)
