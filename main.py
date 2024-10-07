import subprocess
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import spacy
import uvicorn

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

def generate_sentence(input_words):
    rough_sentence = ' '.join(input_words)
    
    doc = nlp(rough_sentence)
    
    subject = ""
    verb = ""
    obj = ""
    adjective = ""
    aux_verb = ""
    named_entities = []
    
    has_noun = False
    has_verb = False

    for token in doc:
        if token.dep_ == 'nsubj':
            subject = token.text
            has_noun = True
        elif token.pos_ == 'VERB':
            verb = token.text
            has_verb = True
        elif token.dep_ == 'dobj':
            obj = token.text
            has_noun = True 
        elif token.dep_ == 'advmod' or token.dep_ == 'acomp' or token.dep_ == 'amod':  # Adjective
            adjective = token.text
        elif token.ent_type_:
            named_entities.append(token.text)
            has_noun = True

    if not has_noun:
        raise HTTPException(status_code=400, detail="Error: The sentence is missing a noun.")
    if not has_verb:
        raise HTTPException(status_code=400, detail="Error: The sentence is missing a verb.")
    
    if subject.lower() == "i":
        aux_verb = "am"
    elif subject.lower() in ["he", "she", "it"]:
        aux_verb = "is"
    elif subject.lower() in ["you", "we", "they"]:
        aux_verb = "are"
    
    sentence = ""
    
    if adjective and aux_verb:
        sentence = f"{subject.capitalize()} {aux_verb} {adjective}"
    
    if verb:
        if sentence:
            sentence += f", {subject.capitalize()} want to {verb}"
        else:
            sentence = f"{subject.capitalize()} {verb}"
        if obj:
            sentence += f" {obj}"

    sentence = sentence.strip()
    if sentence and not sentence.endswith('.'):
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

#if __name__ == "__main__":
   #uvicorn.run("main:app", host="192.168.1.21", port=5724)