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
    words: list[str]

class TextOutput(BaseModel):
    sentence: str
    pos_tags: list
    ner_tags: list
    dependency: list

# Dictionary of words with their correct POS tags
force_pos_tags = {
    "bored": "ADJ",
    "cake": "NOUN",
    "eat": "VERB",
    "keyboard": "NOUN",
    "ballpen": "NOUN",
    "study": "VERB",
    "shocked": "ADJ",
}

def get_independent_pos_tags(words):
    tags = []
    for word in words:
        # Process each word independently
        doc = nlp(word)
        if doc[0].text in force_pos_tags:
            pos = force_pos_tags[doc[0].text]  # Override with custom POS tag if available
        else:
            pos = doc[0].pos_
        tags.append({"word": word, "pos": pos})
    return tags

def arrange_words_by_order(pos_tags):
    pronoun = None
    adjective = None
    verb = None
    nouns = []

    # Iterate over POS tags and classify them based on their tags
    for tag in pos_tags:
        if tag["pos"] == "PRON" and pronoun is None:
            pronoun = tag["word"]
        elif tag["pos"] == "ADJ" and adjective is None:
            adjective = tag["word"]
        elif tag["pos"] == "VERB" and verb is None:
            verb = tag["word"]
        elif tag["pos"] == "NOUN":
            nouns.append(tag["word"])

    # Create a list of the words in the correct order
    ordered_sentence = []
    if pronoun:
        ordered_sentence.append(pronoun)
    if adjective:
        ordered_sentence.append(adjective)
    if verb:
        ordered_sentence.append(verb)
    ordered_sentence.extend(nouns)

    return ordered_sentence

def generate_sentence(input_words):
    pos_tags = get_independent_pos_tags(input_words)
    ordered_words = arrange_words_by_order(pos_tags)

    subject = ""
    verb = ""
    adjective = ""
    aux_verb = ""
    nouns = []

    # Extract relevant components based on POS tags
    for tag in pos_tags:
        if tag["pos"] == 'PRON':
            subject = tag["word"]
        elif tag["pos"] == 'VERB':
            verb = tag["word"]
        elif tag["pos"] == 'NOUN' or tag["pos"] == 'PROPN':
            nouns.append(tag["word"])
        elif tag["pos"] == 'ADJ':
            adjective = tag["word"]

    # Determine auxiliary verb based on the subject
    if subject.lower() == "i":
        aux_verb = "am"
    elif subject.lower() in ["he", "she", "it"]:
        aux_verb = "is"
    elif subject.lower() in ["you", "we", "they"]:
        aux_verb = "are"

    # Sentence formation with multiple nouns
    sentence = ""
    
    if adjective and aux_verb and subject:
        # Handle case with adjective
        sentence = f"{subject.capitalize()} {aux_verb} {adjective}"
        
        if nouns:
            # Add objects after adjective statement
            if len(nouns) == 1:
                article = "an" if nouns[0][0].lower() in "aeiou" else "a"
                sentence += f", {subject.lower()} want {article} {nouns[0]}"
            else:
                # Join multiple nouns with "and"
                sentence += f", {subject.lower()} want {' and '.join(nouns)}"
    else:
        # Handle case without adjective
        if subject:
            sentence = f"{subject.capitalize()} want"
            if len(nouns) == 1:
                article = "an" if nouns[0][0].lower() in "aeiou" else "a"
                sentence += f" {article} {nouns[0]}"
            elif len(nouns) > 1:
                sentence += f" {' and '.join(nouns)}"
            
            if verb:
                sentence = f"{subject.capitalize()} want to {verb} {' and '.join(nouns)}"

    # Ensure proper sentence ending
    sentence = sentence.strip()
    if sentence and not sentence.endswith('.'):
        sentence += '.'

    return sentence

@app.post("/complete_sentence", response_model=TextOutput)
def generate_sentence_endpoint(text_input: TextInput):
    input_words = text_input.words
    generated_sentence = generate_sentence(input_words)
    
    pos_tags = get_independent_pos_tags(input_words)
    ner_tags = [{"word": ent.text, "label": ent.label_} for ent in nlp(' '.join(input_words)).ents]
    dependencies = [{"word": tag["word"], "pos": tag["pos"]} for tag in pos_tags]

    return {
        "sentence": generated_sentence,
        "pos_tags": pos_tags,
        "ner_tags": ner_tags,
        "dependency": dependencies
    }

# if __name__ == "__main__":
#    uvicorn.run("main:app", host="192.168.1.19", port=5724)