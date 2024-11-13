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
    noun = None

    # Iterate over POS tags and classify them based on their tags
    for tag in pos_tags:
        if tag["pos"] == "PRON" and pronoun is None:
            pronoun = tag["word"]
        elif tag["pos"] == "ADJ" and adjective is None:
            adjective = tag["word"]
        elif tag["pos"] == "VERB" and verb is None:
            verb = tag["word"]
        elif tag["pos"] == "NOUN" and noun is None:
            noun = tag["word"]

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

def generate_sentence(input_words):
    pos_tags = get_independent_pos_tags(input_words)

    # Arrange words in "Pronoun + Adjective + Verb + Noun" order
    ordered_sentence = arrange_words_by_order(pos_tags)

    # Process POS and dependencies to form the sentence based on provided logic
    subject = ""
    verb = ""
    obj = ""
    adjective = ""
    aux_verb = ""
    named_entities = []

    has_noun = False
    has_verb = False

    # Extract relevant components based on POS tags
    for tag in pos_tags:
        if tag["pos"] == 'PRON':
            subject = tag["word"]
            has_noun = True
        elif tag["pos"] == 'VERB':
            verb = tag["word"]
            has_verb = True
        elif tag["pos"] == 'NOUN':
            obj = tag["word"]
            has_noun = True
        elif tag["pos"] == 'ADJ':
            adjective = tag["word"]

    # Determine auxiliary verb based on the subject
    if subject.lower() == "i":
        aux_verb = "am"
    elif subject.lower() in ["he", "she", "it"]:
        aux_verb = "is"
    elif subject.lower() in ["you", "we", "they"]:
        aux_verb = "are"

    # Final sentence formation, combining ordered words and other logic
    sentence = ordered_sentence  # Start with the ordered sentence
    
    if adjective and aux_verb and subject:
        sentence = f"{subject.capitalize()} {aux_verb} {adjective}"
    elif subject and not adjective and verb and obj:
        sentence = f"{subject.capitalize()} want to {verb} {obj}"
    elif subject and not adjective and obj:
        article = "an" if obj[0].lower() in "aeiou" else "a"
        if len([tag for tag in pos_tags if tag["pos"] == 'NOUN']) > 1:
            objects = [tag["word"] for tag in pos_tags if tag["pos"] == 'NOUN']
            sentence = f"{subject.capitalize()} want {obj[0]} and {obj[1]}"
        sentence = f"{subject.capitalize()} want {article} {obj}"
    elif subject and not adjective and verb:
        sentence = f"{subject.capitalize()} want to {verb}"
    

    if adjective and aux_verb and subject:
        if verb and obj:
            sentence += f", {subject.capitalize()} want to {verb} {obj}"
        elif obj:
            article = "an" if obj[0].lower() in "aeiou" else "a"
            sentence += f", {subject.capitalize()} want {article} {obj}"
        elif verb:
            sentence += f", {subject.capitalize()} want to {verb}"

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
