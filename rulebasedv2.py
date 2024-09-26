import spacy

# Load the spaCy model for English
nlp = spacy.load("en_core_web_sm")

# Define a function to apply sentence generation rules with error handling
def generate_sentence(input_words):
    # Join the input words into a rough sentence
    rough_sentence = ' '.join(input_words)
    
    # Process the text using spaCy
    doc = nlp(rough_sentence)
    
    # Print the NLP tags (POS, dependency, entity) for each word
    print("Word details for debugging:\n")
    for token in doc:
        print(f"Word: {token.text}, POS: {token.pos_}, Dependency: {token.dep_}, Head: {token.head.text}")
    
    # Initialize components of the sentence
    subject = ""
    verb = ""
    obj = ""
    adjective = ""
    aux_verb = ""  # Auxiliary verb for the subject
    named_entities = []
    
    # Flags to check for missing parts of the sentence
    has_noun = False
    has_verb = False

    # Rule 1: Identify subject, verb, object, and adjectives
    for token in doc:
        if token.dep_ == 'nsubj':  # Subject
            subject = token.text
            has_noun = True  # Mark noun as found
        elif token.pos_ == 'VERB':  # Main verb
            verb = token.text
            has_verb = True  # Mark verb as found
        elif token.dep_ == 'dobj':  # Direct object
            obj = token.text
            has_noun = True  # Mark noun as found
        elif token.dep_ == 'advmod' or token.dep_ == 'acomp' or token.dep_ == 'amod':  # Adjective
            adjective = token.text
        elif token.ent_type_:  # Named entities like proper nouns
            named_entities.append(token.text)
            has_noun = True  # Named entities are often proper nouns

    # Check for missing nouns or verbs and handle errors
    if not has_noun:
        return "Error: The sentence is missing a noun."
    if not has_verb:
        return "Error: The sentence is missing a verb."
    
    # Rule 2: Assign auxiliary verb based on the subject
    if subject.lower() == "i":
        aux_verb = "am"
    elif subject.lower() in ["he", "she", "it"]:
        aux_verb = "is"
    elif subject.lower() in ["you", "we", "they"]:
        aux_verb = "are"
    
    # Rule 3: Construct the sentence
    sentence = ""
    
    # If there's an adjective, construct a sentence with auxiliary verb
    if adjective and aux_verb:
        sentence = f"{subject.capitalize()} {aux_verb} {adjective}"
    
    # Add verb and object to complete the sentence
    if verb:
        if sentence:
            sentence += f" want to {verb}"
        else:
            sentence = f"{subject.capitalize()} {verb}"
        if obj:
            sentence += f" {obj}"

    # Ensure sentence ends with a period and capitalize it
    sentence = sentence.strip()
    if sentence and not sentence.endswith('.'):
        sentence += '.'

    return sentence

# Example 1: Input words
input_words1 = ["I", "Oreo", "eat", "hungry"]
# Example 2: Input words missing a verb
input_words2 = ["He", "tired", "coffee"]
# Example 3: Input words missing a noun
input_words3 = ["quickly", "run"]

# Generate sentences
output_sentence1 = generate_sentence(input_words1)
output_sentence2 = generate_sentence(input_words2)
output_sentence3 = generate_sentence(input_words3)

print("\nGenerated Sentence 1: ", output_sentence1)
print("Generated Sentence 2: ", output_sentence2)
print("Generated Sentence 3: ", output_sentence3)
