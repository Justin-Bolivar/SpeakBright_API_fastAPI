import nltk
from nltk.util import ngrams
from collections import Counter
from datasets import load_dataset
from nltk.stem import WordNetLemmatizer

nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')

# Load WikiText-2 dataset from Hugging Face
dataset = load_dataset('wikitext', 'wikitext-2-v1', split='train')

# Extract text from the dataset
text_data = dataset['text']

# Concatenate all sentences to form a long text
all_text = ' '.join(text_data)

# Tokenize the text using NLTK
tokens = nltk.word_tokenize(all_text.lower())

# Generate bigrams
bigrams = list(ngrams(tokens, 2))

# Count the frequency of each bigram
bigram_freq = Counter(bigrams)

# Generate trigrams
trigrams = list(ngrams(tokens, 3))

# Count the frequency of each trigram
trigram_freq = Counter(trigrams)

def predict_word_between(word1, word2, trigram_freq, bigram_freq, default_word="am"):
    # Predict from trigram
    candidates = [trigram for trigram in trigram_freq if trigram[0] == word1 and trigram[2] == word2]
    
    if candidates:
        best_trigram = max(candidates, key=lambda trigram: trigram_freq[trigram])
        return best_trigram[1] 
    
    # if not found go to bigram
    candidates = [bigram for bigram in bigram_freq if bigram[0] == word1]
    
    if candidates:
        best_bigram = max(candidates, key=lambda bigram: bigram_freq[bigram])
        return best_bigram[1]
    
    # if no match use default_word which is "am"
    return default_word

def predict_full_sentence(words, trigram_freq, bigram_freq):
    result = [words[0]]  # Start with the first word
    
    for i in range(len(words) - 1):
        word1 = words[i]
        word2 = words[i + 1]
        
        # Try predicting a word between the two words
        predicted_word = predict_word_between(word1, word2, trigram_freq, bigram_freq)
        
        # Append the predicted word (if any) and the next word
        if predicted_word:
            result.append(predicted_word)
        result.append(word2)
    
    return ' '.join(result)


    import nltk
from nltk import pos_tag

# Manual corrections for known words
manual_corrections = {
    'eat': 'VB',
    'draw': 'VB',
    'run': 'VB',
    'swim': 'VB',  # Ensure case sensitivity
    'study': 'VB',
}

def display_pos_tags(input_words):
    # Perform POS tagging
    pos_tagged = pos_tag(input_words)
    print("POS Tags:", pos_tagged)
    return pos_tagged

def reorder_words(input_words):
    # Perform POS tagging
    pos_tagged = pos_tag(input_words)
    
    # Apply manual corrections to the POS tags
    corrected_pos_tagged = [(word, manual_corrections.get(word.lower(), tag)) for word, tag in pos_tagged]
    
    # Separate the words by their POS tags (simplified structure)
    pronoun = []
    subject = []
    adjective = []
    verbs = []
    objects = []
    others = []
    
    # Flags to check for verb and noun
    has_verb = False
    has_noun = False
    
    # Classify the words based on their POS tags
    for word, tag in corrected_pos_tagged:
        if tag.startswith('PRP'):  # Pronouns
            pronoun.append(word)
        elif tag.startswith('NN'):  # Subject nouns
            subject.append(word)
            has_noun = True
        elif tag.startswith('JJ'):  # Adjectives
            adjective.append(word)
        elif tag.startswith('VB'):  # Verbs
            verbs.append(word)
            has_verb = True
        elif tag.startswith('NN') or tag.startswith('NNS'):  # Object nouns
            objects.append(word)
            has_noun = True
        else:
            others.append(word)
    
    # Check if both verb and noun are present
    if not (has_verb and has_noun):
        missing_parts = []
        if not has_verb:
            missing_parts.append("verb")
        if not has_noun:
            missing_parts.append("noun")
        raise ValueError(f"Error: A sentence needs a {' and '.join(missing_parts)}.")
    
    # Reorder based on a simple sentence structure: Pronoun + Adjective + Verb + Subject + Objects + Others
    reordered_sentence = pronoun + adjective + verbs + subject + objects + others
    
    return ' '.join(reordered_sentence)

# Example input words
input_words = ["happy", "pool", "I", "swim"]

# Call the function
reordered_sentence = reorder_words(input_words)
print(f"Original: {input_words}\nReordered: {reordered_sentence}")
pos_tags = display_pos_tags(input_words)

reordered = reordered_sentence.split()
print(reordered)

# Predict full sentence by filling in the words between
sentence = predict_full_sentence(reordered, trigram_freq, bigram_freq)
print("Predicted Sentence:", sentence)