from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertForMaskedLM

app = FastAPI()

# Initialize the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

class InputText(BaseModel):
    text: str  # Input model for receiving the text

# Function for dynamic masking
def dynamic_masking(input_sentence):
    words = input_sentence.split()
    new_words = []
    
    for i, word in enumerate(words):
        new_words.append(word)
        if i < len(words) - 1:
            new_words.append('[MASK]')
    
    return " ".join(new_words)

# Function for filling in the masked words using the BERT model
def fill_masks(sentence):
    input_ids = tokenizer.encode(sentence, return_tensors='pt')
    mask_token_indices = torch.where(input_ids == tokenizer.mask_token_id)[1]
    
    with torch.no_grad():
        output = model(input_ids)
    
    logits = output.logits
    softmax = torch.nn.functional.softmax(logits, dim=-1)
    
    for mask_index in mask_token_indices:
        mask_word_logits = softmax[0, mask_index, :]
        top_token_id = torch.topk(mask_word_logits, 1, dim=0).indices[0].item()
        predicted_token = tokenizer.convert_ids_to_tokens([top_token_id])[0]
        if predicted_token.startswith("##"):
            sentence = sentence.replace('[MASK]', predicted_token[2:], 1)
        else:
            sentence = sentence.replace('[MASK]', predicted_token, 1)
    
    return sentence

# Function to create a coherent sentence using BERT
def create_sentence_with_bert(input_sentence):
    masked_sentence = dynamic_masking(input_sentence)
    filled_sentence = fill_masks(masked_sentence)
    sentence = filled_sentence.replace('[CLS]', '').replace('[SEP]', '').strip()
    sentence = sentence.replace(' ,', ',').replace(' .', '.').replace(' !', '!').replace(' ?', '?')
    final_sentence = sentence.capitalize()
    return final_sentence

# API endpoint to complete the sentence
@app.post("/complete_sentence")
async def create_sentence(input_text: InputText):
    try:
        sentences = input_text.text.split('.')
        output_sentences = []
        for sentence in sentences:
            if sentence.strip():
                output_sentence = create_sentence_with_bert(sentence.strip())
                output_sentences.append(output_sentence)
        completed_sentence = '. '.join(output_sentences).strip()
        return {"completed_sentence": completed_sentence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
