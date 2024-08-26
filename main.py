from fastapi import FastAPI
import torch
from transformers import BertTokenizer, BertForMaskedLM
from pydantic import BaseModel

app = FastAPI()

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

class SentenceInput(BaseModel):
    text: str

@app.post("/complete_sentence")
async def complete_sentence(input_data: SentenceInput):
    input_text = input_data.text
    
    # Tokenize input
    input_tokens = tokenizer.tokenize(input_text)
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    
    # Add CLS and SEP tokens
    input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]
    
    # Add mask token
    mask_token = tokenizer.mask_token_id
    input_ids.insert(2, mask_token)  # Insert mask after the first word
    
    # Convert to tensor
    input_ids = torch.tensor([input_ids])
    
    # Predict the masked word
    with torch.no_grad():
        outputs = model(input_ids)
        predictions = outputs.logits
    
    # Get the predicted token
    predicted_index = torch.argmax(predictions[0, 2]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])
    
    # Replace the mask with the predicted word
    input_tokens.insert(2, predicted_token[0])
    output_text = tokenizer.convert_tokens_to_string(input_tokens)
    
    return {"completed_sentence": output_text}
