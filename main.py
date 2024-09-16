from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from utils import predict_full_sentence, reorder_words, load_dataset_and_generate_ngrams
import uvicorn

app = FastAPI()

class TextRequest(BaseModel):
    text: str

# Load bigram and trigram frequencies when the app starts
bigram_freq, trigram_freq = load_dataset_and_generate_ngrams()

@app.post("/complete_sentence/")
def generate_sentence(request: TextRequest):
    # Tokenize the input text
    input_words = request.text.split()
    
    try:
        # Reorder words using POS tagging
        reordered_sentence = reorder_words(input_words)
        reordered = reordered_sentence.split()
        
        # Predict full sentence by filling in the words between
        predicted_sentence = predict_full_sentence(reordered, trigram_freq, bigram_freq)
        
        return {"original": request.text, "reordered": reordered_sentence, "predicted_sentence": predicted_sentence}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

#if __name__ == "__main__":
    #uvicorn.run(app, host="192.168.1.21", port=8080)