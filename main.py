from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from utils import predict_full_sentence, reorder_words, load_ngrams_from_file
import uvicorn
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str

# Global variables for caching
bigram_freq = None
trigram_freq = None

def load_ngrams_on_demand():
    global bigram_freq, trigram_freq
    if bigram_freq is None or trigram_freq is None:
        bigram_freq, trigram_freq = load_ngrams_from_file()

@app.on_event("startup")
async def startup_event():
    load_ngrams_on_demand()

def process_text(text):
    # First, lowercase everything
    text = text.lower()
    
    # Define a regex pattern to match 'i' as a word (surrounded by spaces or at the start/end of the string)
    pattern = r'\bi\b'
    
    # Use a lambda function to capitalize 'i' only when it's a word by itself
    return re.sub(pattern, lambda m: 'I', text)

@app.post("/complete_sentence")
def generate_sentence(request: TextRequest):
    load_ngrams_on_demand()

    # Process the input text
    processed_input = process_text(request.text)

    try:
        # Reorder words using POS tagging
        reordered_sentence = reorder_words(processed_input.split())
        reordered = reordered_sentence.split()

        # Predict full sentence by filling in the words between
        predicted_sentence = predict_full_sentence(reordered, trigram_freq, bigram_freq)

        return {
            "original": request.text,
            "processed": processed_input,
            "reordered": reordered_sentence,
            "sentence": predicted_sentence
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error: " + str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="192.168.1.21", port=5724)