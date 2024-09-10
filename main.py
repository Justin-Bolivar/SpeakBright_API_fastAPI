from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_generation import create_sentence_with_bert

app = FastAPI()

class InputText(BaseModel):
    text: str

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
