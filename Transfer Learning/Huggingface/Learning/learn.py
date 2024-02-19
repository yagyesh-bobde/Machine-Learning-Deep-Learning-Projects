from dotenv import load_dotenv, find_dotenv
# Load model directly
from transformers import pipeline, AutoProcessor, AutoModelForVisualQuestionAnswering
from PIL import Image
import requests


load_dotenv(find_dotenv())

#* img2text
def img2text(url) : 
    # processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    # model = AutoModelForVisualQuestionAnswering.from_pretrained("Salesforce/blip2-opt-2.7b")

    pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

    text = pipe(url)[0]["generated_text"]
    print(text)
    return text


img2text("https://img.freepik.com/free-photo/medium-shot-kids-drawing-together_23-2149199889.jpg")

#* llm

#* text-to-speech