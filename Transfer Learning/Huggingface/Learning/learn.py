from dotenv import load_dotenv, find_dotenv
# Load model directly
from transformers import pipeline, AutoProcessor, AutoModelForVisualQuestionAnswering
from PIL import Image
import requests
import os
from openai import OpenAI
import streamlit as st

load_dotenv(find_dotenv())

HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

#* img2text
def img2text(url) : 
    # processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    # model = AutoModelForVisualQuestionAnswering.from_pretrained("Salesforce/blip2-opt-2.7b")

    pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

    text = pipe(url)[0]["generated_text"]
    print(text)
    return text




#* llm
def generate_story(scenario): 
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a story teller, skilled in narrating short stories based on a small scenario."},
            {"role": "user", "content": scenario}
        ],
        
    )
    print(completion.choices[0].message.content)

    return completion.choices[0].message.content




#* text-to-speech

def text2speech(story):
    API_URL = "https://api-inference.huggingface.co/models/speechbrain/tts-tacotron2-ljspeech"
    headers = {
        "Authorization": "Bearer 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"
    }
    payload = {
        "inputs": story
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    print(response)
    if response.status_code != 200:
        print("Error")
        return
    with open("audio.flac", "wb") as file: 
        file.write(response.content)



def main() : 
    '''
    
    '''
    st.set_page_config(page_title="Image To Story", page_icon="🚀")

    st.header("Turn Image Into a Amazing story")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()

        with open(uploaded_file.name, "wb") as file: 
            file.write(bytes_data)

        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        scenario = img2text(uploaded_file.name)
        story = generate_story(scenario)
        text2speech(story)

        with st.expander("scenario"): 
            st.write(scenario)

        with st.expander("story"): 
            st.write(story)

        st.audio("audio.flac")
if __name__ == "__main__":
    main()