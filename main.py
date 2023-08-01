import streamlit as st
import json
from pydub import AudioSegment
from io import BytesIO

import openai

# Improt llm chains
from langchain.chains import LLMChain

# Import chatbot models
from langchain.chat_models import ChatOpenAI

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate

#Import message schemas
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from langchain.chains.openai_functions import (
    create_openai_fn_chain,
    create_structured_output_chain,
)

from elevenlabs import generate, voices
from pydantic import BaseModel, Field
from typing import Sequence

from typing import Optional

with st.sidebar:
    st.subheader("API keys")
    openai_api_key = st.text_input ("Open AI API key", type="password", key="OPENAI_API_KEY")
    model = st.selectbox(
        'Choose Open AI Model (GPT-4 is recommended)',
        ('gpt-4', 'gpt-3.5-turbo')
    )
    eleven_labs_api_key = st.text_input ("Eleven labs API key", type="password", key="ELEVENLABS_API_KEY")

    st.subheader("Examples")
    st.write("Example 1")
    st.audio("meditations_examples/meditation_1.mp3", format="audio/mp3", start_time=0)
    st.write("Example 2")
    st.audio("meditations_examples/meditation_2.mp3", format="audio/mp3", start_time=0)

# Define Class
class Section(BaseModel):
    """Identifying information about a section."""

    text: str = Field(..., description="Section of meditation. After section always comes a pause or end.")
    pause: int = Field(..., description="Pause after section of meditation session. In seconds.")
    
class Sections(BaseModel):
    """Identifying all sections"""
    sections: Sequence[Section] = Field(..., description="Sections of meditation.")

# Define voice types
voice_list = {
    "Charlotte": "XB0fDUnXU5powFXDhCwa",
    "Thomas": "GBv7mTt0atIp3Br8iCZE"
}

# Open file systemprompt.txt
with open("prompts/systemprompt.txt", "r") as f:
    system_template = f.read()

with open("prompts/humanprompt.txt", "r") as f:
    human_template = f.read()
# Define prompt template
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages(messages=[system_message_prompt, human_message_prompt])

# Open file meditation_types.json
with open('meditation_types.json', 'r') as f:
    # Load JSON data from file
    data = json.load(f)


#Streamlit stucture 
st.title("🧘 12Meditations")

goal = st.text_area("Descibre the goals of meditation meditation session. Why do you want to meditate? What challenges face you? What do you want to achieve?")
meditation_type = st.selectbox(
    'Choose type of meditation',
     (list(data.keys()))
)

meditation_length = st.selectbox(
    'Select length of meditation',
     ('Short meditation. Up to 5 min', 'Medium lenght meditation. Around 5-10 minutes', 'Long meditation. More than 10 minutes.')
)

language= st.selectbox(
    'Select language of meditation',
     ('English', 'Spanish', 'French', 'Hindi', 'Italian', 'German', 'Polish', 'Portuguese')
)

background_sound_type = st.selectbox(
            'Select background sounds',
            ('forest', 'rain')
        )
st.audio("sounds/"+background_sound_type + ".mp3", format="audio/mp3", start_time=0)

voice= st.selectbox(
    'Select voice',
     ('Charlotte', 'Thomas')
)
            
if st.button("Generate", type="primary"):
    if eleven_labs_api_key is None or openai_api_key is None:
        st.error("Please enter API keys")
        st.stop()
    else:
        # Define llm
        llm = ChatOpenAI(openai_api_key=openai_api_key, model=model, temperature=0.7)
        llm_chain = create_structured_output_chain(Sections, llm=llm, prompt=chat_prompt)
        with st.spinner("Generating meditation script 📝. It can take few minutes"):

            # Get meditation value from data dictionary based on selected meditation type
            desc = meditation_type + ": " + data.get(meditation_type, "No meditation type selected")
            response = llm_chain.run({"type": meditation_type, "desc": desc, "goal": goal, "duration": meditation_length, "lang": language})

        with st.expander("see script"):
            for Section in response.sections:
                st.write(Section.text)
                st.write(Section.pause)

        with st.spinner("Generating audio 🎤. It can take few minutes"):
            voice_over = AudioSegment.empty()
            for Section in response.sections:

                ## Voice XB0fDUnXU5powFXDhCwa"
                audio = generate(
                        api_key=eleven_labs_api_key,
                        text=Section.text,
                        voice=voice_list[voice],
                        model='eleven_multilingual_v1'
                        )
            
                audio_file = BytesIO(audio)
                audio_proc = AudioSegment.from_file(audio_file, format="mp3")
                voice_over += audio_proc
                pause_timing = Section.pause * 1000
                pause = AudioSegment.silent(duration=pause_timing)
                voice_over += pause
                
            # Create background sound path which consist of "sounds" folder + background_sound_type + ".mp3"
            background_sound_path = "sounds/" + background_sound_type + ".mp3"
            # Create overlay
            background = AudioSegment.from_file(background_sound_path, format="mp3")
            meditation = voice_over.overlay(background, loop=True)
            
        # Creating audio file
        meditation_file = meditation.export("meditation.mp3", format="mp3")
        audio_file = open("meditation.mp3", "rb")
        audio_bytes = audio_file.read()

        st.audio(audio_bytes, format="audio/mp3", start_time=0)
        st.download_button(label="Download audio", data=audio_bytes, file_name="meditation.mp3", mime="audio/mp3")

        
    


