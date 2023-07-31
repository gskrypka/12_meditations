import streamlit as st

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

from elevenlabs import generate
from pydantic import BaseModel, Field
from typing import Sequence

from typing import Optional

with st.sidebar:
    openai_api_key = st.text_input ("Open AI API key", type="password", key="OPENAI_API_KEY")
    eleven_labs_api_key = st.text_input ("Eleven labs API key", type="password", key="ELEVENLABS_API_KEY")


# Define llm
llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4", temperature=0.7)

# Define Class
class Section(BaseModel):
    """Identifying information about a section."""

    text: str = Field(..., description="Section of meditation. After section always comes a pause or end.")
    pause: int = Field(..., description="Pause after section of meditation session. In seconds.")
    
class Sections(BaseModel):
    """Identifying all sections"""
    sections: Sequence[Section] = Field(..., description="Sections of meditation.")


# Open file systemprompt.txt
with open("systemprompt.txt", "r") as f:
    system_template = f.read()
    st.write(system_template)


# Define prompt template
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
human_template="{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages(messages=[system_message_prompt, human_message_prompt])


# Streamlit stucture 
st.title("11Meditations")
input = st.text_area("Descibe meditation")
if st.button("Generate"):
    llm_chain = create_structured_output_chain(Sections, llm=llm, prompt=chat_prompt)
    with st.spinner("Generating meditation..."):
        response = llm_chain.run({"text": input})

    for Section in response.sections:
        st.write(Section.text)
        st.write(Section.pause)

    with st.spinner("Generating audio..."):
        voice_over = AudioSegment.empty()
        for Section in response.sections:

            audio = generate(
                    text=Section.text,
                    voice="Charlotte",
                    model='eleven_multilingual_v1',
                    api_key=eleven_labs_api_key)
        
            audio_file = BytesIO(audio)
            audio_proc = AudioSegment.from_file(audio_file, format="mp3")
            voice_over += audio_proc
            pause_timing = Section.pause * 1000
            pause = AudioSegment.silent(duration=pause_timing)
            voice_over += pause
            
            
        background = AudioSegment.from_file("forest.mp3", format="mp3")
        meditation = voice_over.overlay(background, loop=True)

    
    meditation_file = meditation.export("meditation.mp3", format="mp3")
    audio_file = open("meditation.mp3", "rb")
    audio_bytes = audio_file.read()

    st.audio(audio_bytes, format="audio/mp3", start_time=0)

        
    


