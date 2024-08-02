import streamlit as st
import argparse
import time
import pyaudio
import wave
import pyttsx3
import speech_recognition as sr
from googletrans import Translator
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from loader import get_embedding_function
from gtts import gTTS
import os
import tempfile
from playsound import playsound

CHROMA_PATH = "nomicDB1600"

# context - all chunks from db that best match the query
# question - actual question we want to ask

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def record_audio(output_filename, record_seconds=5, sample_rate=44100, chunk_size=1024):
    audio_format = pyaudio.paInt16
    channels = 1

    p = pyaudio.PyAudio()

    stream = p.open(format=audio_format,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk_size)

    st.info("Recording...")
    frames = []

    for _ in range(0, int(sample_rate / chunk_size * record_seconds)):
        data = stream.read(chunk_size)
        frames.append(data)

    st.success("Finished recording.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(output_filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(audio_format))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

def voice_to_text(audio_path, language='ur'):
    recognizer = sr.Recognizer()
    audio_file = sr.AudioFile(audio_path)

    with audio_file as source:
        audio_data = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio_data, language='language')
        return text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return f"Could not request results; {e}"

def translate_text(text, src_lang='ur', dest_lang='en'):
    translator = Translator()
    translation = translator.translate(text, src=src_lang, dest=dest_lang)
    return translation.text
def translate_text2(text, src_lang='en', dest_lang='ur'):
    translator = Translator()
    translation = translator.translate(text, src=src_lang, dest=dest_lang)
    return translation.text
def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB and give top k most relevant chunks
    results = db.similarity_search_with_score(query_text, k=4)

    # Combine the top results with the original question to generate the prompt
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print("prompt", prompt)

    model = Ollama(model="llama2")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    return response_text, sources

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    
def speak_text2(text, lang='en'):
    # Generate the audio using gTTS
    tts = gTTS(text=text, lang='ur', slow=False)
    
    # Save the audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        tts.save(temp_audio.name)
        temp_audio_path = temp_audio.name
    
    # Play the audio using playsound
    playsound(temp_audio_path)
    
    # Cleanup: Remove the temporary file
    os.remove(temp_audio_path)
    
def main():
    st.title("BOT X: Ask anything regarding Urdu X")

    # Initialize conversation history
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
        
    # Initialize language selection
    if "language" not in st.session_state:
        st.session_state.language = "en"
    
    # Language selection buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Urdu"):
            st.session_state.language = "ur"
    with col2:
        if st.button("English"):
            st.session_state.language = "en"
    
    st.write(f"Selected Language: {st.session_state.language}")

    # User input for the question
    user_question = st.text_input("Enter your question here:")



    if st.button("Send"):
        if user_question:
            with st.spinner("Assistant is typing..."):
                response, sources = query_rag(user_question)
                #response=user_question
                st.session_state.conversation.append({"question": user_question, "response": response})#, "sources": sources})
                #if st.session_state.language == "ur":
                #    speak_text2(roman_urdu_text, lang='ur')
                #else:
                #    speak_text(response)
        # Voice input for the question
    if st.button("Record Voice in Urdu"):
        audio_path = "input_audio.wav"
        record_audio(audio_path)
        #roman_urdu_text = voice_to_text(audio_path)
        roman_urdu_text = voice_to_text(audio_path, language=st.session_state.language)
        st.write(roman_urdu_text)
        
        #if st.session_state.language == "ur":
        #speak_text2(roman_urdu_text, lang='ur')
        #else:
        english_text = translate_text(roman_urdu_text)
        user_question = english_text
        st.write(user_question)
        #speak_text(user_question)
        with st.spinner("Assistant is typing..."):
                response, sources = query_rag(user_question)
                #response=user_question
                st.session_state.conversation.append({"question": user_question, "response": response})#, "sources": sources})
                #speak_text(response)
    
    if st.button("🔊 Speak"):
        if st.session_state.conversation:
            last_convo = st.session_state.conversation[-1]
            if st.session_state.language == "ur":
                #st.write('aaaa')
                t = translate_text2(last_convo["response"])
                #st.write(t)
                speak_text2(t, lang='en')
            else:
                speak_text(last_convo["response"])

    # Display conversation history
    for idx, convo in enumerate(st.session_state.conversation):
        st.markdown(f'<p style="font-family:Helvetica; font-weight:bold">Question {idx + 1}: {convo["question"]}</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-family:Helvetica;">{convo["response"]}</p>', unsafe_allow_html=True)
        #st.write(f"Sources: {convo['sources']}")

if __name__ == "__main__":
    main()
