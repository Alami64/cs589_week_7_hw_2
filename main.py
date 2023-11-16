import whisper
import queue
import threading
import click
import torch
import speech_recognition as sr
import torch
import numpy as np
import re
from openai import OpenAI
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import os
from dotenv import load_dotenv
import queue
import numpy as np
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
load_dotenv()

openai_api_key = os.environ['OPENAI_API_KEY']



@click.command()
@click.option("--model", default="base", help="Model to use", type=click.Choice(["tiny", "base", "small", 
              "medium", "large"]))
@click.option("--english", default=False, help="Whether to use the English model", is_flag=True, type=bool)
@click.option("--energy", default=300, help="Energy level for the mic to detect", type=int)
@click.option("--pause", default=0.8, help="Pause time before entry ends", type=float)
@click.option("--dynamic_energy", default=False, is_flag=True, help="Flag to enable dynamic energy", type=bool)
@click.option("--wake_word", default="jarvis", help="Wake word to listen for", type=str)
@click.option("--verbose", default=False, help="Whether to print verbose output", is_flag=True, type=bool)

def main(model, english, energy, pause, dynamic_energy, wake_word,verbose):

    if model != "large" and english:
        model = model + ".en"

    audio_model = whisper.load_model(model)
    audio_queue = queue.Queue()
    result_queue = queue.Queue()


    threading.Thread(target=record_audio, args=(audio_queue,energy,pause,dynamic_energy)).start()


    threading.Thread(target=transcribe_forever, args=(audio_queue, result_queue,audio_model,english,wake_word,verbose)).start()


    threading.Thread(target=reply, args=(result_queue,qa)).start()

    while True:
        print(result_queue.get())


def record_audio(audio_queue, energy,pause, dynamic_energy):
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = energy
    recognizer.pause_threshold = pause
    recognizer.dynamic_energy_threshold = dynamic_energy

    with sr.Microphone(sample_rate=16000) as source:
        print("Listening...")

        while True:
            
            audio = recognizer.listen(source)
            torch_audio = torch.from_numpy(
                np.frombuffer(audio.get_raw_data(), np.int16).flatten().astype(np.float32) / 32768.0
            )
            audio_queue.put_nowait(torch_audio)
 
def transcribe_forever(audio_queue, result_queue, audio_model,english,wake_word,verbose):

    while True:

        audio_data = audio_queue.get()
        
        try:
            if english:
                result = audio_model.transcribe(audio_data, language='english')
            
            else:
                result = audio_model.transcribe(audio_data)


            predicted_text = result["text"]

            if predicted_text.strip().lower().startswith(wake_word.strip().lower()):
                pattern = re.compile(re.escape(wake_word), re.IGNORECASE)
                predicted_text = pattern.sub("", predicted_text).strip()

                punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
                predicted_text = predicted_text.translate({ord(i): None for i in punc})


                if verbose:
                    print("You said the wake word...Processing {}...".format(predicted_text))

                print(f"Putting item into queue: {predicted_text}")
                result_queue.put_nowait(predicted_text)
                print("Item put into queue.")

            else:
                if verbose:

                    print(f"You did not say the wake word...Ignoring {predicted_text}")
        
        except Exception as e:
            print(f"An error occurred during transcription: {e}")



   
def load_db(file, chain_type, k):
    loader = PyPDFLoader(file_path=file)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0, max_tokens=100),
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=False,
        return_generated_question=False,
        memory=memory,
        output_key='answer'
    )

    return qa


qa = load_db(file='Module 6.pdf', chain_type='stuff', k=2)


# Function to reply to the transcribed text using the loaded PDF and OpenAI's model
def reply(result_queue, conversational_chain):
    while True:
        question = result_queue.get()
        response = conversational_chain({
            'question': question,
            'chat_history': []
        })
        answer = response['answer']
        print(f"Answer: {answer}")
        mp3_obj = gTTS(text=answer, lang="en", slow=False)
        mp3_obj.save("reply.mp3")
        reply_audio = AudioSegment.from_mp3("reply.mp3")
        play(reply_audio)

if __name__ == "__main__":
    main()