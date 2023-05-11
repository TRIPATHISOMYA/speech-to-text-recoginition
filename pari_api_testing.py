import speech_recognition as sr
import speech_recognition as sr
from time import ctime
import time
import os #it has an attribute remove so it will prevent our audio from piling up
import random  #to randomly generate a file name
import webbrowser
from gtts import gTTS  #it will record our audio
import playsound #it will play instantly not opening our default player
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import operator
import librosa
import gradio as gr

model = SentenceTransformer('all-MiniLM-L6-v2')
df = pd.read_excel("Faq's.xlsx")

recognizer = sr.Recognizer()

df = pd.read_excel("Faq's.xlsx")

def MakingEmbeddings(df):
  embeddings = {}
  for idx,ques in enumerate(df['questions']):
    embeddings[idx] = model.encode(ques)
  return embeddings


def find_most_similar_ques(query, embeddings, df):
    similarity_dict = {}
    query_embedding = model.encode(query)
    for i in embeddings.items():
        similarity = util.dot_score(i[1], query_embedding)
        similarity_dict[i[0]] = similarity
    # print('similarity_dict: ',similarity_dict)
    sorted_dict = dict(sorted(similarity_dict.items(), key=operator.itemgetter(1), reverse=True))
    # print(sorted_dict)
    if sorted_dict:
        key = list(sorted_dict.keys())[0]
        # print(key)
        matched_answer = df.iloc[key]['answers']
        return matched_answer
    else:
        return "No Answer found"

embeddings = MakingEmbeddings(df)
r = sr.Recognizer()
def pari_speak(audio_string):

    tts = gTTS(audio_string,lang='en')
    r = random.randint(1,1000000)
    audio_file = 'audio_'+str(r)+'.mp3'
    tts.save(audio_file)
    #playsound.playsound(audio_file)
    #print(audio_string)
    #os.remove(audio_file)
    return audio_file


def asr_transcript(audio,state = ""):
    print(audio)
    # speech = load_data(input_file)
    harvard = sr.AudioFile(audio)
    with harvard as source:

        audio = r.record(source)
    try:
        text = recognizer.recognize_google(
            #open(audio,'rb'),
            audio,
            language="en-US"
        )
        print('Decoded text: {}'.format(text))
        answer = find_most_similar_ques(text, embeddings, df)
        print('responded answer by pari: ', answer)
        audio_file = pari_speak(answer)
        return text, answer, audio_file
    except sr.RequestError as e:

        print("Could not request results; {0}".format(e))


    except sr.UnknownValueError:

        print("unknown error occurred")

demo = gr.Interface(
    title='NuWe.AI',
    fn=asr_transcript,
    inputs=[
        gr.inputs.Audio(source="microphone", type="filepath")
    ],

    outputs=[
        gr.Textbox(label="Question"),
        gr.Textbox(label="Answer"),
        'audio'
    ],
    article=
    '''<div>
        <p style="text-align: center"> All you need to do is to ask pari your concern, then wait for compiling. After that click on Play/Pause for listing to the audio for the asked query!</p>
    </div>''',
    live=True).launch(debug=True,share=True)

