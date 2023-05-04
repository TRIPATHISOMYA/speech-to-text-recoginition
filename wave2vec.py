import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from pathlib import Path
#load pre-trained model and tokenizer
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
from IPython.display import Audio
from IPython.core.display import display

def beep():
    display(Audio('speech.wav', autoplay=True))

speech,rate = librosa.load(r".\sample_audio\speech.wav",sr=16000)
input_values = tokenizer(speech, return_tensors = 'pt').input_values
#Store logits (non-normalized predictions)
logits = model(input_values).logits
#Store predicted id's
predicted_ids = torch.argmax(logits, dim =-1)
#decode the audio to generate text
transcriptions = tokenizer.decode(predicted_ids[0])
print(transcriptions)