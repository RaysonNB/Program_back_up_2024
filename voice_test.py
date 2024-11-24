from gtts import gTTS
from playsound import playsound
import time
def say(g):
    tts = gTTS(g)

    # Save the speech as an audio file
    speech_file = "speech.mp3"
    tts.save(speech_file)

    # Play the speech
    playsound(speech_file)

say_list=['first', 'second', 'third', '4th', '5th', '6th', '7th', '8th', '9th', '10th']
for i in range(len(say_list)):
    t2="the "+say_list[i]+" object is "+"isacck"
    say(t2)
    time.sleep(0.5)
