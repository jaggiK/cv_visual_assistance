import subprocess
import pyttsx3
from ..voice.voice_settings import rate, vol

class TTS(object):
    def __init__(self, engine="festival"):
        self._engine = engine
        if engine == 'pyttsx3':
            self._pyttsx3_engine = pyttsx3.init()
            self._pyttsx3_engine.setProperty('rate', rate)
            self._pyttsx3_engine.setProperty('volume', vol)

    def speak(self, text):
        if self._engine == "festival":
            subprocess.call('echo ' + text + '|festival --tts', shell=True)
        elif self._engine == "pyttsx3":
            self._pyttsx3_engine.say(text)
            self._pyttsx3_engine.runAndWait()

    def save_text(self, text, filename):
        file = open(filename, 'a')
        file.write(text)
        file.close()
