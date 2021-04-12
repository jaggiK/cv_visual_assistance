import sys

sys.path.append('../../')

from user_interface.voice.text_to_speech_convertor import TTS


def main():
    # voice using festival
    text = "person aaan left side, 3 feet"
    tts_f = TTS(engine="festival")
    tts_f.speak(text)

    # voice using pyttxs3
    tts_p = TTS(engine="pyttsx3")
    tts_p.speak(text)


if __name__ == "__main__":
    main()
