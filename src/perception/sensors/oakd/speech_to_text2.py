# from pocketsphinx import LiveSpeech
# for phrase in LiveSpeech(): print(phrase)

from pocketsphinx import LiveSpeech

# speech = LiveSpeech(lm=False, keyphrase='save', kws_threshold=1e-20)
# for phrase in speech:
#     print(phrase.segments(detailed=True))

# from pocketsphinx import LiveSpeech
for phrase in LiveSpeech(kws_threshold=1e-2): print(phrase)