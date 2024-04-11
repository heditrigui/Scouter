from num2words import num2words
from subprocess import call
import subprocess


cmd_beg= 'espeak '
cmd_end= ' omxplayer /home/pi/Desktop/FACE2/audioText.wav  ' # To play back the stored .wav file and to dump the std errors to /dev/null
cmd_out= '--stdout > /home/pi/Desktop/FACE2/audio/Text.wav ' # To store the voice file

text = input()
print(text)


text = text.replace(' ', '_')

#Calls the Espeak TTS Engine to read aloud a Text
call([cmd_beg+cmd_out+text+cmd_end], shell=True)
moviepath = '/home/pi/Desktop/FACE2/audio/Text.wav'
omxprocess = subprocess.Popen(['omxplayer', moviepath], stdin=subprocess.PIPE)