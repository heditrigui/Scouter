from gtts import gTTS
import os

myText = str(id)
print(str(id))
language ='en'
output = gTTS(text=myText, lang=language, slow=False)

output.save("output.mp3")

os.system("omxplayer output.mp3")