# voice interface
Played around with`festival`, `pyttx3`, and `gtts`.

`gtts` is most natural, however it requires internet and requires intermediate step of  serializing audio file.

`pyttx3` is a flexible offline, with various setting levels. 
voice clarity can be better.

`festival` does not seem like a complete package and is old. 
However, it is clearer than `pyttx3`. So sticking to this for meantime. 

## set up `festival`
`sudo apt-get install festival -y`
 
## set up `pyttsx3`
 `pip install pyttsx3`
 
 `sudo apt-get update && sudo apt-get install espeak`