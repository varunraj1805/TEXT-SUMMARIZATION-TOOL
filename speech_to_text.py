import speech_recognition as sr

def transcribe_audio(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        print("Listening to audio...")
        audio_data = recognizer.record(source)
        print("Recognizing...")
        text = recognizer.recognize_google(audio_data)
        return text

if __name__ == "__main__":
    audio_path = "sample.wav"  # Replace with your own audio file path
    try:
        transcription = transcribe_audio(audio_path)
        print("Transcribed Text:\n", transcription)
    except Exception as e:
        print("Error:", e)
