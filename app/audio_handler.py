import io
from pydub import AudioSegment
import speech_recognition as sr


class AudioHandler:
    def __init__(self):
        self.recognizer = sr.Recognizer()

    def convert_audio_to_wav(self, audio_bytes: bytes) -> io.BytesIO:
        """
        Converts raw audio bytes to WAV format.
        """
        audio_file = io.BytesIO(audio_bytes)
        audio = AudioSegment.from_file(audio_file)
        wav_audio = io.BytesIO()
        audio.export(wav_audio, format="wav")
        wav_audio.seek(0)
        return wav_audio

    def transcribe_audio(self, wav_audio: io.BytesIO) -> str:
        """
        Transcribes the given WAV audio into text using Google Speech Recognition.
        """
        with sr.AudioFile(wav_audio) as source:
            audio_data = self.recognizer.record(source)
            try:
                return self.recognizer.recognize_google(audio_data)
            except sr.UnknownValueError:
                raise ValueError("Could not understand audio")
            except sr.RequestError as e:
                raise ConnectionError(f"Speech recognition service error: {e}")
