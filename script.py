import os
import json
import logging
from flask import Flask, request, jsonify
from vosk import Model, KaldiRecognizer
import wave
import urllib.request
import subprocess
from pydub import AudioSegment


# Настройка путей и логирования
FFMPEG_PATH = r"C:\Users\YCHIK\PycharmProjects\main.py\ffmpeg-2024-11-28-git-bc991ca048-full_build\ffmpeg-2024-11-28-git-bc991ca048-full_build\bin\ffmpeg.exe"
MODEL_PATH = r"C:\Users\YCHIK\PycharmProjects\main.py\vosk-model-small-ru-0.22"
os.environ["FFMPEG_BINARY"] = FFMPEG_PATH
AudioSegment.converter = FFMPEG_PATH


logging.basicConfig(level=logging.INFO)


# Проверка наличия модели Vosk
if not os.path.isdir(MODEL_PATH):
    raise FileNotFoundError(f"Модель не найдена по пути: {MODEL_PATH}")
model = Model(MODEL_PATH)


# Flask приложение
app = Flask(__name__)


# Функция конвертации MP3 в WAV
def convert_mp3_to_wav(mp3_path, wav_path):
    try:
        sound = AudioSegment.from_mp3(mp3_path).set_channels(1).set_frame_rate(16000)
        sound.export(wav_path, format="wav", codec="pcm_s16le")
        logging.info(f"MP3 конвертирован в WAV: {mp3_path} → {wav_path}")
    except Exception as e:
        logging.error(f"Ошибка при конвертации MP3 в WAV: {e}")
        raise


# Функция обработки аудиофайла для распознавания речи
def process_audio(file_path):
    try:
        with wave.open(file_path, "rb") as wf:
            recognizer = KaldiRecognizer(model, wf.getframerate())
            recognizer.SetWords(True)
            results = []
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if recognizer.AcceptWaveform(data):
                    results.append(json.loads(recognizer.Result()))
            results.append(json.loads(recognizer.FinalResult()))
        # Возвращаем словарь с ключом "results"
        return {"results": results}
    except Exception as e:
        logging.error(f"Ошибка при обработке аудио: {e}")
        return {"results": []}


# Скачивание аудиофайла по URL
def download_audio(url, download_path):
    try:
        urllib.request.urlretrieve(url, download_path)
        logging.info(f"Файл скачан: {download_path}")
    except Exception as e:
        logging.error(f"Ошибка при скачивании файла: {e}")
        raise


# Функция обработки результатов распознавания речи
def process_asr_results(raw_results):
    dialog = []
    durations = {"receiver": 0, "transmitter": 0}

    for result in raw_results["results"]:
        text = result.get("text", "").strip()
        if not text:
            continue

        # Определение источника, пола и громкости
        source = "receiver" if "день" in text.lower() else "transmitter"
        gender = "male" if source == "receiver" else "female"
        raised_voice = any(word.get("conf", 0) < 0.6 for word in result.get("result", []))

        duration = sum(
            word.get("end", 0) - word.get("start", 0)
            for word in result.get("result", [])
        )

        dialog.append({
            "source": source,
            "text": text,
            "duration": round(duration, 2),
            "raised_voice": raised_voice,
            "gender": gender
        })

        durations[source] += duration

    return {
        "dialog": dialog,
        "result_duration": {
            "receiver": round(durations["receiver"], 2),
            "transmitter": round(durations["transmitter"], 2),
        }
    }


# Эндпоинт для распознавания речи
@app.route('/asr', methods=['POST'])
def asr():
    try:
        data = request.json
        audio_path = data.get("path")
        if not audio_path:
            return jsonify({"error": "Не предоставлен путь к аудио"}), 400

        local_mp3 = "input.mp3"
        local_wav = "input.wav"

        if audio_path.startswith("http"):
            download_audio(audio_path, local_mp3)
        else:
            local_mp3 = audio_path

        # Конвертация MP3 в WAV
        convert_mp3_to_wav(local_mp3, local_wav)

        # Обработка файла и получение результатов
        vosk_results = process_audio(local_wav)

        # Преобразование результатов в нужный формат
        formatted_output = process_asr_results(vosk_results)
        return jsonify(formatted_output)

    except Exception as e:
        logging.error(f"Ошибка на endpoint ASR: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    try:
        subprocess.run([FFMPEG_PATH, '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        app.run(host='0.0.0.0', port=5000)
    except FileNotFoundError:
        logging.error(f"FFmpeg не найден по пути: {FFMPEG_PATH}")
