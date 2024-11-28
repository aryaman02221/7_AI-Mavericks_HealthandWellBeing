from flask import Flask, render_template, request, jsonify
import subprocess
import torch
from backend import text_to_speech, speech_to_text, tokenizer, model, dataset

app = Flask(__name__)

# Route to open the command prompt
@app.route('/open-cmd', methods=['GET'])
def open_cmd():
    try:
        subprocess.Popen('start cmd', shell=True)  # Open command prompt in Windows
        return jsonify(success=True), 200
    except Exception as e:
        print(f"Error opening command prompt: {e}")
        return jsonify(success=False), 500

# Home Route
@app.route("/")
def home():
    return render_template("services.ejs")  # Adjusted path to match your file structure

# API Endpoint for Chat
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()  # Receive user input as JSON
    user_input = data.get("message", "")

    # Process user input using the backend
    if user_input.lower() == "exit":
        exit_greet = "Feel free to reach out anytime, later. Thank you!!"
        text_to_speech(exit_greet)
        return jsonify({"response": exit_greet})

    # Predict response using the model
    encoded = torch.tensor(tokenizer.encode(user_input)).unsqueeze(0)
    output = model(encoded)
    predicted_label = torch.argmax(output, dim=1)
    tag = dataset.labels.inverse_transform(predicted_label.cpu().numpy())[0]

    # Find appropriate response
    for intent in intents:
        if intent["tag"] == tag:
            response = random.choice(intent["responses"])
            text_to_speech(response)
            return jsonify({"response": response})

    return jsonify({"response": "Sorry, I couldn't understand that."})

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)

# from flask import Flask, request, jsonify
# import speech_recognition as sr  # For speech recognition
# from gtts import gTTS  # For converting text to speech
# import os
# import tempfile
#
# # Simulated model loading
# print("Loading model...")
# model_loaded = True
#
# app = Flask(__name__)
#
# @app.route("/")
# def home():
#     return "Python Backend is running."
#
# @app.route("/chat", methods=["POST"])
# def chat():
#     data = request.get_json()
#     user_message = data.get("message", "")
#     if not model_loaded:
#         return jsonify({"error": "Model not loaded"}), 500
#
#     # Simulated AI response
#     ai_response = f"AI Response to: '{user_message}'"
#     return jsonify({"response": ai_response})
#
# @app.route("/start-voice-chat", methods=["GET", "POST"])
# def start_voice_chat():
#     if request.method == "POST":
#         # Receive an audio file
#         if "audio" not in request.files:
#             return jsonify({"error": "No audio file uploaded."}), 400
#
#         audio_file = request.files["audio"]
#
#         # Save the audio file temporarily
#         temp_dir = tempfile.mkdtemp()
#         audio_path = os.path.join(temp_dir, "input_audio.wav")
#         audio_file.save(audio_path)
#
#         # Recognize speech
#         recognizer = sr.Recognizer()
#         try:
#             with sr.AudioFile(audio_path) as source:
#                 audio_data = recognizer.record(source)
#                 user_message = recognizer.recognize_google(audio_data)
#
#                 # Process recognized text with the AI model
#                 ai_response = f"AI Response to: '{user_message}'"
#
#                 # Convert AI response to speech
#                 tts = gTTS(ai_response)
#                 response_audio_path = os.path.join(temp_dir, "response_audio.mp3")
#                 tts.save(response_audio_path)
#
#                 # Send AI response text and audio
#                 with open(response_audio_path, "rb") as audio_response:
#                     audio_content = audio_response.read()
#                 return jsonify({"response": ai_response, "audio": audio_content.decode("latin1")})
#
#         except sr.UnknownValueError:
#             return jsonify({"error": "Speech not recognized. Please try again."}), 400
#         except Exception as e:
#             return jsonify({"error": f"An error occurred: {str(e)}"}), 500
#         finally:
#             # Clean up temporary files
#             if os.path.exists(audio_path):
#                 os.remove(audio_path)
#     else:
#         # Instructions for GET request
#         return jsonify({"message": "Please send audio as a POST request to start voice chat."})
#
# if __name__ == "__main__":
#     app.run(port=5000, debug=True)
