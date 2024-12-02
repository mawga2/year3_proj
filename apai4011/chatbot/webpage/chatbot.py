from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Sample data for disease definitions
disease_definitions = {
    "flu": "Influenza, commonly known as the flu, is a viral infection that attacks your respiratory system.",
    "cold": "The common cold is a viral infection of your upper respiratory tract.",
    "diabetes": "Diabetes is a disease that occurs when your blood glucose, also called blood sugar, is too high.",
    "hypertension": "Hypertension, or high blood pressure, is a long-term medical condition in which the blood pressure in the arteries is persistently elevated."
}

@app.route('/')
def home():
    return render_template('home.html')  # Serve the homepage

@app.route('/chat')
def chat():
    return render_template('chatbot_interface.html')  # Serve the chatbot interface

@app.route('/get_response', methods=['POST'])
def get_response():
    user_message = request.form['message'].lower()
    if user_message in disease_definitions:
        response = disease_definitions[user_message]
    elif "symptoms" in user_message:
        response = "Please tell me your symptoms, and I will suggest possible diseases."
    elif "diseases" in user_message:
        response = "Please tell me the disease you want to know about."
    elif "help" in user_message:
        response = "I can assist you with two types of questions:<br>1. You can ask for information of diseases and symptoms.<br>2. You can provide your symptoms, and I will suggest possible diseases and recommend what you should do."
    else:
        response = f"I received your message: '{user_message}'"
    
    return jsonify({"response": response})

@app.route('/send_message', methods=['POST'])
def send_message():
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    message = data.get('message')

    # Process the message (e.g., store in database, send email)
    # For demonstration, we'll just print it
    print(f"Received message from {name} ({email}): {message}")

    return jsonify({"status": "success"}), 200

if __name__ == '__main__':
    app.run(debug=True)