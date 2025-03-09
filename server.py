from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from PIL import Image
import io
import base64
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# AI API Configuration
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
    system_instruction="You are an AI specifically responsible for the returns department, helping the company handle customer return requests. Your main task is to determine whether the product described by the customer meets the return/refund criteria. Lastly, answering questions from the customer.\n\nFirst you would ask for the product ID, it should be an 8 digit number. If it fits inside the Product_ID.txt, pass to the next stage, if not, raise an error message: “Invalid Product ID, please try again.” The Product_ID goes like “ Return ID | Product name | The date it was sold | allowed period | is it a final sale or personalized item | ”\n\nNext, you need to ask for information about the product: \n- Return requested within the allowed period, allowed period would be 1 year, if it’s over the allowed period, no matter what the condition is, it is not eligible.\n- Unused, original condition & packaging\n- Not a final sale or personalized item\n- Defective, damaged, or malfunctioning item\n- Item arrived damaged\n- Product differs significantly from description\n\nThere could also be image uploading to you, and you should get some information to determine if it is eligible or not. If the Product_ID does not contain every element, ask for the user.\n\nThe analysis should be reasonable but also not too restrictive. Based on the information, come up with a conclusion of if the product is eligible or not, give a valid reason too.\n\nIf  eligible : \nAsk the customer if they want to replace the product or just get a refund.\n\nIf non-eligible: \nTell them the reason and the refund/ replacement process is not successful\n",
)

history = []

def get_product_info(product_id):
    """Fetch product details from Product ID.txt and let AI process it."""
    try:
        with open("Product ID.txt", "r") as file:
            for line in file:
                parts = line.strip().split(" | ")
                if parts[0] == product_id:
                    return " | ".join(parts)  # Return raw data for AI processing
    except FileNotFoundError:
        return None
    return None

@app.route("/chat", methods=["POST"])
def chat():
    """Handles text-based chat and retrieves product info when ID is provided."""
    data = request.json
    user_input = data.get("message", "").strip()

    if not user_input:
        return jsonify({"response": "Invalid input"}), 400

    # Check if input is an 8-digit Product ID
    if user_input.isdigit() and len(user_input) == 8:
        product_info = get_product_info(user_input)
        if product_info:
            # Append product details to chat history
            history.append({"role": "user", "parts": [f"Product ID: {user_input}"]})
            history.append({"role": "model", "parts": [f"Product Details: {product_info}"]})

            # Let AI process the product details
            chat_session = model.start_chat(history=history)
            response = chat_session.send_message(f"Here are the details of the product:\n{product_info}\n"
                                                 "Please analyze if it is eligible for return.")
            model_response = response.text

            history.append({"role": "model", "parts": [model_response]})
            return jsonify({"response": model_response})

        else:
            return jsonify({"response": "Invalid Product ID, please try again."})

    # Continue AI chat as usual
    chat_session = model.start_chat(history=history)
    response = chat_session.send_message(user_input)
    model_response = response.text

    history.append({"role": "user", "parts": [user_input]})
    history.append({"role": "model", "parts": [model_response]})

    return jsonify({"response": model_response})


@app.route("/upload", methods=["POST"])
def upload_image():
    """Handles image uploads and integrates them into the chat history."""
    if "image" not in request.files:
        return jsonify({"response": "No image provided"}), 400

    file = request.files["image"]
    image = Image.open(file)

    try:
        # Send the image to the AI while maintaining chat history
        chat_session = model.start_chat(history=history)
        response = chat_session.send_message([image])

        response.resolve()
        model_response = response.text

        # Append the image message to history
        history.append({"role": "user", "parts": ["[User uploaded an image]"]})
        history.append({"role": "model", "parts": [model_response]})

    except Exception as e:
        return jsonify({"response": f"Error processing image: {str(e)}"}), 500

    return jsonify({"response": model_response})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
