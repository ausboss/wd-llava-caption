import os
from flask import Flask, request, jsonify
import base64
from io import BytesIO
from PIL import Image
from wdclass import WaifuDiffusionTagger
from dotenv import load_dotenv
from ollama import generate

load_dotenv()

# Assuming 'llama:34b-v1.6-q4_K_M' is the intended model; adjust as necessary
ollama_model = 'llava:34b-v1.6-q4_K_M'

hf_token = os.getenv('HF_TOKEN')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    tagger = WaifuDiffusionTagger(hf_token)  # Ensure hf_token is defined
    data = request.json

    # Directly use the base64 string from the request
    image64 = data['image']
    image_data = BytesIO(base64.b64decode(image64))
    image = Image.open(image_data)

    # Other parameters remain the same
    model_repo = data.get('model_repo', "SmilingWolf/wd-v1-4-swinv2-tagger-v2")
    # general_thresh = float(data.get('general_threshold', 0.75))
    general_thresh = float(data.get('general_threshold', 0.25))
    general_mcut_enabled = data.get('general_mcut_enabled', False)
    character_thresh = float(data.get('character_threshold', 0.85))
    character_mcut_enabled = data.get('character_mcut_enabled', False)

    # Pass the base64 string directly to predict
    general_tags, rating, character_tags, general_tags = tagger.predict(
        image,
        model_repo,
        general_thresh,
        general_mcut_enabled,
        character_thresh,
        character_mcut_enabled,
    )

    # Assuming the processing in predict method is adjusted as previously suggested
    general_tags = ", ".join(general_tags)
    character_tags = ", ".join(character_tags.keys())

    return jsonify({
        'general_tags': general_tags,
        'rating': rating,
        'character_tags': character_tags
    })



@app.route('/llava_caption_prompt', methods=['POST'])
def llava_caption_prompt():
        print('received image')
        data = request.json
        # gets a base64 iamge and converts it to a PIL image
        base64_image = data.get('image')
        prompt = data.get('prompt')
        image_data = BytesIO(base64.b64decode(base64_image))
        image = Image.open(image_data).convert("RGB")
        image = image.convert("RGB")
        buffered = BytesIO()
        image.save(buffered, format="JPEG")    
        image_bytes = buffered.getvalue()
        # Assume 'llama:13b-v1.6' is the intended model; adjust as necessary
        responses = generate(model=ollama_model,
                            prompt=f'{prompt}',
                            images=[image_bytes], 
                            stream=False)  # Using stream=False for a single response
    
        print(responses)
        return jsonify({'description': responses['response']})


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5003)
