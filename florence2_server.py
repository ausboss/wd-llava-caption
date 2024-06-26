from flask import Flask, request, jsonify
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import base64
from io import BytesIO
import torch

app = Flask(__name__)

# Load the model and processor (do this outside the route for efficiency)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

@app.route('/florence2_caption', methods=['POST'])
def florence2_caption():
    try:
        data = request.json
        base64_image = data.get('image')
        
        # Decode base64 image
        image_data = BytesIO(base64.b64decode(base64_image))
        image = Image.open(image_data).convert("RGB")
        
        # Set the prompt for detailed caption
        prompt = "<DETAILED_CAPTION>"
        
        # Prepare the inputs
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
        
        # Generate the caption
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=False,
                num_beams=3,
            )
        
        # Decode and post-process the generated text
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(generated_text, task=prompt, image_size=(image.width, image.height))
        
        # Extract the caption
        caption = parsed_answer[prompt]
        
        return jsonify({'description': caption})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5004)