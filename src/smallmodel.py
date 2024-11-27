import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)
CORS(app)

model_name = "Qwen/Qwen2-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, low_cpu_mem_usage=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cpu",
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16
)

@app.route('/', methods=['POST'])
def generate_response():
    try:
        prompt = request.json.get('prompt', '')
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400

        messages = [
            {"role": "system", "content": "You are a professional gym trainer providing fitness advice."},
            {"role": "user", "content": prompt}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt")
        
        response = model.generate(
            input_ids=inputs.input_ids,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.6
        )
        
        decoded = tokenizer.decode(response[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        return jsonify({"response": decoded})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "model": model_name}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)