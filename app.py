import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import dispatch_model


torch.cuda.set_per_process_memory_fraction(0.5) 

app = Flask(__name__)
CORS(app) 


#model auto loadedd

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name
)



#route to get res

@app.route('/', methods=['POST'])
def generate_response():
    try:
        
        data = request.json
        prompt = data.get('prompt', '')
        
        
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400

        # Prepare messages for the model
        messages = [
            {"role": "system", "content": "You are a professional gym trainer. You provide expert advice on fitness and training routines."},
            {"role": "user", "content": prompt}
        ]

        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize inputs
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # Generate response
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            use_cache=True
        )

        # Decode generated tokens
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "model": model_name}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)