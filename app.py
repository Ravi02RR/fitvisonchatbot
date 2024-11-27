
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


app = Flask(__name__)


model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)


@app.route('/generate', methods=['POST'])
def generate_response():
    user_input = request.json.get('prompt', "")
    
   
    messages = [
        {"role": "system", "content": "You are a professional gym trainer. You provide expert advice on fitness and training routines."},
        {"role": "user", "content": user_input}
    ]
    
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return jsonify({"response": response})


if __name__ == '__main__':
    app.run(debug=True)
