from transformers import AutoTokenizer, GPTJForCausalLM, AutoModelForCausalLM
import torch

def load_model(model_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    if 'gptj' in model_path:
        model = GPTJForCausalLM.from_pretrained(model_path, revision="float16", low_cpu_mem_usage=True).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, revision="float16").to(device)
    print("model loaded")
    return model, tokenizer


def generate(model, tokenizer, texts, decoder_params, device=('cuda' if torch.cuda.is_available() else 'cpu')):
    encoding = tokenizer(texts, padding=True, return_tensors='pt').to(device)
    with torch.no_grad():
        generated_output = model.generate(**encoding, **decoder_params)
        scores = model(generated_output).logits
        generated_ids = generated_output[:, encoding['input_ids'].size(1):]
        generated_ids = generated_ids[:, 1:]
        scores = scores[:, :-1, :].log_softmax(-1)
        scores = scores.gather(2, generated_ids.unsqueeze(-1)).squeeze(-1)
        mask = ((generated_ids == tokenizer.eos_token_id).cumsum(1) <= 1).float()  # WARNING: assumes eos and pad are the same token
        logps = (scores*mask).sum(1)
        n_tokens = mask.sum(1)

    generated_texts = tokenizer.batch_decode(
        generated_output[:, encoding['input_ids'].size(1):],
        skip_special_tokens=True
    )
    completions = []
    for i in range(len(texts)):
        n = decoder_params['num_return_sequences']
        choices = []
        for j in range(n):
            idx = (i*n)+j
            choices.append({
                'text': generated_texts[idx],
                'logp': logps[idx].item(),
                'n_tokens': n_tokens[idx].item()
            })
        completions.append({
            'choices': choices
        })
    return completions
