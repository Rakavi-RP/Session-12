import gradio as gr
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from dataclasses import dataclass
import tiktoken

# Config and model classes (same as training)
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

# [Include all model classes from your training code: CausalSelfAttention, MLP, Block, GPT]
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Load model and tokenizer
def load_model():
    model = GPT(GPTConfig())
    model.load_state_dict(torch.load('best_model.pt', map_location='cpu', weights_only=True))
    model.eval()
    return model

enc = tiktoken.get_encoding('gpt2')
model = load_model()

def generate_text(prompt, max_length=20, num_sequences=1):
    if not prompt:
        return "Please enter a prompt."
    
    # Encode the prompt
    input_ids = torch.tensor(enc.encode(prompt)).unsqueeze(0)
    
    # Generate multiple sequences
    generated_sequences = []
    for _ in range(num_sequences):
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_new_tokens=max_length)
            generated_text = enc.decode(output_ids[0].tolist())
            generated_sequences.append(generated_text)
    
    return "\n\n=== Next Sequence ===\n\n".join(generated_sequences)

def clear_inputs():
    return "", 20, 1, ""

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Text Generation with GPT")
    gr.Markdown("Enter a prompt and generate text completions.")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="Enter your prompt", lines=3)
            max_length = gr.Slider(minimum=1, maximum=50, value=20, step=1, label="Maximum Length")
            num_sequences = gr.Slider(minimum=1, maximum=5, value=1, step=1, label="Number of Sequences")
            
            with gr.Row():
                submit_btn = gr.Button("Submit")
                clear_btn = gr.Button("Clear")
        
        with gr.Column():
            output_text = gr.Textbox(label="Generated Text", lines=10)

    # Set up event handlers
    submit_btn.click(
        fn=generate_text,
        inputs=[input_text, max_length, num_sequences],
        outputs=output_text
    )
    
    clear_btn.click(
        fn=clear_inputs,
        inputs=[],
        outputs=[input_text, max_length, num_sequences, output_text]
    )

    # Add example
    gr.Examples(
        examples=[["Once upon a time", 30, 2]],
        inputs=[input_text, max_length, num_sequences]
    )

demo.launch() 