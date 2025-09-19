#Korean SNS Multi-turn Conversation Dataset Decoder-only Transformer LLM Tutorial

import os
import json
import random
import math
import time
import pickle
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from glob import glob
import gc

# GPU and Hyperparameter Configuration

# GPU setup
def setup_device_and_mode():
    print(f"\n{'='*10} GPU Environment {'='*10}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not torch.cuda.is_available():
        print("GPU is not available. Setting CPU mode.")
        return device, "CPU"
    
    gpu_name = torch.cuda.get_device_name()
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"‧ GPU: {gpu_name}")
    print(f"‧ GPU Memory: {gpu_memory:.1f}GB")
    
    if "A100" in gpu_name:
        print("‧ A100 GPU detected - Setting A100 mode")
        return device, "A100"
    elif any(gpu in gpu_name for gpu in ["L4", "T4", "V100", "K80", "P100"]):
        print("‧ L4/T4/V100-class GPU detected - Setting medium performance mode")
        return device, "T4"
    else:
        print("‧ Other GPU detected - Setting general GPU mode")
        return device, "T4"

# Configuration dictionary by GPU mode
GPU_CONFIGS = {
    "A100": {
        "SEQ_LEN": 512, "BATCH_SIZE": 32, "EMBED_DIM": 1024, "N_LAYERS": 12,
        "N_HEADS": 16, "FFN_DIM": 4096, "MAX_CONVERSATIONS": 10000,
        "EPOCHS": 5, "NUM_WORKERS": 4, "ENABLE_TRANSFORMER_DEMO": True
    },
    "T4": {
        "SEQ_LEN": 256, "BATCH_SIZE": 16, "EMBED_DIM": 768, "N_LAYERS": 12,
        "N_HEADS": 8, "FFN_DIM": 2048, "MAX_CONVERSATIONS": 5000,
        "EPOCHS": 2, "NUM_WORKERS": 2, "ENABLE_TRANSFORMER_DEMO": True
    },
    "CPU": {
        "SEQ_LEN": 64, "BATCH_SIZE": 4, "EMBED_DIM": 256, "N_LAYERS": 4,
        "N_HEADS": 4, "FFN_DIM": 1024, "MAX_CONVERSATIONS": 1000,
        "EPOCHS": 1, "NUM_WORKERS": 0, "ENABLE_TRANSFORMER_DEMO": True
    }
}

# Global configuration
GLOBAL_CONFIG = {
    "USE_CACHE": True,

    "USE_CUSTOM_TOKENIZER": False,
    "PRETRAINED_TOKENIZER": "klue/bert-base",

    "MAX_FILES_FOR_FULL_DATASET": 10000,

    "DEFAULT_CACHE_FILE": "conversations_default.pkl",
    "DATA_CACHE_PATH": "/content/drive/MyDrive/Colab Notebooks/data_cache",
    "LABELED_DATA_PATH": "/content/drive/MyDrive/Colab Notebooks/sns_multi_turn_dataset",

    "VOCAB_SIZE": 20000,
    "SAVE_EVERY": 6000,  # Model checkpoint save interval

    "LR": 2e-4,
    "WARMUP_RATIO": 0.1,  # Learning rate warmup ratio
    "SEED": 42    
}

# Directory creation function
def create_directories():
    print(f"\n{'='*10} Directory Setup {'='*10}")
    
    paths_to_create = [
        GLOBAL_CONFIG["DATA_CACHE_PATH"],
        GLOBAL_CONFIG["LABELED_DATA_PATH"]
    ]
    
    for path in paths_to_create:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            print(f"‧ Created directory: {path}")
        else:
            print(f"‧ Directory already exists: {path}")

# Execute setup and apply configuration
DEVICE, GPU_MODE = setup_device_and_mode()
config = GPU_CONFIGS[GPU_MODE]

# Set global variables
for key, value in {**config, **GLOBAL_CONFIG}.items():
    globals()[key] = value

# Create necessary directories
create_directories()

# Configuration output
print(f"\n{'='*10} Configuration {'='*10}")
print(f"‧ Batch size: {BATCH_SIZE}, Sequence length: {SEQ_LEN}")
print(f"‧ Model size: {EMBED_DIM}D, {N_LAYERS} layers, {N_HEADS} heads")
print(f"‧ Transformer Demo Log: {'Enabled' if ENABLE_TRANSFORMER_DEMO else 'Disabled'}")
print(f"‧ Max conversations: {MAX_CONVERSATIONS:,}")

# GPU optimization and seed setup
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

torch.manual_seed(SEED)
random.seed(SEED)

# Korean SNS Multi-turn Conversation Dataset Decoder-only Transformer LLM Tutorial

# Time measurement utility function
def log_time(start_time, step_name):
    elapsed = time.time() - start_time
    hours, minutes = int(elapsed // 3600), int((elapsed % 3600) // 60)
    seconds = elapsed % 60

    if hours > 0:
        time_str = f"{hours}h {minutes}m {seconds:.1f}s"
    elif minutes > 0:
        time_str = f"{minutes}m {seconds:.1f}s"
    else:
        time_str = f"{seconds:.1f}s"

    print(f"==> [Elapsed Time] {step_name}: {time_str}")
    return time.time()

# JSON file filtering function for training data
def quick_filter_json_files(json_files, max_files):
    if len(json_files) <= max_files:
        return json_files

    print(f"File filtering: {len(json_files):,} → {max_files:,} files")

    file_info = []
    for file_path in json_files:
        try:
            size = os.path.getsize(file_path)
            if 500 < size < 50000:
                file_info.append((file_path, size))
        except:
            continue

    file_info.sort(key=lambda x: x[1])
    step = max(1, len(file_info) // max_files)
    selected_files = [info[0] for info in file_info[::step][:max_files]]

    print(f"Selection completed: {len(selected_files):,} files")
    return selected_files

# Tokenizer management class for KLUE/BERT-base or custom tokenizer
class TokenizerManager:
    def __init__(self, use_custom=False, pretrained_model="klue/bert-base"):
        self.use_custom = use_custom
        self.pretrained_model = pretrained_model
        self.tokenizer = None
        self.vocab_size = None

    def setup_tokenizer(self, conversations=None):
        start_time = time.time()

        if self.use_custom:
            print("> Using BERT tokenizer instead of custom tokenizer")
            self.use_custom = False

        print(f"> Loading tokenizer: {self.pretrained_model}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)
            if not hasattr(self.tokenizer, 'pad_token') or self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token if hasattr(self.tokenizer, 'eos_token') else '[PAD]'
            special_tokens = ["[TURN]", "[SPKA]", "[SPKB]"]
            self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
            self.vocab_size = len(self.tokenizer)
            tokenizer_type = "KLUE/BERT"

        except Exception as e:
            print(f"> Tokenizer loading failed: {e}")
            print("> Using BERT multilingual tokenizer")

            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
            special_tokens = ["[TURN]", "[SPKA]", "[SPKB]"]
            self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
            self.vocab_size = len(self.tokenizer)
            tokenizer_type = "BERT Multilingual"

        log_time(start_time, f"> {tokenizer_type} tokenizer setup complete")
        print(f"> Vocabulary size: {self.vocab_size:,}")
        return self.tokenizer

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def get_special_tokens(self):
        special_tokens = {}
        special_tokens['PAD'] = self.tokenizer.pad_token_id or 0
        special_tokens['UNK'] = self.tokenizer.unk_token_id or 1
        special_tokens['BOS'] = self.tokenizer.cls_token_id or 2
        special_tokens['EOS'] = self.tokenizer.sep_token_id or 3

        try:
            special_tokens['TURN_SEP'] = self.tokenizer.convert_tokens_to_ids('[TURN]')
            special_tokens['SPEAKER_A'] = self.tokenizer.convert_tokens_to_ids('[SPKA]')
            special_tokens['SPEAKER_B'] = self.tokenizer.convert_tokens_to_ids('[SPKB]')
        except:
            special_tokens['TURN_SEP'] = 4
            special_tokens['SPEAKER_A'] = 5
            special_tokens['SPEAKER_B'] = 6

        return special_tokens

def process_json_file(json_file_path):
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if 'utterances' in data and len(data['utterances']) >= 2:
            conversation = []
            for utterance in data['utterances']:
                speaker = utterance.get('speaker', 'unknown')
                text = utterance.get('text', '').strip()
                if text and len(text) < 2000:
                    conversation.append({'speaker': speaker, 'text': text})

            if 2 <= len(conversation) <= 100:
                return conversation
    except:
        pass
    return None

# LLM training data cache file loading/saving class
class DataCache:
    def __init__(self):
        self.cache_dir = Path(DATA_CACHE_PATH)
        self.cache_dir.mkdir(exist_ok=True)

    def load_any_existing_cache(self):
        try:
            cache_path = self.cache_dir / DEFAULT_CACHE_FILE

            if not cache_path.exists():
                print(f"> [Cache] Default cache file not found: {DEFAULT_CACHE_FILE}")
                return None

            print(f"> [Cache] Loading default cache file: {DEFAULT_CACHE_FILE}")

            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)

            conversations = cache_data.get('conversations', [])
            print(f"> [Cache] Cache loaded successfully ({len(conversations):,} conversations)")

            if len(conversations) == 0:
                print("> [Cache] Cache is empty - returning None")
                return None

            return conversations

        except Exception as e:
            print(f"[Cache] Cache loading failed: {e}")
            return None

    def save_default_cache(self, conversations):
        cache_path = self.cache_dir / DEFAULT_CACHE_FILE

        try:
            cache_data = {
                'conversations': conversations,
                'timestamp': time.time(),
                'total_conversations': len(conversations)
            }

            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            file_size = cache_path.stat().st_size / (1024 * 1024)
            print(f"> [Cache] New cache saved successfully: {DEFAULT_CACHE_FILE} ({file_size:.1f}MB)")

        except Exception as e:
            print(f"> [Cache] Cache saving failed: {e}")

# LLM training JSON dataset loader class (uses cache file if available instead of JSON)
class FastSNSDataLoader:
    def __init__(self, labeled_data_path):
        self.labeled_data_path = labeled_data_path
        self.cache = DataCache() if USE_CACHE else None

    def load_json_conversations(self):
        start_time = time.time()

        if self.cache:
            cached_conversations = self.cache.load_any_existing_cache()
            if cached_conversations and len(cached_conversations) > 0:
                log_time(start_time, f"> Loaded from cache ({len(cached_conversations):,} conversations)")
                return cached_conversations

        print("> [Processing] Starting new data processing...")

        json_pattern = os.path.join(self.labeled_data_path, "**", "*.json")
        json_files = glob(json_pattern, recursive=True)
        print(f"Found JSON files: {len(json_files):,}")

        if MAX_FILES_FOR_FULL_DATASET and len(json_files) > MAX_FILES_FOR_FULL_DATASET:
            json_files = quick_filter_json_files(json_files, MAX_FILES_FOR_FULL_DATASET)

        file_search_time = log_time(start_time, "JSON file search and filtering")

        print("Sequential JSON parsing...")
        conversations = []

        batch_size = 2000
        for i in range(0, len(json_files), batch_size):
            batch_files = json_files[i:i+batch_size]
            batch_desc = f"JSON processing batch {i//batch_size + 1}/{(len(json_files)-1)//batch_size + 1}"

            for json_file in tqdm(batch_files, desc=batch_desc, leave=False):
                conv = process_json_file(json_file)
                if conv:
                    conversations.append(conv)

            if i % (batch_size * 2) == 0:
                gc.collect()

        print(f"> Extracted conversations: {len(conversations):,}")
        log_time(file_search_time, "JSON file processing complete")

        if self.cache and conversations:
            self.cache.save_default_cache(conversations)

        return conversations

# Transformer process sample output function
def detailed_transformer_demo(model, tokenizer_manager, sample_text="안녕하세요! 오늘 어떤 하루 보내셨나요?"):
    print("\n" + "="*100)
    print("■ TRANSFORMER INTERNAL PROCESS ANALYSIS")
    print("="*100)

    model.eval()

    print(f"\n1. Input Sentence Analysis")
    print(f"   Original sentence: '{sample_text}'")
    print(f"   Sentence length: {len(sample_text)} characters")

    # Tokenization process
    print(f"\n2. Tokenization Process")
    tokens = tokenizer_manager.encode(sample_text)
    print(f"   Token ID array: {tokens[:10]}...")
    print(f"   Token count: {len(tokens)}")

    try:
        token_texts = tokenizer_manager.tokenizer.convert_ids_to_tokens(tokens[:10])
        print(f"   Token texts (first 10): {token_texts}")
    except:
        pass

    # Add special tokens
    special_tokens = tokenizer_manager.get_special_tokens()
    full_tokens = [special_tokens['BOS'], special_tokens['SPEAKER_A']] + tokens[:10] + [special_tokens['EOS']]

    print(f"\n3. Special Token Addition")
    print(f"   BOS (start): {special_tokens['BOS']}")
    print(f"   SPEAKER_A: {special_tokens['SPEAKER_A']}")
    print(f"   EOS (end): {special_tokens['EOS']}")
    print(f"   Full sequence: {full_tokens}")

    # Padding
    seq_len = min(16, model.seq_len)  # Smaller size for visualization
    if len(full_tokens) < seq_len:
        full_tokens.extend([special_tokens['PAD']] * (seq_len - len(full_tokens)))
    else:
        full_tokens = full_tokens[:seq_len]

    print(f"\n4. Padding (sequence length adjustment)")
    print(f"   Target length: {seq_len}")
    print(f"   Padded sequence: {full_tokens}")

    # Tensor conversion and embedding
    input_ids = torch.tensor([full_tokens], dtype=torch.long, device=DEVICE)
    print(f"\n5. Tensor Conversion")
    print(f"   Input tensor shape: {input_ids.shape} (batch=1, sequence={seq_len})")

    with torch.no_grad():
        # Token embedding
        token_emb = model.token_emb(input_ids)
        print(f"\n6. Token Embedding (word → vector conversion)")
        print(f"   Embedding shape: {token_emb.shape}")
        print(f"   First token vector (first 5 dims): {token_emb[0, 0, :5].cpu().numpy().round(4).tolist()}")

        # Position embedding/encoding
        if model.use_learnable_pos_emb:
            positions = torch.arange(0, seq_len, device=DEVICE).unsqueeze(0)
            pos_emb = model.pos_emb(positions)
            print(f"\n7. Position Embedding (learnable position vectors)")
            print(f"   Position embedding shape: {pos_emb.shape}")
            print(f"   Position[0] vector (first 5 dims): {pos_emb[0, 0, :5].cpu().numpy().round(4).tolist()}")
            print(f"   Position[1] vector (first 5 dims): {pos_emb[0, 1, :5].cpu().numpy().round(4).tolist()}")
            print(f"   Type: Learnable parameters (GPT-style)")
        else:
            pos_emb = model.pos_encoding[:, :seq_len, :]
            print(f"\n7. Positional Encoding (fixed sinusoidal patterns)")
            print(f"   Positional encoding shape: {pos_emb.shape}")
            print(f"   Position[0] vector (first 5 dims): {pos_emb[0, 0, :5].cpu().numpy().round(4).tolist()}")
            print(f"   Position[1] vector (first 5 dims): {pos_emb[0, 1, :5].cpu().numpy().round(4).tolist()}")
            print(f"   Type: Fixed sinusoidal (original Transformer)")
            
        # Show the mathematical difference
        if not model.use_learnable_pos_emb:
            print(f"   Sin/Cos pattern: pos_0 = sin(0/10000^(0/d)), pos_1 = cos(0/10000^(0/d)), ...")

        # Embedding combination
        x = token_emb + pos_emb
        print(f"\n8. Embedding Combination (token + position)")
        print(f"   Combined embedding shape: {x.shape}")
        print(f"   First token final vector (first 5 dims): {x[0, 0, :5].cpu().numpy().round(4).tolist()}")

        # First Transformer layer analysis
        if len(model.layers) > 0:
            first_layer = model.layers[0]

            print(f"\n9. Transformer Layer (Layer 1/{len(model.layers)})")
            print(f"   Config: embed_dim={first_layer.embed_dim}, n_heads={first_layer.n_heads}, head_dim={first_layer.head_dim}")

            # Q, K, V calculation
            q = first_layer.q_proj(x)
            k = first_layer.k_proj(x)
            v = first_layer.v_proj(x)

            print(f"\n10. Q, K, V Projection (weight matrix multiplication)")
            print(f"   Input X shape: {x.shape}")
            print(f"   Weight W_Q shape: {list(first_layer.q_proj.weight.shape)}")
            print(f"   Query (Q = X @ W_Q) shape: {q.shape}")
            print(f"   Key   (K = X @ W_K) shape: {k.shape}")
            print(f"   Value (V = X @ W_V) shape: {v.shape}")
            print(f"   Q[0,0] sample (first 5 dims): {q[0, 0, :5].cpu().numpy().round(4).tolist()}")

            # Multi-head transformation
            batch_size = 1
            n_heads = first_layer.n_heads
            head_dim = first_layer.head_dim

            q_heads = q.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
            k_heads = k.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
            v_heads = v.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)

            print(f"\n11. Multi-head Transformation")
            print(f"   Q multi-head shape: {q_heads.shape} (batch, heads, sequence, head_dim)")
            print(f"   Each head performs independent {head_dim}-dimensional attention")

            # Attention Score calculation
            scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) / math.sqrt(head_dim)
            print(f"\n12. Attention Score Calculation (Q @ K^T / √d_k)")
            print(f"   Score shape: {scores.shape} (batch, heads, sequence, sequence)")
            print(f"   Normalization: √{head_dim} = {math.sqrt(head_dim):.2f}")

            # First head's score matrix sample
            print(f"\n   [Head 1 Attention Score Matrix (4x4 sample)]")
            sample_scores = scores[0, 0, :4, :4].cpu().numpy()
            for i in range(4):
                print(f"   {[f'{val:6.3f}' for val in sample_scores[i]]}")

            # causal_mask application
            causal_mask = torch.triu(torch.ones((seq_len, seq_len), device=DEVICE) * float('-inf'), diagonal=1)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            masked_scores = scores + causal_mask

            print(f"\n13. causal_mask Application (future token blocking)")
            print(f"   Masked scores (4x4 sample, -inf indicates future tokens):")
            sample_masked = masked_scores[0, 0, :4, :4].cpu().numpy()
            for i in range(4):
                row = []
                for val in sample_masked[i]:
                    if val == float('-inf'):
                        row.append('  -∞  ')
                    else:
                        row.append(f'{val:6.3f}')
                print(f"   {row}")

            # Softmax
            attn_weights = F.softmax(masked_scores, dim=-1)
            print(f"\n14. Softmax Normalization (probability distribution)")
            print(f"   Attention weight shape: {attn_weights.shape}")
            print(f"   [Head 1 Attention Weights (4x4, each row sum=1.0)]")
            sample_weights = attn_weights[0, 0, :4, :4].cpu().numpy()
            for i in range(4):
                print(f"   {[f'{val:.4f}' for val in sample_weights[i]]} → sum: {sample_weights[i].sum():.4f}")

            # Attention output
            attn_output = torch.matmul(attn_weights, v_heads)
            print(f"\n15. Attention Output (Attention @ V)")
            print(f"   Attention output shape: {attn_output.shape}")

            # Head combination
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, first_layer.embed_dim)
            attn_output = first_layer.out_proj(attn_output)
            print(f"   Head combination shape: {attn_output.shape}")

            # Add & Norm
            x_norm1 = first_layer.ln1(x + first_layer.dropout(attn_output))
            print(f"\n16. Add & LayerNorm (residual connection)")
            print(f"   Residual connection: X + Attention(X)")
            print(f"   Input range: [{x.min().item():.3f}, {x.max().item():.3f}] → Normalized: [{x_norm1.min().item():.3f}, {x_norm1.max().item():.3f}]")

            # FFN
            ff_output = first_layer.ff(x_norm1)
            print(f"\n17. Feed-Forward Network (FFN)")
            print(f"   FFN structure: Linear({first_layer.embed_dim} → {first_layer.ff[0].out_features}) → GELU → Linear({first_layer.ff[3].in_features} → {first_layer.embed_dim})")
            print(f"   FFN output shape: {ff_output.shape}")

            # Final Add & Norm
            x_final = first_layer.ln2(x_norm1 + first_layer.dropout(ff_output))
            print(f"\n18. Final Add & LayerNorm")
            print(f"   Layer 1 final output shape: {x_final.shape}")
            print(f"   Information flow: Input → Attention → FFN → Output")

        # Full model output
        print(f"\n19. Final Model Output (through all layers)")
        output = model(input_ids)
        print(f"   Logit shape: {output.shape} (batch, sequence, vocab_size)")

        # Last position prediction
        last_logits = output[0, -1, :]
        print(f"\n20. Next Token Prediction (last position)")
        print(f"   Vocabulary size: {len(last_logits)}, Token distribution generated")

        # Softmax probabilities
        probs = F.softmax(last_logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, 5)
        top_logits, _ = torch.topk(last_logits, 5)

        print(f"\n   Top 5 Token Predictions (Softmax probabilities):")
        for i, (prob, idx, logit) in enumerate(zip(top_probs, top_indices, top_logits)):
            try:
                token_text = tokenizer_manager.tokenizer.convert_ids_to_tokens([idx.item()])[0]
                print(f"   {i+1}. Token {idx.item():5d} ('{token_text:10s}'): logit={logit.item():7.4f}, prob={prob.item()*100:6.3f}%")
            except:
                print(f"   {i+1}. Token {idx.item():5d}: logit={logit.item():7.4f}, prob={prob.item()*100:6.3f}%")

    print("\n" + "="*100  + "\n")

# Multi-turn dataset tokenization class
class MultiTurnDataset(Dataset):
    def __init__(self, tokenizer_manager, conversations, seq_len, max_conversations=None):
        start_time = time.time()
        self.tokenizer_manager = tokenizer_manager
        self.seq_len = seq_len
        self.conversations = conversations[:max_conversations] if max_conversations else conversations
        self.special_tokens = tokenizer_manager.get_special_tokens()

        print(f"> Conversation count: {len(self.conversations):,}")
        print(f"> Vocabulary size: {tokenizer_manager.vocab_size:,}")

        self.token_sequences = []
        tokenization_start = log_time(start_time, "Dataset initialization")

        # GPU mode-specific processing
        if GPU_MODE == "A100":
            print("> A100 mode: Tokenization in progress...")

            all_tokens = []
            for conversation in tqdm(self.conversations, desc="Tokenization"):
                conv_tokens = [self.special_tokens['BOS']]

                for i, turn in enumerate(conversation):
                    speaker_token = self.special_tokens['SPEAKER_A'] if i % 2 == 0 else self.special_tokens['SPEAKER_B']
                    conv_tokens.append(speaker_token)
                    text_tokens = self.tokenizer_manager.encode(turn['text'])
                    conv_tokens.extend(text_tokens)

                    if i < len(conversation) - 1:
                        conv_tokens.append(self.special_tokens['TURN_SEP'])

                conv_tokens.append(self.special_tokens['EOS'])
                all_tokens.extend(conv_tokens)

            # Generate overlapping sequences
            step_size = seq_len // 4
            for i in range(0, len(all_tokens) - seq_len + 1, step_size):
                chunk = all_tokens[i:i + seq_len]
                if len(chunk) == seq_len:
                    self.token_sequences.append(chunk)
        else:
            # T4/CPU: Tokenization
            print("> T4/CPU tokenization in progress...")

            for conversation in tqdm(self.conversations, desc="Tokenization progress"):
                conv_tokens = [self.special_tokens['BOS']]

                for i, turn in enumerate(conversation):
                    speaker_token = self.special_tokens['SPEAKER_A'] if i % 2 == 0 else self.special_tokens['SPEAKER_B']
                    conv_tokens.append(speaker_token)
                    text = turn['text'][:200] if GPU_MODE == "T4" else turn['text'][:100]
                    text_tokens = self.tokenizer_manager.encode(text)
                    if GPU_MODE == "T4":
                        text_tokens = text_tokens[:50]
                    else:
                        text_tokens = text_tokens[:30]
                    conv_tokens.extend(text_tokens)

                    if i < len(conversation) - 1:
                        conv_tokens.append(self.special_tokens['TURN_SEP'])

                conv_tokens.append(self.special_tokens['EOS'])

                # Padding or truncation
                if len(conv_tokens) < seq_len:
                    conv_tokens.extend([self.special_tokens['PAD']] * (seq_len - len(conv_tokens)))
                else:
                    conv_tokens = conv_tokens[:seq_len]

                self.token_sequences.append(conv_tokens)

                if GPU_MODE == "CPU" and len(self.token_sequences) >= 500:
                    break

        print(f"> Generated sequences: {len(self.token_sequences):,}")
        log_time(tokenization_start, "Tokenization complete")

        # Sample analysis
        if self.conversations:
            conversation = self.conversations[0]
            print(f"\n[Sample Conversation] {min(len(conversation), 3)} turns")
            for i, turn in enumerate(conversation[:3]):  # Show all 3 turns
                text = turn['text'][:40] + ('...' if len(turn['text']) > 40 else '')
                print(f"  Turn{i+1} ({turn['speaker']}): {text}")
            print()

    def __len__(self):
        return len(self.token_sequences)

    def __getitem__(self, idx):
        sequence = self.token_sequences[idx]
        return torch.tensor(sequence, dtype=torch.long)

# Transformer decoder implementation
class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim)
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask=None):
        # Self-attention
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        batch_size, seq_len = q.size(0), q.size(1)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if causal_mask is not None:
            scores = scores + causal_mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        x = self.ln1(x + self.dropout(attn_output))

        # Feed forward
        ff_output = self.ff(x)
        x = self.ln2(x + self.dropout(ff_output))

        return x, attn_weights

# Fixed positional encoding function (original Transformer)
def create_sinusoidal_positional_encoding(seq_len, embed_dim):
    """Create fixed sinusoidal positional encoding as in original Transformer paper"""
    pe = torch.zeros(seq_len, embed_dim)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe.unsqueeze(0)  # Add batch dimension

# Transformer decoder class with position encoding option
# use_learnable_pos_emb = True: GPT-style (learnable Position Embedding), False: Original Transformer style (fixed Positional Encoding)
class SNSDecoderLM(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_dim, n_layers, n_heads, ffn_dim, 
                 dropout=0.1, use_learnable_pos_emb=True):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.use_learnable_pos_emb = use_learnable_pos_emb
        
        # Position encoding/embedding selection
        if use_learnable_pos_emb:
            # Learnable position embedding (GPT-style)
            self.pos_emb = nn.Embedding(seq_len, embed_dim)
            print(f"> Using learnable Position Embedding (GPT-style)")
        else:
            # Fixed sinusoidal positional encoding (original Transformer)
            self.register_buffer('pos_encoding', create_sinusoidal_positional_encoding(seq_len, embed_dim))
            print(f"> Using fixed Positional Encoding (original Transformer)")

        self.layers = nn.ModuleList([
            DecoderBlock(embed_dim, n_heads, ffn_dim, dropout) for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        b, t = idx.size()
        assert t <= self.seq_len, f"Sequence length {t} exceeds maximum {self.seq_len}"

        tok_emb = self.token_emb(idx)
        
        if self.use_learnable_pos_emb:
            # Learnable position embedding
            positions = torch.arange(0, t, device=idx.device).unsqueeze(0).expand(b, t)
            pos_emb = self.pos_emb(positions)
        else:
            # Fixed positional encoding
            pos_emb = self.pos_encoding[:, :t, :].expand(b, t, self.embed_dim)
        
        x = tok_emb + pos_emb

        causal_mask = torch.triu(torch.ones((t, t), device=idx.device) * float('-inf'), diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        for layer in self.layers:
            x, _ = layer(x, causal_mask)

        x = self.ln_f(x)
        logits = self.head(x)

        return logits

# CrossEntropyLoss calculation
def multiturn_loss(logits, targets, special_tokens):
    b, t, v = logits.size()
    logits = logits[:, :-1, :].contiguous()
    targets = targets[:, 1:].contiguous()

    loss_fn = nn.CrossEntropyLoss(ignore_index=special_tokens['PAD'])
    loss = loss_fn(logits.view(-1, v), targets.view(-1))

    return loss

# Multi-turn conversation generation 
# returns top-K candidates with probabilities

@torch.no_grad()
def generate_multiturn_top_k(model, tokenizer_manager, prompt, max_new_tokens=40, temperature=0.8, top_k=30, num_candidates=3, device=DEVICE):    
    model.eval()
    special_tokens = tokenizer_manager.get_special_tokens()

    responses_with_probs = []

    for candidate in range(num_candidates):
        # Temperature variation for diversity
        temp = temperature + (candidate * 0.15)

        prompt_tokens = tokenizer_manager.encode(prompt)
        tokens = [special_tokens['BOS'], special_tokens['SPEAKER_A']] + prompt_tokens + [special_tokens['TURN_SEP'], special_tokens['SPEAKER_B']]

        # Accumulate probability for each token
        token_probs = []

        for step in range(max_new_tokens):
            # Context window management
            context = tokens[-SEQ_LEN:]
            input_ids = torch.tensor([context], dtype=torch.long, device=device)

            # Model forward pass
            logits = model(input_ids)
            last_logits = logits[0, -1, :] / temp

            # Top-K filtering
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(last_logits, min(top_k, last_logits.size(-1)))
                last_logits = torch.full_like(last_logits, float('-inf'))
                last_logits.scatter_(0, top_k_indices, top_k_logits)

            # Convert to probabilities
            probs = F.softmax(last_logits, dim=-1)
            probs[special_tokens['UNK']] = 0.0
            probs[special_tokens['PAD']] = 0.0
            probs = probs / probs.sum()

            # Sample next token
            next_token = torch.multinomial(probs, 1).item()
            token_prob = probs[next_token].item()
            token_probs.append(token_prob)
            tokens.append(next_token)

            # Stop if EOS token generated
            if next_token == special_tokens['EOS']:
                break

        # Decode response tokens
        response_tokens = []
        speaker_b_started = False

        for token in tokens:
            if token == special_tokens['SPEAKER_B']:
                speaker_b_started = True
                continue
            elif token in [special_tokens['TURN_SEP'], special_tokens['EOS'], special_tokens['BOS']]:
                continue
            elif speaker_b_started and token not in [special_tokens['PAD'], special_tokens['UNK']]:
                response_tokens.append(token)

        response = tokenizer_manager.decode(response_tokens).strip()
        if response and len(response) > 0:
            # Calculate average probability
            avg_prob = sum(token_probs) / len(token_probs) if token_probs else 0
            responses_with_probs.append((response, avg_prob))

    # Sort by probability
    responses_with_probs.sort(key=lambda x: x[1], reverse=True)

    # Remove duplicates
    unique_responses = []
    seen_texts = set()
    for resp, prob in responses_with_probs:
        if resp not in seen_texts:
            unique_responses.append((resp, prob))
            seen_texts.add(resp)

    # Fallback responses
    if not unique_responses:
        unique_responses = [
            ("안녕하세요!", 0.33),
            ("네, 반가워요!", 0.33),
            ("좋은 하루네요!", 0.34)
        ]

    return unique_responses[:3]

# Enhanced model generation test (with probability values)
def test_model_generation_improved(model, tokenizer_manager):    
    print(f"\n" + "="*80)
    print("■ Enhanced Model Generation Test (3 topics, 3 candidates each + probabilities)")
    print("="*80)

    test_prompts = [
        "안녕하세요! 오늘 어떤 하루 보내셨나요?",
        "요즘 재미있게 본 영화가 있나요?",
        "취미가 뭐예요?"
    ]

    model.eval()

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[Topic {i}/3]")
        print(f"> Prompt: {prompt}")

        try:
            responses_with_probs = generate_multiturn_top_k(
                model, tokenizer_manager, prompt,
                max_new_tokens=35, temperature=0.7, top_k=40, num_candidates=3
            )

            print("> LLM Responses (sorted by probability):")
            for j, (response, prob) in enumerate(responses_with_probs[:3], 1):
                print(f"‧ {j}. [{prob*100:.2f}%] {response}")

        except Exception as e:
            print(f"X Generation error: {e}")

# Interactive multi-turn test (with probabilities)
def interactive_multiturn_test(model, tokenizer_manager, max_turns=10):    
    print(f"\n" + "="*80)
    print(f"■ Interactive Multi-turn Test (max {max_turns} turns)")
    print("‧ To exit, type 'quit', 'exit', or '종료'")
    print("="*80)

    model.eval()
    turn_count = 0
    conversation_history = []

    while turn_count < max_turns:
        turn_count += 1
        print(f"\n> [Turn {turn_count}/{max_turns}]")

        try:
            user_input = input("> User: ").strip()

            if user_input.lower() in ['quit', 'exit', '종료', 'q']:
                print("> Ending conversation.")
                break

            if not user_input:
                print("> Empty input. Please try again.")
                turn_count -= 1
                continue

            conversation_history.append(f"User: {user_input}")

            responses_with_probs = generate_multiturn_top_k(
                model, tokenizer_manager, user_input,
                max_new_tokens=40, temperature=0.8, top_k=35, num_candidates=3
            )

            print("> LLM Responses (sorted by probability):")
            for i, (response, prob) in enumerate(responses_with_probs[:3], 1):
                print(f"‧ {i}. [{prob*100:.2f}%] {response}")

            if responses_with_probs:
                conversation_history.append(f"AI: {responses_with_probs[0][0]}")

        except KeyboardInterrupt:
            print("\n\n> Conversation interrupted.")
            break
        except Exception as e:
            print(f"X Error occurred: {e}")

    print(f"\n> Total {turn_count} turns of conversation completed.")

    if conversation_history:
        print(f"\n■ Conversation Summary:")
        for line in conversation_history[-6:]:
            print(f"‧ {line}")

    print("="*80)

def main():
    print("="*80)
    print("■ SNS Multi-turn Conversation LLM Training")
    print("="*80)

    tokenizer_type = "KLUE/BERT Tokenizer"
    print(f"‧ Tokenizer: {tokenizer_type}")
    print(f"‧ Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}, Sequence length: {SEQ_LEN}")
    print(f"‧ Max # of data files: {MAX_FILES_FOR_FULL_DATASET:,}")
    print("="*80)

    total_start_time = time.time()

    # Data loading
    step_start = time.time()
    sns_loader = FastSNSDataLoader(LABELED_DATA_PATH)
    log_time(step_start, "Data loader initialization")

    step_start = time.time()

    # Attempt to load real data or generate dummy data
    try:
        conversations = sns_loader.load_json_conversations()
        if not conversations or len(conversations) < 10:
            raise ValueError("Insufficient conversation data available.")
    except Exception as e:
        print(f"> SNS data loading failed: {e}")
        print("> Generating dummy data...")

        # Generate dummy data
        conversations = []
        sample_dialogues = [
            [
                {"speaker": "A", "text": "안녕하세요! 오늘 날씨가 참 좋네요."},
                {"speaker": "B", "text": "네, 정말 좋아요. 산책하기 딱 좋은 날씨예요."},
                {"speaker": "A", "text": "맞아요. 공원에 가려고 하는데 같이 가실래요?"},
                {"speaker": "B", "text": "좋아요! 언제 출발할까요?"}
            ],
            [
                {"speaker": "A", "text": "요즘 뭐하고 지내세요?"},
                {"speaker": "B", "text": "새로운 프로젝트 준비 중이에요. 많이 바쁘네요."},
                {"speaker": "A", "text": "어떤 프로젝트인가요?"},
                {"speaker": "B", "text": "AI 관련 프로젝트예요. 정말 흥미로워요!"}
            ]
        ]

        # Generate dummy conversations (repeated)
        for _ in range(500):
            conversations.append(random.choice(sample_dialogues))

    load_time = log_time(step_start, f"Data loading complete ({len(conversations):,} conversations)")

    # Tokenizer setup
    print("\n‧ Tokenizer setup started...")
    step_start = time.time()
    tokenizer_manager = TokenizerManager(use_custom=USE_CUSTOM_TOKENIZER, pretrained_model=PRETRAINED_TOKENIZER)
    tokenizer = tokenizer_manager.setup_tokenizer()
    tokenizer_time = log_time(step_start, "Tokenizer setup complete")

    # Dataset creation
    print("\n‧ Dataset creation started...")
    step_start = time.time()

    max_conversations = min(MAX_CONVERSATIONS, len(conversations))
    print(f"{GPU_MODE} mode: Using {max_conversations:,} out of {len(conversations):,} conversations")

    dataset = MultiTurnDataset(tokenizer_manager, conversations, SEQ_LEN, max_conversations)

    if len(dataset) == 0:
        print("X Dataset is empty.")
        return

    dataset_time = log_time(step_start, f"Dataset creation complete ({len(dataset):,} sequences)")

    actual_batch_size = min(BATCH_SIZE, len(dataset))
    if actual_batch_size < BATCH_SIZE:
        print(f"> Adjusting batch size from {BATCH_SIZE} to {actual_batch_size}")

    # DataLoader setup - resolve num_workers issue
    dataloader = DataLoader(
        dataset,
        batch_size=actual_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,  # Disable multiprocessing to prevent deadlock
        pin_memory=False  # Also disable pin_memory
    )

    total_steps = max(EPOCHS * len(dataloader), 10)
    if len(dataloader) == 0:
        print("> DataLoader is empty.")
        return

    # Model initialization
    print("\n‧ Model initialization started...")
    step_start = time.time()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = SNSDecoderLM(
        vocab_size=tokenizer_manager.vocab_size,
        seq_len=SEQ_LEN,
        embed_dim=EMBED_DIM,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        ffn_dim=FFN_DIM,
        dropout=0.1,
        use_learnable_pos_emb=True  # Add position embedding option
    ).to(DEVICE)

    # Optimizer setup (AMP disabled)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        eps=1e-8
    )

    # AMP scaler disabled
    scaler = None  # Completely disable AMP

    warmup_steps = max(int(total_steps * WARMUP_RATIO), 1)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        total_steps=total_steps,
        pct_start=WARMUP_RATIO,
        div_factor=25,
        final_div_factor=10000,
        anneal_strategy='cos'
    )

    print(f"Parameter count: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Batch count: {len(dataloader)}")
    print(f"Total steps: {total_steps}")

    model_init_time = log_time(step_start, "Model initialization complete")

    # Transformer process analysis
    if ENABLE_TRANSFORMER_DEMO:
        print("\n‧ Running Transformer analysis demo...")
        try:
            detailed_transformer_demo(model, tokenizer_manager, "안녕하세요! 오늘 어떤 하루 보내셨나요?")
        except Exception as e:
            print(f"X Transformer demo error: {e}")

    # Model training
    print(f"\n{'='*80}")
    print("■ Model Training Started")
    print("="*80)

    training_start_time = time.time()
    model.train()
    global_step = 0
    special_tokens = tokenizer_manager.get_special_tokens()

    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")

        epoch_loss = 0
        batch_count = 0

        # Debugging information
        print(f"‧ DataLoader batch count: {len(dataloader)}")
        print(f"‧ Batch size: {actual_batch_size}")
        print(f"‧ Dataset size: {len(dataset)}")

        pbar = tqdm(dataloader, desc=f"‧ Epoch {epoch+1}")

        for batch_idx, batch in enumerate(pbar):
            try:
                batch = batch.to(DEVICE, non_blocking=False)  # Changed to non_blocking=False

                # Simplified without AMP
                logits = model(batch)
                loss = multiturn_loss(logits, batch, special_tokens)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                scheduler.step()
                epoch_loss += loss.item()
                batch_count += 1
                global_step += 1

                # Increased memory cleanup interval
                if global_step % 50 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Progress update
                gpu_memory = f'{torch.cuda.memory_allocated()/1e9:.1f}GB' if torch.cuda.is_available() else 'CPU'
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                    'mem': gpu_memory,
                    'batch': f'{batch_idx+1}/{len(dataloader)}'
                })

                # Debug first few batches
                if batch_idx < 3:
                    print(f"\n‧ Batch {batch_idx+1} processing complete - Loss: {loss.item():.4f}")

            except Exception as e:
                print(f"\n‧ X Error processing batch {batch_idx}: {e}")
                print(f"Batch shape: {batch.shape if 'batch' in locals() else 'N/A'}")
                continue

        if batch_count > 0:
            avg_epoch_loss = epoch_loss / batch_count
            log_time(epoch_start_time, f"Epoch {epoch+1} complete (avg loss: {avg_epoch_loss:.4f})")
        else:
            print(f"> No batches processed in Epoch {epoch+1}.")

    training_time = log_time(training_start_time, "Complete training finished")

    # Final model saving
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'tokenizer_config': {
            'use_custom': USE_CUSTOM_TOKENIZER,
            'pretrained_model': PRETRAINED_TOKENIZER,
            'vocab_size': tokenizer_manager.vocab_size
        },
        'config': {
            'vocab_size': tokenizer_manager.vocab_size,
            'seq_len': SEQ_LEN,
            'embed_dim': EMBED_DIM,
            'n_layers': N_LAYERS,
            'n_heads': N_HEADS,
            'ffn_dim': FFN_DIM,
            'use_learnable_pos_emb': True
        }
    }

    model_path = f"final_sns_model_{GPU_MODE.lower()}.pt"
    torch.save(final_checkpoint, model_path)
    print(f"> Final model saved: {model_path}")

    # Model testing
    test_model_generation_improved(model, tokenizer_manager)

    # Interactive testing
    try:
        interactive_multiturn_test(model, tokenizer_manager, 10)
    except Exception as e:
        print(f"X Interactive test error: {e}")

    # Execution time summary
    print(f"\n{'='*80}")
    print("■ Execution Time Summary")
    print("="*80)
    total_time = time.time() - total_start_time
    print(f"Total execution time: {total_time/60:.1f} minutes")

    print("\n> Training completed!!")

if __name__ == "__main__":
    if hasattr(torch.multiprocessing, 'set_start_method'):
        try:
            torch.multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

    main()
