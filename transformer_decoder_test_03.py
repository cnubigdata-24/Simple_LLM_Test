!apt-get -qq install fonts-nanum
!fc-cache -fv
!rm -rf ~/.cache/matplotlib

import matplotlib.pyplot as plt
plt.rc('font', family='NanumGothic')

!{sys.executable} -m pip install seaborn

# Transformer 언어 생성 모델 - GPT 스타일 Decoder-Only

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import math
import re
import warnings
warnings.filterwarnings('ignore')

# CUDA 텐서를 numpy로 변환
def safe_numpy(tensor):
    if tensor.is_cuda:
        return tensor.cpu().detach().numpy()
    else:
        return tensor.detach().numpy()

# 재현성을 위한 시드 설정
def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

set_seed(42)

print("=" * 100)
print("TRANSFORMER 언어 생성 모델")
print("=" * 100)

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n STEP 1: 환경 설정")
print("-" * 80)

# GPU 환경 확인
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"   사용 디바이스: {device}")
    print(f"   CUDA 사용 가능: True")
    print(f"   GPU 개수: {torch.cuda.device_count()}")
    print(f"   현재 GPU: {torch.cuda.get_device_name(0)}")
    print(f"   GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"   CUDA 버전: {torch.version.cuda}")
else:
    device = torch.device('cpu')
    print(f"   사용 디바이스: {device}")
    print(f"   CUDA 사용 가능: False")
    print(f"   CPU로 실행합니다")
    print(f"   주의: CPU에서는 학습이 상당히 느릴 수 있습니다")

print(f"   PyTorch 버전: {torch.__version__}")

# 하이퍼파라미터 설정
config = {
    'vocab_size': 5000,
    'd_model': 128,
    'n_heads': 8,
    'n_layers': 4,
    'd_ff': 512,
    'max_seq_len': 64,
    'dropout': 0.1,
    'learning_rate': 0.0001,
    'batch_size': 4,
    'epochs': 30
}

# CPU 환경에서는 더 빠른 학습을 위해 설정 조정
if device.type == 'cpu':
    print(f"\n  CPU 환경 감지 - 학습 속도 최적화")
    print("-" * 80)
    config['epochs'] = 10  # 에폭 수 감소
    config['d_model'] = 64  # 모델 크기 축소
    config['d_ff'] = 256   # 피드포워드 크기 축소
    print(f"   CPU 최적화: 에폭 {config['epochs']}, 모델차원 {config['d_model']}")
    print(f"   예상 학습 시간: 약 3-5분")
else:
    print(f"   예상 학습 시간: 약 1-2분")

print(f"\n 최종 하이퍼파라미터:")
for key, value in config.items():
    print(f"   {key}: {value}")

# 샘플 한국어 데이터
sample_texts = [
    "안녕하세요 좋은 하루 되세요",
    "오늘 날씨가 정말 좋네요",
    "파이썬 프로그래밍을 배우고 있습니다",
    "인공지능 기술이 빠르게 발전하고 있습니다",
    "컴퓨터 과학은 매우 흥미로운 분야입니다",
    "머신러닝 알고리즘을 공부하고 있어요",
    "딥러닝 모델을 구현해보고 싶습니다",
    "자연어 처리 기술이 놀랍습니다",
    "데이터 사이언스는 미래의 핵심 기술입니다",
    "코딩을 통해 문제를 해결하는 것이 재미있어요"
]

print(f"\n STEP 2: 학습 데이터 준비")
print("-" * 80)
print(f"   총 샘플 수: {len(sample_texts)}")
print(f"   샘플 데이터 예시:")
for i, text in enumerate(sample_texts[:3]):
    print(f"     {i+1}. {text}")
print(f"     ... (총 {len(sample_texts)}개)")

# 고급 토크나이저 클래스
class AdvancedTokenizer:
    def __init__(self, texts, min_freq=1):
        print(f"\n 토크나이저 생성 중...")
        
        # 모든 텍스트에서 단어 추출
        all_words = []
        for text in texts:
            # 한국어와 영어, 숫자를 모두 포함하도록 정규식 개선
            words = re.findall(r'\w+', text)
            all_words.extend(words)
        
        # 단어 빈도 계산
        word_counts = Counter(all_words)
        
        # 최소 빈도 이상의 단어만 어휘에 포함
        filtered_words = [word for word, count in word_counts.items() if count >= min_freq]
        
        # 특수 토큰 정의
        special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        
        # 최종 어휘 구성
        self.vocab = special_tokens + sorted(filtered_words)
        self.word_to_id = {word: i for i, word in enumerate(self.vocab)}
        self.id_to_word = {i: word for i, word in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        
        # 특수 토큰 ID
        self.pad_id = self.word_to_id['<PAD>']
        self.unk_id = self.word_to_id['<UNK>']
        self.bos_id = self.word_to_id['<BOS>']
        self.eos_id = self.word_to_id['<EOS>']
        
        print(f"   전체 단어 수: {len(all_words)}")
        print(f"   고유 단어 수: {len(word_counts)}")
        print(f"   필터링 후 어휘 크기: {self.vocab_size}")
        print(f"   가장 빈번한 단어 5개: {dict(word_counts.most_common(5))}")
        print(f"   특수 토큰: {special_tokens}")
        print(f"   어휘 예시: {self.vocab[:10]}")
    
    def encode(self, text, add_special_tokens=True, show_process=False):
        words = re.findall(r'\w+', text)
        token_ids = [self.word_to_id.get(word, self.unk_id) for word in words]
        
        if add_special_tokens:
            token_ids = [self.bos_id] + token_ids + [self.eos_id]
        
        if show_process:
            print(f"     인코딩: '{text}'")
            print(f"     단어 분할: {words}")
            print(f"     토큰 ID: {token_ids}")
            decoded_words = [self.id_to_word[id] for id in token_ids]
            print(f"     검증: {decoded_words}")
        
        return token_ids
    
    def decode(self, token_ids, remove_special_tokens=True):
        if remove_special_tokens:
            # 특수 토큰 제거
            token_ids = [id for id in token_ids if id not in [self.pad_id, self.bos_id, self.eos_id]]
        
        words = [self.id_to_word.get(id, '<UNK>') for id in token_ids]
        return ' '.join(words)
    
    def batch_encode(self, texts, max_length=None, padding=True):
        # 배치 인코딩
        encoded_texts = [self.encode(text) for text in texts]
        
        if max_length is None:
            max_length = max(len(seq) for seq in encoded_texts)
        
        if padding:
            padded_texts = []
            for seq in encoded_texts:
                if len(seq) > max_length:
                    seq = seq[:max_length]
                else:
                    seq = seq + [self.pad_id] * (max_length - len(seq))
                padded_texts.append(seq)
            return padded_texts
        
        return encoded_texts

# 위치 인코딩 클래스 (sin/cos 기반)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # 위치 인코딩 행렬 생성
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # div_term 계산: 1 / (10000^(2i/d_model))
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # sin과 cos를 번갈아 적용
        pe[:, 0::2] = torch.sin(position * div_term)  # 짝수 인덱스
        pe[:, 1::2] = torch.cos(position * div_term)  # 홀수 인덱스
        
        pe = pe.unsqueeze(0)  # 배치 차원 추가
        self.register_buffer('pe', pe)
        
        print(f"\n 위치 인코딩 초기화:")
        print(f"   최대 길이: {max_len}")
        print(f"   모델 차원: {d_model}")
        print(f"   위치 인코딩 행렬 모양: {pe.shape}")
        print(f"   div_term 샘플 (처음 5개): {div_term[:5].numpy()}")
    
    def forward(self, x, show_process=False):
        seq_len = x.size(1)
        pos_encoding = self.pe[:, :seq_len]
        
        if show_process:
            print(f"\n   위치 인코딩 적용:")
            print(f"     입력 모양: {x.shape}")
            print(f"     위치 인코딩 모양: {pos_encoding.shape}")
            print(f"     위치 0의 인코딩 (처음 8차원): {safe_numpy(pos_encoding[0, 0, :8])}")
            print(f"     위치 1의 인코딩 (처음 8차원): {safe_numpy(pos_encoding[0, 1, :8])}")
        
        return x + pos_encoding

# 마스킹 유틸리티 함수들
def create_padding_mask(seq, pad_id=0):
    # 패딩 토큰 위치를 False로 마킹
    return (seq != pad_id).unsqueeze(1).unsqueeze(2)

def create_causal_mask(seq_len):
    # 하삼각 마스크 생성 (미래 토큰을 보지 못하게 함)
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask == 0  # True는 허용, False는 마스킹

def demonstrate_masking(tokenizer):
    print(f"\nSTEP 3: 마스킹 메커니즘 이해")
    print("-" * 80)
    
    # 샘플 시퀀스 생성
    sample_text = "안녕하세요 좋은 하루"
    tokens = tokenizer.encode(sample_text)
    
    # 패딩 추가
    max_len = 8
    if len(tokens) < max_len:
        tokens.extend([tokenizer.pad_id] * (max_len - len(tokens)))
    else:
        tokens = tokens[:max_len]
    
    token_tensor = torch.tensor([tokens])
    
    print(f"   샘플 텍스트: '{sample_text}'")
    print(f"   토큰 ID: {tokens}")
    token_words = [tokenizer.id_to_word[id] for id in tokens]
    print(f"   토큰 단어: {token_words}")
    
    # 1. 패딩 마스크
    padding_mask = create_padding_mask(token_tensor, tokenizer.pad_id)
    print(f"\n   패딩 마스크 (True=유효한 토큰, False=패딩):")
    print(f"   마스크 모양: {padding_mask.shape}")
    mask_1d = padding_mask[0, 0, 0].numpy()
    for i, (word, mask_val) in enumerate(zip(token_words, mask_1d)):
        status = "✓유효" if mask_val else "✗패딩"
        print(f"     위치 {i}: '{word}' -> {status}")
    
    # 2. 인과적 마스크 (Causal Mask)
    causal_mask = create_causal_mask(len(tokens))
    print(f"\n   인과적 마스크 (True=참조가능, False=미래토큰):")
    print(f"   마스크 모양: {causal_mask.shape}")
    print(f"   각 토큰이 참조할 수 있는 토큰들:")
    
    for i, word in enumerate(token_words):
        allowed_positions = [j for j, allowed in enumerate(causal_mask[i]) if allowed]
        allowed_words = [token_words[j] for j in allowed_positions if j < len(token_words)]
        print(f"     '{word}' -> {allowed_words}")
    
    # 3. 결합된 마스크
    combined_mask = padding_mask & causal_mask.unsqueeze(0).unsqueeze(0)
    print(f"\n   결합된 마스크 (패딩 + 인과적):")
    print(f"   최종 어텐션에서 사용되는 마스크")
    print(f"   마스크 행렬 시각화:")
    
    final_mask = combined_mask[0, 0].numpy()
    print(f"{'':>12}", end="")
    for word in token_words:
        print(f"{word[:6]:>8}", end="")
    print()
    
    for i, query_word in enumerate(token_words):
        print(f"{query_word[:10]:>12}", end="")
        for j in range(len(token_words)):
            symbol = "✓" if final_mask[i, j] else "✗"
            print(f"{symbol:>8}", end="")
        print()
    
    return token_tensor, combined_mask

# 스케일드 닷-프로덕트 어텐션
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout=0.1):
        super().__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, Q, K, V, mask=None, show_process=False):
        batch_size, n_heads, seq_len, d_k = Q.shape
        
        if show_process:
            print(f"\n   스케일드 닷-프로덕트 어텐션 계산:")
            print(f"     Q 모양: {Q.shape}")
            print(f"     K 모양: {K.shape}")  
            print(f"     V 모양: {V.shape}")
            print(f"     스케일링 팩터: 1/√{d_k} = {1/math.sqrt(d_k):.4f}")
        
        # 1. 어텐션 스코어 계산: Q @ K^T / √d_k
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        if show_process:
            print(f"     어텐션 스코어 모양: {scores.shape}")
            # 첫 번째 헤드의 첫 번째 샘플 스코어 출력
            sample_scores = safe_numpy(scores[0, 0, :3, :3])
            print(f"     스코어 예시 (첫 헤드, 3x3):")
            for i in range(3):
                print(f"       {sample_scores[i]}")
        
        # 2. 마스크 적용
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            if show_process:
                print(f"     마스크 적용 완료 (False 위치를 -1e9로 설정)")
        
        # 3. 소프트맥스로 확률 분포 변환
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        if show_process:
            print(f"     소프트맥스 후 어텐션 가중치:")
            sample_weights = safe_numpy(attention_weights[0, 0, :3, :3])
            print(f"     가중치 예시 (첫 헤드, 3x3):")
            for i in range(3):
                row_sum = sample_weights[i].sum()
                print(f"       {sample_weights[i]} (합계: {row_sum:.3f})")
        
        # 4. Value와 가중합
        context = torch.matmul(attention_weights, V)
        
        if show_process:
            print(f"     최종 컨텍스트 벡터 모양: {context.shape}")
        
        return context, attention_weights

# 멀티헤드 어텐션
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # 선형 변환 레이어들
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(self.d_k, dropout)
        
        print(f"\n 멀티헤드 어텐션 초기화:")
        print(f"   모델 차원: {d_model}")
        print(f"   헤드 수: {n_heads}")
        print(f"   헤드당 차원: {self.d_k}")
        print(f"   총 파라미터 수: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, query, key, value, mask=None, show_process=False):
        batch_size, seq_len, d_model = query.shape
        
        if show_process:
            print(f"\n   멀티헤드 어텐션 forward:")
            print(f"     입력 모양: {query.shape}")
        
        # 1. Q, K, V 변환
        Q = self.W_q(query)
        K = self.W_k(key)  
        V = self.W_v(value)
        
        if show_process:
            print(f"     Q 변환 후: {Q.shape}")
            print(f"     K 변환 후: {K.shape}")
            print(f"     V 변환 후: {V.shape}")
        
        # 2. 멀티헤드로 분할 및 reshape
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        if show_process:
            print(f"     멀티헤드 분할 후 Q: {Q.shape}")
            print(f"     [배치, 헤드, 시퀀스, 헤드차원]")
        
        # 3. 어텐션 계산
        context, attention_weights = self.attention(Q, K, V, mask, show_process)
        
        # 4. 헤드들을 다시 연결
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # 5. 최종 선형 변환
        output = self.W_o(context)
        
        if show_process:
            print(f"     헤드 연결 후: {context.shape}")
            print(f"     최종 출력: {output.shape}")
        
        return output, attention_weights

# 포지션-와이즈 피드포워드
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
        print(f"\n 피드포워드 네트워크 초기화:")
        print(f"   입력 차원: {d_model}")
        print(f"   은닉 차원: {d_ff}")
        print(f"   출력 차원: {d_model}")
        print(f"   활성화 함수: ReLU")
        print(f"   파라미터 수: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, x, show_process=False):
        if show_process:
            print(f"\n   피드포워드 처리:")
            print(f"     입력 모양: {x.shape}")
            print(f"     입력 값 범위: [{x.min().item():.3f}, {x.max().item():.3f}]")
        
        # 첫 번째 선형 변환 + ReLU
        hidden = self.activation(self.linear1(x))
        
        if show_process:
            print(f"     은닉층 모양: {hidden.shape}")
            print(f"     은닉층 값 범위: [{hidden.min().item():.3f}, {hidden.max().item():.3f}]")
            print(f"     ReLU 후 0 값 비율: {(hidden == 0).float().mean().item():.3f}")
        
        # 드롭아웃
        hidden = self.dropout(hidden)
        
        # 두 번째 선형 변환
        output = self.linear2(hidden)
        
        if show_process:
            print(f"     최종 출력 모양: {output.shape}")
            print(f"     출력 값 범위: [{output.min().item():.3f}, {output.max().item():.3f}]")
        
        return output

# 트랜스포머 디코더 레이어
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        print(f"\n 트랜스포머 디코더 레이어 초기화:")
        print(f"   레이어 구성: Self-Attention + FFN")
        print(f"   잔차 연결 + 레이어 정규화 적용")
        print(f"   총 파라미터: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, x, mask=None, show_process=False):
        if show_process:
            print(f"\n   디코더 레이어 처리:")
            print(f"     입력 모양: {x.shape}")
            print(f"     입력 평균/표준편차: {x.mean().item():.4f} / {x.std().item():.4f}")
        
        # 1. 셀프 어텐션 + 잔차 연결 + 정규화
        residual = x
        attn_output, attention_weights = self.self_attention(x, x, x, mask, show_process)
        x = self.norm1(residual + self.dropout(attn_output))
        
        if show_process:
            print(f"     어텐션 후 평균/표준편차: {x.mean().item():.4f} / {x.std().item():.4f}")
        
        # 2. 피드포워드 + 잔차 연결 + 정규화  
        residual = x
        ff_output = self.feed_forward(x, show_process)
        x = self.norm2(residual + self.dropout(ff_output))
        
        if show_process:
            print(f"     FFN 후 평균/표준편차: {x.mean().item():.4f} / {x.std().item():.4f}")
        
        return x, attention_weights

# 메인 트랜스포머 언어 모델
class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len, dropout=0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # 임베딩 레이어들
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # 트랜스포머 디코더 스택
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # 출력 레이어
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # 드롭아웃
        self.dropout = nn.Dropout(dropout)
        
        # 파라미터 초기화
        self.init_weights()
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n 트랜스포머 언어 모델 완성:")
        print(f"   어휘 크기: {vocab_size:,}")
        print(f"   모델 차원: {d_model}")
        print(f"   헤드 수: {n_heads}")
        print(f"   레이어 수: {n_layers}")
        print(f"   총 파라미터: {total_params:,}")
        print(f"   모델 크기: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    def init_weights(self):
        # Xavier 초기화
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=self.d_model ** -0.5)
    
    def forward(self, x, show_process=False, return_attention=False):
        batch_size, seq_len = x.shape
        
        if show_process:
            print(f"\n 모델 FORWARD 시작:")
            print(f"   입력 텐서 모양: {x.shape}")
            print(f"   토큰 ID 범위: [{x.min().item()}, {x.max().item()}]")
        
        # 1. 마스크 생성
        padding_mask = create_padding_mask(x)
        causal_mask = create_causal_mask(seq_len)
        
        # 마스크 결합: 패딩 마스크와 인과 마스크
        if causal_mask.device != x.device:
            causal_mask = causal_mask.to(x.device)
        
        combined_mask = padding_mask & causal_mask.unsqueeze(0).unsqueeze(0)
        
        if show_process:
            print(f"   패딩 마스크 모양: {padding_mask.shape}")
            print(f"   인과 마스크 모양: {causal_mask.shape}")
            print(f"   결합 마스크 모양: {combined_mask.shape}")
        
        # 2. 토큰 임베딩
        token_embeddings = self.token_embedding(x)
        
        if show_process:
            print(f"\n   토큰 임베딩:")
            print(f"     임베딩 후 모양: {token_embeddings.shape}")
            print(f"     임베딩 값 범위: [{token_embeddings.min().item():.3f}, {token_embeddings.max().item():.3f}]")
            print(f"     첫 번째 토큰 임베딩 (처음 8차원): {safe_numpy(token_embeddings[0, 0, :8])}")
        
        # 3. 임베딩 스케일링 (논문에서 제안)
        token_embeddings = token_embeddings * math.sqrt(self.d_model)
        
        if show_process:
            print(f"     스케일링 후 범위: [{token_embeddings.min().item():.3f}, {token_embeddings.max().item():.3f}]")
        
        # 4. 위치 인코딩 추가
        x = self.position_encoding(token_embeddings, show_process=show_process)
        x = self.dropout(x)
        
        if show_process:
            print(f"     드롭아웃 후 모양: {x.shape}")
        
        # 5. 트랜스포머 디코더 레이어들을 통과
        attention_weights_list = []
        
        for i, decoder_layer in enumerate(self.decoder_layers):
            if show_process:
                print(f"\n   디코더 레이어 {i+1}/{len(self.decoder_layers)}:")
            
            x, attention_weights = decoder_layer(x, combined_mask, show_process=(show_process and i == 0))
            
            if return_attention:
                attention_weights_list.append(attention_weights)
            
            if show_process:
                print(f"     레이어 {i+1} 출력 범위: [{x.min().item():.3f}, {x.max().item():.3f}]")
        
        # 6. 최종 레이어 정규화
        x = self.layer_norm(x)
        
        if show_process:
            print(f"\n   최종 정규화:")
            print(f"     정규화 후 평균: {x.mean().item():.4f}")
            print(f"     정규화 후 표준편차: {x.std().item():.4f}")
        
        # 7. 출력 프로젝션 (어휘 크기로 변환)
        logits = self.output_projection(x)
        
        if show_process:
            print(f"\n   출력 프로젝션:")
            print(f"     로짓 모양: {logits.shape}")
            print(f"     로짓 범위: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
            
            # 첫 번째 토큰에 대한 상위 5개 예측 출력
            first_token_logits = logits[0, 0]
            top5_values, top5_indices = torch.topk(first_token_logits, 5)
            print(f"     첫 번째 위치 상위 5개 예측:")
            for j, (val, idx) in enumerate(zip(top5_values, top5_indices)):
                print(f"       {j+1}. ID {idx.item()}: {val.item():.3f}")
        
        if return_attention:
            return logits, attention_weights_list
        
        return logits

# 학습 데이터 준비 함수
def prepare_training_data(tokenizer, texts, max_length=32):
    print(f"\n STEP 4: 학습 데이터 준비")
    print("-" * 80)
    
    # 텍스트들을 토큰화
    encoded_texts = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        encoded_texts.append(tokens)
    
    print(f"   원본 텍스트 수: {len(texts)}")
    print(f"   토큰화 예시:")
    for i, (text, tokens) in enumerate(zip(texts[:3], encoded_texts[:3])):
        print(f"     {i+1}. '{text}'")
        print(f"        토큰: {tokens}")
        words = [tokenizer.id_to_word[id] for id in tokens]
        print(f"        단어: {words}")
    
    # 배치 패딩
    padded_sequences = []
    original_lengths = []
    
    for tokens in encoded_texts:
        original_lengths.append(len(tokens))
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        else:
            tokens = tokens + [tokenizer.pad_id] * (max_length - len(tokens))
        padded_sequences.append(tokens)
    
    # 입력과 타겟 준비 (다음 토큰 예측)
    input_sequences = [seq[:-1] for seq in padded_sequences]  # 마지막 토큰 제외
    target_sequences = [seq[1:] for seq in padded_sequences]  # 첫 토큰 제외
    
    print(f"\n   패딩 후 시퀀스 길이: {max_length}")
    print(f"   평균 원본 길이: {np.mean(original_lengths):.1f}")
    print(f"   최대 원본 길이: {max(original_lengths)}")
    print(f"   입력 시퀀스 모양: ({len(input_sequences)}, {len(input_sequences[0])})")
    print(f"   타겟 시퀀스 모양: ({len(target_sequences)}, {len(target_sequences[0])})")
    
    # 입력/타겟 예시 출력
    print(f"\n   입력/타겟 예시 (첫 번째 샘플):")
    print(f"     입력:  {input_sequences[0][:10]}...")
    print(f"     타겟:  {target_sequences[0][:10]}...")
    
    input_words = [tokenizer.id_to_word[id] for id in input_sequences[0][:10]]
    target_words = [tokenizer.id_to_word[id] for id in target_sequences[0][:10]]
    print(f"     입력 단어: {input_words}")
    print(f"     타겟 단어: {target_words}")
    
    return torch.tensor(input_sequences), torch.tensor(target_sequences)

# 학습 함수
def train_model(model, train_inputs, train_targets, tokenizer, config):
    print(f"\n STEP 5: 모델 학습")
    print("-" * 80)
    
    model.train()
    
    # 옵티마이저와 손실 함수
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    
    print(f"   옵티마이저: Adam (lr={config['learning_rate']})")
    print(f"   손실 함수: CrossEntropyLoss (패딩 토큰 무시)")
    print(f"   에폭 수: {config['epochs']}")
    
    losses = []
    perplexities = []
    
    # 첫 번째 에폭은 상세히 출력
    for epoch in range(config['epochs']):
        optimizer.zero_grad()
        
        # 순전파
        show_detail = (epoch == 0)  # 첫 번째 에폭만 상세 출력
        outputs = model(train_inputs, show_process=show_detail)
        
        # 손실 계산
        loss = criterion(outputs.reshape(-1, model.vocab_size), train_targets.reshape(-1))
        
        # 역전파
        loss.backward()
        
        # 그래디언트 클리핑 (폭발 방지)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 통계 기록
        losses.append(loss.item())
        perplexity = torch.exp(loss).item()
        perplexities.append(perplexity)
        
        # 주기적 출력
        if epoch % 5 == 0 or epoch == config['epochs'] - 1:
            print(f"   에폭 {epoch+1:3d}: 손실={loss.item():.4f}, 펄플렉시티={perplexity:.2f}")
        
        # 첫 번째 에폭 상세 분석
        if epoch == 0:
            print(f"\n   첫 번째 에폭 상세 분석:")
            with torch.no_grad():
                sample_logits = outputs[0, 0]  # 첫 번째 샘플, 첫 번째 토큰
                probs = F.softmax(sample_logits, dim=-1)
                
                print(f"     샘플 로짓 범위: [{sample_logits.min().item():.3f}, {sample_logits.max().item():.3f}]")
                print(f"     최대 확률: {probs.max().item():.4f}")
                print(f"     엔트로피: {(-probs * torch.log(probs + 1e-8)).sum().item():.3f}")
                
                # 그래디언트 정보
                total_grad_norm = 0
                for param in model.parameters():
                    if param.grad is not None:
                        total_grad_norm += param.grad.data.norm(2).item() ** 2
                total_grad_norm = total_grad_norm ** 0.5
                print(f"     그래디언트 노름: {total_grad_norm:.6f}")
    
    print(f"\n   최종 손실: {losses[-1]:.4f}")
    print(f"   최종 펄플렉시티: {perplexities[-1]:.2f}")
    
    return losses, perplexities

# 학습 곡선 시각화
def plot_training_curves(losses, perplexities):
    print(f"\n STEP 6: 학습 곡선 시각화")
    print("-" * 80)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 손실 곡선
    ax1.plot(losses, 'b-', linewidth=2, label='Training Loss')
    ax1.set_title('학습 손실 (Training Loss)', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 펄플렉시티 곡선
    ax2.plot(perplexities, 'r-', linewidth=2, label='Perplexity')
    ax2.set_title('펄플렉시티 (Perplexity)', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Perplexity')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"   손실 감소율: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")
    print(f"   펄플렉시티 개선: {perplexities[0]:.2f} → {perplexities[-1]:.2f}")

# 어텐션 가중치 시각화
def visualize_attention_patterns(model, tokenizer, text="안녕하세요 좋은 하루입니다"):
    print(f"\n STEP 7: 어텐션 패턴 분석")
    print("-" * 80)
    
    model.eval()
    
    # 텍스트 토큰화
    tokens = tokenizer.encode(text, add_special_tokens=True)
    max_len = 16
    if len(tokens) < max_len:
        tokens.extend([tokenizer.pad_id] * (max_len - len(tokens)))
    else:
        tokens = tokens[:max_len]
    
    input_tensor = torch.tensor([tokens]).to(device)
    token_words = [tokenizer.id_to_word[id] for id in tokens]
    
    print(f"   분석할 텍스트: '{text}'")
    print(f"   토큰들: {token_words}")
    
    # 어텐션 가중치 추출
    with torch.no_grad():
        outputs, attention_weights_list = model(input_tensor, return_attention=True)
    
    # 유효한 토큰 수 계산 (패딩 제외)
    valid_len = len([t for t in tokens if t != tokenizer.pad_id])
    valid_words = token_words[:valid_len]
    
    print(f"   유효 토큰 수: {valid_len}")
    
    # 각 레이어별 어텐션 시각화
    n_layers = len(attention_weights_list)
    n_heads = attention_weights_list[0].shape[1]
    
    print(f"   레이어 수: {n_layers}, 헤드 수: {n_heads}")
    
    # 첫 번째 레이어의 모든 헤드 시각화
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    first_layer_attention = attention_weights_list[0][0]  # 첫 번째 배치
    
    for head in range(min(8, n_heads)):
        ax = axes[head]
        
        # 유효한 부분만 추출
        head_attention = safe_numpy(first_layer_attention[head, :valid_len, :valid_len])
        
        # 히트맵 그리기
        im = ax.imshow(head_attention, cmap='Blues', aspect='auto')
        ax.set_title(f'헤드 {head+1}')
        ax.set_xticks(range(valid_len))
        ax.set_yticks(range(valid_len))
        ax.set_xticklabels(valid_words, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(valid_words, fontsize=8)
        
        # 어텐션 값 표시 (높은 값만)
        for i in range(valid_len):
            for j in range(valid_len):
                if head_attention[i, j] > 0.1:  # 임계값 이상만 표시
                    ax.text(j, i, f'{head_attention[i, j]:.2f}',
                           ha='center', va='center', color='white', fontsize=6)
    
    plt.suptitle(f'멀티헤드 어텐션 패턴 (첫 번째 레이어)\n"{text}"', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # 어텐션 패턴 분석
    print(f"\n   어텐션 패턴 분석:")
    for head in range(min(4, n_heads)):
        head_attention = safe_numpy(first_layer_attention[head, :valid_len, :valid_len])
        print(f"\n   헤드 {head+1} 주요 패턴:")
        
        for i, query_word in enumerate(valid_words):
            # 자기 자신 제외하고 가장 높은 어텐션 찾기
            attention_row = head_attention[i].copy()
            attention_row[i] = 0  # 자기 자신 제외
            
            if attention_row.max() > 0.1:  # 의미있는 어텐션만
                max_idx = attention_row.argmax()
                max_attention = attention_row[max_idx]
                target_word = valid_words[max_idx]
                print(f"     '{query_word}' → '{target_word}' ({max_attention:.3f})")

# 텍스트 생성 함수 (온도 조절 가능)
def generate_text(model, tokenizer, prompt="안녕하세요", max_length=20, temperature=1.0, top_k=5):
    print(f"\n STEP 8: 텍스트 생성")
    print("-" * 80)
    
    model.eval()
    
    print(f"   시작 프롬프트: '{prompt}'")
    print(f"   최대 길이: {max_length}")
    print(f"   온도: {temperature}")
    print(f"   Top-K: {top_k}")
    
    # 프롬프트 토큰화
    tokens = tokenizer.encode(prompt, add_special_tokens=True, show_process=True)
    generated_tokens = tokens.copy()
    
    print(f"\n   생성 과정:")
    
    with torch.no_grad():
        for step in range(max_length):
            # 현재 시퀀스 준비
            current_sequence = generated_tokens[-model.max_seq_len + 1:] if len(generated_tokens) > model.max_seq_len - 1 else generated_tokens
            
            # 패딩
            padded_sequence = current_sequence + [tokenizer.pad_id] * (model.max_seq_len - 1 - len(current_sequence))
            input_tensor = torch.tensor([padded_sequence]).to(device)
            
            # 예측
            outputs = model(input_tensor)
            next_token_logits = outputs[0, len(current_sequence) - 1]
            
            # 온도 적용
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Top-K 샘플링
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, k=min(top_k, next_token_logits.size(-1)))
                probs = F.softmax(top_k_logits, dim=-1)
                
                # 다음 토큰 샘플링
                next_token_idx = torch.multinomial(probs, 1).item()
                next_token = top_k_indices[next_token_idx].item()
                
                # 상위 후보들 출력
                if step < 3:  # 처음 3스텝만 상세 출력
                    print(f"     스텝 {step+1} 후보:")
                    for i, (prob, idx) in enumerate(zip(probs, top_k_indices)):
                        word = tokenizer.id_to_word[idx.item()]
                        print(f"       {i+1}. {word} ({prob.item():.3f})")
                    print(f"     선택: {tokenizer.id_to_word[next_token]}")
            else:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
            
            # 종료 조건 확인
            if next_token == tokenizer.eos_id:
                print(f"     스텝 {step+1}: <EOS> 토큰 생성으로 종료")
                break
            
            generated_tokens.append(next_token)
            
            # 현재까지 생성된 텍스트
            current_text = tokenizer.decode(generated_tokens, remove_special_tokens=True)
            if step < 5 or step % 5 == 4:  # 처음 5스텝과 이후 5스텝마다
                print(f"     스텝 {step+1}: '{current_text}'")
    
    # 최종 결과
    final_text = tokenizer.decode(generated_tokens, remove_special_tokens=True)
    print(f"\n   최종 생성 텍스트: '{final_text}'")
    print(f"   생성된 토큰 수: {len(generated_tokens) - len(tokens)}")
    
    return final_text

# 다양한 온도로 생성 비교
def compare_generation_temperatures(model, tokenizer, prompt="인공지능은"):
    print(f"\n STEP 9: 온도별 생성 비교")
    print("-" * 80)
    
    temperatures = [0.1, 0.5, 1.0, 1.5, 2.0]
    
    print(f"   프롬프트: '{prompt}'")
    print(f"   비교할 온도: {temperatures}")
    
    for temp in temperatures:
        print(f"\n   온도 {temp}:")
        description = "매우 보수적" if temp < 0.5 else "보수적" if temp < 1.0 else "균형" if temp == 1.0 else "창의적" if temp < 2.0 else "매우 창의적"
        print(f"   ({description})")
        
        # 짧게 생성
        result = generate_text(model, tokenizer, prompt, max_length=8, temperature=temp, top_k=5)
        print(f"   결과: '{result}'")

# 헬퍼 함수: 큰 텐서를 요약해서 출력
def print_tensor_summary(tensor_array, name, show_full=False):
    if show_full or tensor_array.shape[0] <= 2:
        print(f"-- {name} --")
        print(tensor_array)
    else:
        print(f"-- {name} (showing first and last rows) --")
        # 각 행에서도 처음 5개, 마지막 5개만 출력
        first_row = tensor_array[0]
        last_row = tensor_array[-1]
        
        if len(first_row) > 10:
            first_5 = first_row[:5]
            last_5 = first_row[-5:]
            print(f"First row: [{' '.join(f'{x:.8f}' for x in first_5)} ... {' '.join(f'{x:.8f}' for x in last_5)}]")
        else:
            print(f"First row: {first_row}")
            
        if tensor_array.shape[0] > 2:
            print(f"... ({tensor_array.shape[0]-2} rows omitted) ...")
            
        if len(last_row) > 10:
            first_5 = last_row[:5]
            last_5 = last_row[-5:]
            print(f"Last row:  [{' '.join(f'{x:.8f}' for x in first_5)} ... {' '.join(f'{x:.8f}' for x in last_5)}]")
        else:
            print(f"Last row:  {last_row}")

# 단계별 디버깅 및 시각화 함수
def debug_transformer_step_by_step(model, tokenizer, sample_text="안녕하세요 좋은"):
    print(f"\n 트랜스포머 단계별 디버깅")
    print("=" * 80)
    print(f"분석할 텍스트: '{sample_text}'")
    
    model.eval()
    
    # 텍스트를 토큰화하고 패딩 추가
    tokens = tokenizer.encode(sample_text, add_special_tokens=True)
    max_len = 6
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    while len(tokens) < max_len:
        tokens.append(tokenizer.pad_id)
    
    input_tensor = torch.tensor([tokens]).to(device)
    token_words = [tokenizer.id_to_word[id] for id in tokens]
    
    print(f"\n 입력 토큰 처리")
    print("-" * 40)
    print(f"Input token ids: {safe_numpy(input_tensor)}")
    print(f"Token mapping:")
    for i, (token_id, word) in enumerate(zip(tokens, token_words)):
        print(f"  Position {i}: {token_id} -> '{word}'")
    
    padding_mask = (input_tensor == tokenizer.pad_id).float()
    print(f"Padding mask (1 where pad): {safe_numpy(padding_mask)}")
    
    with torch.no_grad():
        # 토큰 임베딩
        token_embeddings = model.token_embedding(input_tensor)
        print(f"\n 토큰 임베딩 (shape: {token_embeddings.shape})")
        print("-" * 40)
        print_tensor_summary(safe_numpy(token_embeddings[0]), "Token embeddings")
        
        # 위치 인코딩 추가
        token_embeddings_scaled = token_embeddings * math.sqrt(model.d_model)
        x = model.position_encoding(token_embeddings_scaled)
        print(f"\n 위치 인코딩 추가 (shape: {x.shape})")
        print("-" * 40)
        print_tensor_summary(safe_numpy(x[0]), "After adding positional encoding")
        
        # 멀티헤드 어텐션 분석
        print(f"\n 멀티헤드 어텐션 상세 분석")
        print("-" * 40)
        
        first_attention = model.decoder_layers[0].self_attention
        Q = first_attention.W_q(x)
        K = first_attention.W_k(x)
        V = first_attention.W_v(x)
        
        print(f"Q shape: {Q.shape}")
        
        # Q 샘플 요약 출력 (처음 2개 토큰만)
        q_sample = safe_numpy(Q[0, :2])
        print(f"Q sample (first 2 tokens, showing first/last 5 values):")
        for i, row in enumerate(q_sample):
            if len(row) > 10:
                first_5 = row[:5]
                last_5 = row[-5:]
                print(f"  Token {i}: [{' '.join(f'{x:.8f}' for x in first_5)} ... {' '.join(f'{x:.8f}' for x in last_5)}]")
            else:
                print(f"  Token {i}: {row}")
        
        # 멀티헤드로 분할
        batch_size, seq_len, d_model = Q.shape
        Q = Q.view(batch_size, seq_len, first_attention.n_heads, first_attention.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, first_attention.n_heads, first_attention.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, first_attention.n_heads, first_attention.d_k).transpose(1, 2)
        
        print(f"Q split into heads shape: {Q.shape}")
        
        # 어텐션 스코어 계산 (첫 번째 헤드만)
        scores = torch.matmul(Q[0, 0], K[0, 0].transpose(-1, -2)) / math.sqrt(first_attention.d_k)
        print(f"Raw attention scores (before mask & softmax):")
        print(safe_numpy(scores))
        
        # 마스크 적용
        padding_mask_expanded = (input_tensor == tokenizer.pad_id).unsqueeze(1).unsqueeze(2)
        causal_mask = create_causal_mask(seq_len).to(device)
        combined_mask = padding_mask_expanded & causal_mask.unsqueeze(0).unsqueeze(0)
        
        mask_for_attention = combined_mask[0, 0]
        scores_masked = scores.clone()
        scores_masked = scores_masked.masked_fill(~mask_for_attention, -1e9)
        
        print(f"Scores after applying mask:")
        print(safe_numpy(scores_masked))
        
        # 소프트맥스 적용
        attention_weights = F.softmax(scores_masked, dim=-1)
        print(f"Attention weights (softmaxed):")
        print(safe_numpy(attention_weights))
        
        # Value와 곱셈
        context = torch.matmul(attention_weights.unsqueeze(0), V[0, 0].unsqueeze(0))
        print(f"Output per head sample:")
        output_per_head = safe_numpy(context[0])
        for i, row in enumerate(output_per_head):
            if len(row) > 10:
                first_5 = row[:5]
                last_5 = row[-5:]
                print(f"  Head output {i}: [{' '.join(f'{x:.8f}' for x in first_5)} ... {' '.join(f'{x:.8f}' for x in last_5)}]")
            else:
                print(f"  Head output {i}: {row}")
        
        # 전체 멀티헤드 어텐션 실행
        attn_output, _ = first_attention(x, x, x, combined_mask)
        print(f"\n 멀티헤드 어텐션 최종 출력 (shape: {attn_output.shape})")
        print("-" * 40)
        print_tensor_summary(safe_numpy(attn_output[0]), "Final MHA output")
        
        # 피드포워드 네트워크
        ff_output = model.decoder_layers[0].feed_forward(attn_output)
        print(f"\n 피드포워드 네트워크 (shape: {ff_output.shape})")
        print("-" * 40)
        print_tensor_summary(safe_numpy(ff_output[0]), "Feed-forward output")
        
        # 잔차 연결 및 레이어 정규화
        print(f"\n 잔차 연결 및 레이어 정규화")
        print("-" * 40)
        
        x_after_attn = model.decoder_layers[0].norm1(x + model.decoder_layers[0].dropout(attn_output))
        print(f"After attention + residual + norm: Mean={x_after_attn.mean().item():.4f}, Std={x_after_attn.std().item():.4f}")
        
        x_final = model.decoder_layers[0].norm2(x_after_attn + model.decoder_layers[0].dropout(ff_output))
        print(f"After feedforward + residual + norm: Mean={x_final.mean().item():.4f}, Std={x_final.std().item():.4f}")
        
        # 최종 출력 프로젝션
        print(f"\n 최종 출력 및 예측")
        print("-" * 40)
        
        # 모든 레이어 통과
        x_all_layers = x
        for layer in model.decoder_layers:
            x_all_layers, _ = layer(x_all_layers, combined_mask)
        
        x_all_layers = model.layer_norm(x_all_layers)
        logits = model.output_projection(x_all_layers)
        
        print(f"Final logits shape: {logits.shape}")
        print(f"Logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
        
        # 각 위치별 상위 예측 출력 (유효한 토큰만)
        print(f"Top predictions per position:")
        for pos in range(seq_len):
            if tokens[pos] != tokenizer.pad_id:
                pos_logits = logits[0, pos]
                top3_values, top3_indices = torch.topk(pos_logits, 3)
                print(f"  Position {pos} ('{token_words[pos]}'):")
                for i, (val, idx) in enumerate(zip(top3_values, top3_indices)):
                    predicted_word = tokenizer.id_to_word[idx.item()]
                    prob = F.softmax(pos_logits, dim=-1)[idx].item()
                    print(f"    {i+1}. {predicted_word} (logit: {val.item():.3f}, prob: {prob:.3f})")
        
        print(f"\n 요약")
        print("-" * 40)
        print(f"✓ 입력: '{sample_text}' → {len([t for t in tokens if t != tokenizer.pad_id])} 토큰")
        print(f"✓ 처리: 임베딩 → 위치인코딩 → {len(model.decoder_layers)}개 레이어 → 출력({logits.shape})")

# 메인 실행 함수
def main():
    # 토크나이저 생성
    tokenizer = AdvancedTokenizer(sample_texts)
    
    # 마스킹 시연
    sample_input, mask = demonstrate_masking(tokenizer)
    
    # 모델 생성
    model = TransformerLanguageModel(
        vocab_size=tokenizer.vocab_size,
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        max_seq_len=config['max_seq_len'],
        dropout=config['dropout']
    ).to(device)
    
    # 단계별 디버깅 (학습 전 - 초기 상태)
    print(f"\n" + "="*100)
    print(" 학습 전 모델 상태 분석")
    print("="*100)
    debug_transformer_step_by_step(model, tokenizer, "안녕하세요 좋은")
    
    # 학습 데이터 준비
    train_inputs, train_targets = prepare_training_data(tokenizer, sample_texts, max_length=config['max_seq_len'] - 1)
    train_inputs = train_inputs.to(device)
    train_targets = train_targets.to(device)
    
    # 모델 학습
    losses, perplexities = train_model(model, train_inputs, train_targets, tokenizer, config)
    
    # 단계별 디버깅 (학습 후 - 훈련된 상태)
    print(f"\n" + "="*100)
    print(" 학습 후 모델 상태 분석")
    print("="*100)
    debug_transformer_step_by_step(model, tokenizer, "안녕하세요 좋은")
    
    # 학습 곡선 시각화
    plot_training_curves(losses, perplexities)
    
    # 어텐션 패턴 분석
    visualize_attention_patterns(model, tokenizer)
    
    # 텍스트 생성
    generate_text(model, tokenizer, "안녕하세요", max_length=15, temperature=1.0)
    
    # 온도별 생성 비교
    compare_generation_temperatures(model, tokenizer)
    
    print(f"\n STEP 10: 학습 완료!")
    print("=" * 80)
    print("Transformer 언어 생성 모델의 핵심 개념들을 모두 학습했습니다!")
    print()
    print(" 학습한 핵심 개념들:")
    print("   1. 토크나이제이션: 텍스트를 숫자로 변환")
    print("   2. 임베딩: 토큰을 벡터로 표현") 
    print("   3. 위치 인코딩: 순서 정보 추가")
    print("   4. 마스킹: 패딩과 미래 토큰 제어")
    print("   5. 멀티헤드 어텐션: 다양한 관점에서 관계 학습")
    print("   6. Q, K, V 메커니즘: 어텐션 계산 과정")
    print("   7. 스케일드 닷-프로덕트: 어텐션 스코어 계산")
    print("   8. 소프트맥스: 확률 분포 변환")
    print("   9. 피드포워드: 비선형 변환")
    print("   10. 잔차 연결 & 층 정규화: 학습 안정화")
    print("   11. 언어 모델링: 다음 토큰 예측")
    print("   12. 텍스트 생성: 자기회귀적 디코딩")
    print("   13. 온도 조절: 생성 다양성 제어")
    print("   14. Top-K 샘플링: 품질 있는 다양성")
    print()
    print(" 핵심 수식들:")
    print("   • Attention(Q,K,V) = softmax(QK^T/√d_k)V")
    print("   • MultiHead = Concat(head_1,...,head_h)W^O")  
    print("   • LayerNorm(x + Sublayer(x))")
    print("   • PE(pos,2i) = sin(pos/10000^(2i/d_model))")
    print("   • PE(pos,2i+1) = cos(pos/10000^(2i/d_model))")
    print()
    print(" 실용적 활용:")
    print("   • 다음 단어 예측 (언어 모델링)")
    print("   • 창의적 텍스트 생성")
    print("   • 대화 시스템")
    print("   • 코드 생성")
    print("   • 번역 시스템")
    print()
    print("=" * 80)

if __name__ == "__main__":
    main()
