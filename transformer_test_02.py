!apt-get -qq install fonts-nanum
!fc-cache -fv
!rm -rf ~/.cache/matplotlib

import matplotlib.pyplot as plt
plt.rc('font', family='NanumGothic')


# Transformer 교육실습 코드

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tqdm import tqdm
import math
import time
import warnings
warnings.filterwarnings('ignore')

# 재현성을 위한 시드 설정
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

set_seed(42)

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 파라미터 설정
params = {
    'batch_size': 32,
    'num_epochs': 10,
    'learning_rate': 0.0001,
    'dropout': 0.1,
    'min_frequency': 2,
    'vocab_size': 16000,
    'num_layers': 6,
    'num_heads': 8,
    'hidden_dim': 512,
    'ffn_dim': 2048,
    'max_len': 128,
    'warmup_steps': 4000
}

print("Transformer 파라미터 설정:")
for key, value in params.items():
    print(f"  {key}: {value}")

# 샘플 데이터 생성 함수
def create_sample_data(num_samples=10000):
    """
    간단한 영어-한국어 번역 샘플 데이터를 생성
    실제 사용 시에는 AI Hub 데이터나 다른 병렬 코퍼스를 사용
    """
    # 간단한 패턴 기반 샘플 데이터
    eng_samples = [
        "I love you",
        "How are you",
        "Good morning",
        "Thank you",
        "See you later",
        "What is your name",
        "Nice to meet you",
        "Have a good day",
        "I am fine",
        "Where are you from"
    ]
    
    kor_samples = [
        "나는 당신을 사랑합니다",
        "어떻게 지내세요",
        "좋은 아침입니다",
        "감사합니다",
        "나중에 봐요",
        "당신의 이름은 무엇입니까",
        "만나서 반갑습니다",
        "좋은 하루 보내세요",
        "저는 괜찮습니다",
        "어디서 오셨나요"
    ]
    
    # 패턴을 반복하여 더 많은 데이터 생성
    eng_data = []
    kor_data = []
    
    for i in range(num_samples):
        idx = i % len(eng_samples)
        eng_data.append(eng_samples[idx])
        kor_data.append(kor_samples[idx])
    
    return eng_data, kor_data

# BPE 토크나이저 생성 및 훈련
def create_and_train_tokenizer(sentences, vocab_size=16000):
    """BPE 토크나이저를 생성하고 훈련"""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=['[PAD]', '[SOS]', '[EOS]', '[UNK]']
    )
    
    # 문장들을 파일로 저장
    with open('temp_data.txt', 'w', encoding='utf-8') as f:
        for sentence in sentences:
            f.write(sentence + '\n')
    
    tokenizer.train(['temp_data.txt'], trainer)
    return tokenizer

# 위치 인코딩 클래스
class PositionalEncoding(nn.Module):
    """
    Transformer의 핵심 컴포넌트: 위치 정보를 추가하는 인코딩
    시퀀스 내 각 토큰의 위치 정보를 사인/코사인 함수로 인코딩
    """
    def __init__(self, hidden_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 짝수 인덱스에는 sin, 홀수 인덱스에는 cos 적용
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * 
                           (-math.log(10000.0) / hidden_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# 멀티헤드 어텐션 클래스
class MultiHeadAttention(nn.Module):
    """
    Transformer의 핵심: 멀티헤드 어텐션 메커니즘
    여러 개의 어텐션 헤드를 병렬로 사용하여 다양한 관점에서 정보를 처리
    """
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Query, Key, Value 변환을 위한 선형 레이어
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Query, Key, Value 변환 및 헤드별로 분할
        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 어텐션 스코어 계산
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # 마스크 적용 (패딩이나 미래 토큰 방지)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 소프트맥스로 어텐션 가중치 계산
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 가중합 계산
        context = torch.matmul(attn_weights, V)
        
        # 헤드 결합
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.hidden_dim)
        
        output = self.output_linear(context)
        return output, attn_weights

# 피드포워드 네트워크
class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed Forward Network
    각 위치에서 독립적으로 적용되는 2층 완전연결 네트워크
    """
    def __init__(self, hidden_dim, ffn_dim, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        return self.fc2(self.dropout(self.activation(self.fc1(x))))

# 인코더 레이어
class EncoderLayer(nn.Module):
    """
    Transformer 인코더의 단일 레이어
    Self-Attention + Feed Forward + Residual Connection + Layer Norm
    """
    def __init__(self, hidden_dim, num_heads, ffn_dim, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_dim, ffn_dim, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-Attention + Residual Connection + Layer Norm
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed Forward + Residual Connection + Layer Norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

# 디코더 레이어
class DecoderLayer(nn.Module):
    """
    Transformer 디코더의 단일 레이어
    Masked Self-Attention + Cross-Attention + Feed Forward + Residual Connections
    """
    def __init__(self, hidden_dim, num_heads, ffn_dim, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_dim, ffn_dim, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, self_attn_mask=None, cross_attn_mask=None):
        # Masked Self-Attention
        attn_output, _ = self.self_attn(x, x, x, self_attn_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-Attention (인코더 출력에 대한 어텐션)
        cross_output, cross_weights = self.cross_attn(x, encoder_output, encoder_output, cross_attn_mask)
        x = self.norm2(x + self.dropout(cross_output))
        
        # Feed Forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x, cross_weights

# Transformer 모델
class Transformer(nn.Module):
    """
    완전한 Transformer 모델
    인코더와 디코더를 결합한 Seq2Seq 모델
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, hidden_dim, num_heads, 
                 num_layers, ffn_dim, max_len, dropout=0.1, pad_idx=0):
        super(Transformer, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.pad_idx = pad_idx
        
        # 임베딩 레이어
        self.src_embedding = nn.Embedding(src_vocab_size, hidden_dim, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, hidden_dim, padding_idx=pad_idx)
        
        # 위치 인코딩
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len)
        
        # 인코더 및 디코더 레이어들
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(hidden_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(hidden_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # 출력 프로젝션
        self.output_projection = nn.Linear(hidden_dim, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # 가중치 초기화
        self.init_weights()
    
    def init_weights(self):
        """Xavier 초기화"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_padding_mask(self, x):
        """패딩 마스크 생성"""
        return (x != self.pad_idx).unsqueeze(1).unsqueeze(2)
    
    def create_subsequent_mask(self, size):
        """미래 토큰을 보지 못하도록 하는 마스크"""
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask == 0
    
    def forward(self, src, tgt):
        # 마스크 생성
        src_mask = self.create_padding_mask(src)
        tgt_mask = self.create_padding_mask(tgt)
        
        # 디코더용 subsequent mask
        seq_len = tgt.size(1)
        subsequent_mask = self.create_subsequent_mask(seq_len).to(tgt.device)
        tgt_mask = tgt_mask & subsequent_mask.unsqueeze(0).unsqueeze(0)
        
        # 임베딩 + 위치 인코딩
        src_embed = self.dropout(self.pos_encoding(self.src_embedding(src) * math.sqrt(self.hidden_dim)))
        tgt_embed = self.dropout(self.pos_encoding(self.tgt_embedding(tgt) * math.sqrt(self.hidden_dim)))
        
        # 인코더
        encoder_output = src_embed.transpose(0, 1)  # seq_len, batch, hidden_dim
        for encoder_layer in self.encoder_layers:
            encoder_output = encoder_layer(encoder_output.transpose(0, 1), src_mask).transpose(0, 1)
        
        # 디코더
        decoder_output = tgt_embed.transpose(0, 1)  # seq_len, batch, hidden_dim
        cross_weights = None
        for decoder_layer in self.decoder_layers:
            decoder_output, cross_weights = decoder_layer(
                decoder_output.transpose(0, 1), 
                encoder_output.transpose(0, 1), 
                tgt_mask, 
                src_mask
            )
            decoder_output = decoder_output.transpose(0, 1)
        
        # 출력 프로젝션
        output = self.output_projection(decoder_output.transpose(0, 1))
        
        return output, cross_weights

# 커스텀 데이터셋 클래스
class TranslationDataset(Dataset):
    """번역 데이터셋 클래스"""
    def __init__(self, src_data, tgt_data, src_tokenizer, tgt_tokenizer, max_len, pad_idx, sos_idx, eos_idx):
        self.src_data = src_data
        self.tgt_data = tgt_data
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
    
    def __len__(self):
        return len(self.src_data)
    
    def __getitem__(self, idx):
        src_text = self.src_data[idx]
        tgt_text = self.tgt_data[idx]
        
        # 토크나이즈
        src_tokens = self.src_tokenizer.encode(src_text).ids
        tgt_tokens = self.tgt_tokenizer.encode(tgt_text).ids
        
        # SOS/EOS 토큰 추가 및 길이 조정
        src_tokens = src_tokens[:self.max_len-2]
        tgt_tokens = tgt_tokens[:self.max_len-2]
        
        src_tokens = [self.sos_idx] + src_tokens + [self.eos_idx]
        tgt_tokens = [self.sos_idx] + tgt_tokens + [self.eos_idx]
        
        # 패딩
        src_tokens += [self.pad_idx] * (self.max_len - len(src_tokens))
        tgt_tokens += [self.pad_idx] * (self.max_len - len(tgt_tokens))
        
        return torch.LongTensor(src_tokens), torch.LongTensor(tgt_tokens)

# 학습률 스케줄러
class TransformerLRScheduler:
    """Transformer 논문에서 제안한 학습률 스케줄러"""
    def __init__(self, optimizer, hidden_dim, warmup_steps=4000):
        self.optimizer = optimizer
        self.hidden_dim = hidden_dim
        self.warmup_steps = warmup_steps
        self.step_num = 0
    
    def step(self):
        self.step_num += 1
        lr = self.hidden_dim ** (-0.5) * min(
            self.step_num ** (-0.5),
            self.step_num * self.warmup_steps ** (-1.5)
        )
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def zero_grad(self):
        self.optimizer.zero_grad()

# 모델 학습 함수
def train_model(model, train_loader, criterion, scheduler, num_epochs, device):
    """모델 학습"""
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (src, tgt) in enumerate(progress_bar):
            src, tgt = src.to(device), tgt.to(device)
            
            # 입력과 타겟 분리 (teacher forcing)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            scheduler.zero_grad()
            
            # 순전파
            output, _ = model(src, tgt_input)
            
            # 손실 계산
            output = output.reshape(-1, output.size(-1))
            tgt_output = tgt_output.reshape(-1)
            
            loss = criterion(output, tgt_output)
            
            # 역전파
            loss.backward()
            
            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scheduler.step()
            
            total_loss += loss.item()
            
            # 진행률 업데이트
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
            })
        
        print(f'Epoch {epoch+1} completed. Average Loss: {total_loss/len(train_loader):.4f}')

# 추론 함수
def translate(model, src_text, src_tokenizer, tgt_tokenizer, device, max_len=50, pad_idx=0, sos_idx=1, eos_idx=2):
    """단일 문장 번역"""
    model.eval()
    
    with torch.no_grad():
        # 소스 문장 토크나이즈
        src_tokens = src_tokenizer.encode(src_text).ids
        src_tokens = [sos_idx] + src_tokens + [eos_idx]
        src_tensor = torch.LongTensor(src_tokens).unsqueeze(0).to(device)
        
        # 타겟 시퀀스 초기화
        tgt_tokens = [sos_idx]
        
        for _ in range(max_len):
            tgt_tensor = torch.LongTensor(tgt_tokens).unsqueeze(0).to(device)
            
            output, _ = model(src_tensor, tgt_tensor)
            next_token = output[0, -1, :].argmax().item()
            
            tgt_tokens.append(next_token)
            
            if next_token == eos_idx:
                break
        
        # 디코딩
        result = tgt_tokenizer.decode(tgt_tokens[1:-1])  # SOS, EOS 제거
        
    return result

# 메인 실행 함수
def main():
    print("=== Transformer 교육실습 시작 ===")
    
    # 1. 데이터 준비
    print("\n1. 샘플 데이터 생성 중...")
    eng_data, kor_data = create_sample_data(1000)
    print(f"생성된 데이터 샘플: {len(eng_data)}개")
    print(f"영어 예시: {eng_data[0]}")
    print(f"한국어 예시: {kor_data[0]}")
    
    # 2. 토크나이저 훈련
    print("\n2. 토크나이저 훈련 중...")
    eng_tokenizer = create_and_train_tokenizer(eng_data, params['vocab_size'])
    kor_tokenizer = create_and_train_tokenizer(kor_data, params['vocab_size'])
    
    # 특수 토큰 인덱스
    pad_idx = eng_tokenizer.token_to_id('[PAD]')
    sos_idx = eng_tokenizer.token_to_id('[SOS]')
    eos_idx = eng_tokenizer.token_to_id('[EOS]')
    
    print(f"어휘 사이즈 - 영어: {eng_tokenizer.get_vocab_size()}, 한국어: {kor_tokenizer.get_vocab_size()}")
    print(f"특수 토큰 - PAD: {pad_idx}, SOS: {sos_idx}, EOS: {eos_idx}")
    
    # 3. 데이터셋 및 데이터로더 생성
    print("\n3. 데이터셋 준비 중...")
    dataset = TranslationDataset(
        eng_data, kor_data, eng_tokenizer, kor_tokenizer,
        params['max_len'], pad_idx, sos_idx, eos_idx
    )
    
    train_loader = DataLoader(
        dataset, 
        batch_size=params['batch_size'], 
        shuffle=True,
        num_workers=0
    )
    
    # 4. 모델 생성
    print("\n4. Transformer 모델 생성 중...")
    model = Transformer(
        src_vocab_size=eng_tokenizer.get_vocab_size(),
        tgt_vocab_size=kor_tokenizer.get_vocab_size(),
        hidden_dim=params['hidden_dim'],
        num_heads=params['num_heads'],
        num_layers=params['num_layers'],
        ffn_dim=params['ffn_dim'],
        max_len=params['max_len'],
        dropout=params['dropout'],
        pad_idx=pad_idx
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"모델 파라미터 수: {total_params:,}")
    
    # 5. 손실 함수 및 옵티마이저 설정
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    scheduler = TransformerLRScheduler(optimizer, params['hidden_dim'], params['warmup_steps'])
    
    # 6. 모델 학습
    print("\n5. 모델 학습 시작...")
    train_model(model, train_loader, criterion, scheduler, params['num_epochs'], device)
    
    # 7. 모델 저장
    torch.save(model.state_dict(), 'transformer_model.pth')
    print("\n모델이 'transformer_model.pth'로 저장되었습니다.")
    
    # 8. 추론 테스트
    print("\n6. 추론 테스트...")
    test_sentences = ["I love you", "How are you", "Good morning"]
    
    for test_sent in test_sentences:
        result = translate(model, test_sent, eng_tokenizer, kor_tokenizer, device, 
                         max_len=50, pad_idx=pad_idx, sos_idx=sos_idx, eos_idx=eos_idx)
        print(f"영어: {test_sent}")
        print(f"번역: {result}")
        print()
    
    print("=== Transformer 교육실습 완료 ===")

# 어텐션 가중치 시각화 함수
def visualize_attention(model, src_text, tgt_text, src_tokenizer, tgt_tokenizer, device, pad_idx=0, sos_idx=1, eos_idx=2):
    """어텐션 가중치 시각화"""
    model.eval()
    
    with torch.no_grad():
        # 토크나이즈
        src_tokens = src_tokenizer.encode(src_text).ids
        tgt_tokens = tgt_tokenizer.encode(tgt_text).ids
        
        src_tokens = [sos_idx] + src_tokens + [eos_idx]
        tgt_tokens = [sos_idx] + tgt_tokens + [eos_idx]
        
        src_tensor = torch.LongTensor(src_tokens).unsqueeze(0).to(device)
        tgt_tensor = torch.LongTensor(tgt_tokens).unsqueeze(0).to(device)
        
        # 순전파
        output, attention_weights = model(src_tensor, tgt_tensor)
        
        # 어텐션 가중치 시각화
        if attention_weights is not None:
            attn = attention_weights[0, 0].cpu().numpy()  # 첫 번째 헤드
            
            plt.figure(figsize=(10, 8))
            plt.imshow(attn, cmap='Blues')
            plt.colorbar()
            plt.title('Cross-Attention Weights (Head 1)')
            plt.xlabel('Source Tokens')
            plt.ylabel('Target Tokens')
            
            # 토큰 라벨 추가
            src_labels = [src_tokenizer.decode([token]) for token in src_tokens]
            tgt_labels = [tgt_tokenizer.decode([token]) for token in tgt_tokens]
            
            plt.xticks(range(len(src_labels)), src_labels, rotation=45)
            plt.yticks(range(len(tgt_labels)), tgt_labels)
            plt.tight_layout()
            plt.show()

# Transformer 구조 설명 및 시각화 함수
def explain_transformer_architecture():
    """Transformer 아키텍처 설명"""
    print("=== Transformer 아키텍처 핵심 개념 ===")
    print()
    
    print("1. 멀티헤드 어텐션 (Multi-Head Attention)")
    print("   - 여러 개의 어텐션 헤드를 병렬로 사용")
    print("   - 각 헤드는 서로 다른 관점에서 관련성을 학습")
    print("   - Query, Key, Value 벡터를 사용한 어텐션 메커니즘")
    print("   - 공식: Attention(Q,K,V) = softmax(QK^T/√d_k)V")
    print()
    
    print("2. 위치 인코딩 (Positional Encoding)")
    print("   - 순서 정보가 없는 어텐션에 위치 정보 추가")
    print("   - 사인/코사인 함수를 사용한 고정 인코딩")
    print("   - PE(pos,2i) = sin(pos/10000^(2i/d_model))")
    print("   - PE(pos,2i+1) = cos(pos/10000^(2i/d_model))")
    print()
    
    print("3. 인코더-디코더 구조")
    print("   - 인코더: 입력 시퀀스를 컨텍스트 표현으로 변환")
    print("   - 디코더: 컨텍스트를 바탕으로 출력 시퀀스 생성")
    print("   - 각각 여러 레이어로 구성 (보통 6개)")
    print()
    
    print("4. 잔차 연결 (Residual Connection)과 레이어 정규화")
    print("   - 그래디언트 소실 문제 해결")
    print("   - LayerNorm(x + Sublayer(x)) 구조")
    print()
    
    print("5. 마스킹 (Masking)")
    print("   - 패딩 마스크: 패딩 토큰 무시")
    print("   - 룩어헤드 마스크: 미래 토큰 참조 방지")
    print()

# 성능 평가 함수
def evaluate_model(model, test_data, src_tokenizer, tgt_tokenizer, device, pad_idx=0, sos_idx=1, eos_idx=2):
    """모델 성능 평가"""
    model.eval()
    
    correct_translations = 0
    total_translations = len(test_data)
    
    print("=== 모델 성능 평가 ===")
    
    for i, (src_text, tgt_text) in enumerate(test_data[:10]):  # 처음 10개만 평가
        predicted = translate(model, src_text, src_tokenizer, tgt_tokenizer, device,
                            max_len=50, pad_idx=pad_idx, sos_idx=sos_idx, eos_idx=eos_idx)
        
        print(f"테스트 {i+1}:")
        print(f"  입력: {src_text}")
        print(f"  정답: {tgt_text}")
        print(f"  예측: {predicted}")
        
        # 간단한 정확도 계산 (완전 일치)
        if predicted.strip() == tgt_text.strip():
            correct_translations += 1
        print()
    
    accuracy = correct_translations / min(10, total_translations) * 100
    print(f"정확도 (완전 일치): {accuracy:.2f}%")

# 고급 기능: 빔 서치 구현
def beam_search_translate(model, src_text, src_tokenizer, tgt_tokenizer, device, 
                         beam_size=3, max_len=50, pad_idx=0, sos_idx=1, eos_idx=2):
    """빔 서치를 사용한 번역"""
    model.eval()
    
    with torch.no_grad():
        # 소스 문장 토크나이즈
        src_tokens = src_tokenizer.encode(src_text).ids
        src_tokens = [sos_idx] + src_tokens + [eos_idx]
        src_tensor = torch.LongTensor(src_tokens).unsqueeze(0).to(device)
        
        # 빔 초기화
        beams = [(torch.LongTensor([sos_idx]).to(device), 0.0)]
        
        for step in range(max_len):
            candidates = []
            
            for seq, score in beams:
                if seq[-1].item() == eos_idx:
                    candidates.append((seq, score))
                    continue
                
                # 현재 시퀀스에 대한 다음 토큰 예측
                tgt_tensor = seq.unsqueeze(0)
                output, _ = model(src_tensor, tgt_tensor)
                next_token_logits = output[0, -1, :]
                next_token_probs = F.log_softmax(next_token_logits, dim=-1)
                
                # 상위 beam_size개 토큰 선택
                top_probs, top_indices = next_token_probs.topk(beam_size)
                
                for prob, idx in zip(top_probs, top_indices):
                    new_seq = torch.cat([seq, idx.unsqueeze(0)])
                    new_score = score + prob.item()
                    candidates.append((new_seq, new_score))
            
            # 상위 beam_size개 후보 선택
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:beam_size]
            
            # 모든 빔이 종료되었는지 확인
            if all(seq[-1].item() == eos_idx for seq, _ in beams):
                break
        
        # 최고 점수 시퀀스 선택
        best_seq = beams[0][0]
        result_tokens = best_seq[1:-1].cpu().tolist()  # SOS, EOS 제거
        result = tgt_tokenizer.decode(result_tokens)
        
    return result

# 학습 곡선 시각화 함수
def plot_training_curves(losses):
    """학습 곡선 시각화"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

# 개선된 메인 함수
def main_advanced():
    """고급 기능을 포함한 메인 함수"""
    print("=== 고급 Transformer 교육실습 ===")
    
    # Transformer 아키텍처 설명
    explain_transformer_architecture()
    
    # 기본 학습 실행
    main()
    
    print("\n=== 고급 기능 테스트 ===")
    
    # 데이터 재준비 (간단한 예시)
    eng_data, kor_data = create_sample_data(100)
    eng_tokenizer = create_and_train_tokenizer(eng_data, params['vocab_size'])
    kor_tokenizer = create_and_train_tokenizer(kor_data, params['vocab_size'])
    
    pad_idx = eng_tokenizer.token_to_id('[PAD]')
    sos_idx = eng_tokenizer.token_to_id('[SOS]')
    eos_idx = eng_tokenizer.token_to_id('[EOS]')
    
    # 모델 로드 (실제로는 저장된 모델을 로드)
    model = Transformer(
        src_vocab_size=eng_tokenizer.get_vocab_size(),
        tgt_vocab_size=kor_tokenizer.get_vocab_size(),
        hidden_dim=params['hidden_dim'],
        num_heads=params['num_heads'],
        num_layers=params['num_layers'],
        ffn_dim=params['ffn_dim'],
        max_len=params['max_len'],
        dropout=params['dropout'],
        pad_idx=pad_idx
    ).to(device)
    
    # 빔 서치 테스트
    print("\n1. 빔 서치 번역 테스트:")
    test_sentence = "I love you"
    
    # 일반 번역
    normal_result = translate(model, test_sentence, eng_tokenizer, kor_tokenizer, device,
                            max_len=50, pad_idx=pad_idx, sos_idx=sos_idx, eos_idx=eos_idx)
    
    # 빔 서치 번역
    beam_result = beam_search_translate(model, test_sentence, eng_tokenizer, kor_tokenizer, device,
                                      beam_size=3, max_len=50, pad_idx=pad_idx, sos_idx=sos_idx, eos_idx=eos_idx)
    
    print(f"입력: {test_sentence}")
    print(f"일반 번역: {normal_result}")
    print(f"빔 서치 번역: {beam_result}")
    
    # 어텐션 시각화 (예시)
    print("\n2. 어텐션 가중치 시각화:")
    print("(실제 시각화는 matplotlib가 설치된 환경에서 확인 가능)")
    
    # 성능 평가
    print("\n3. 모델 성능 평가:")
    test_data = list(zip(eng_data[:5], kor_data[:5]))
    evaluate_model(model, test_data, eng_tokenizer, kor_tokenizer, device,
                  pad_idx=pad_idx, sos_idx=sos_idx, eos_idx=eos_idx)

# 추가 유틸리티 함수들
def count_parameters(model):
    """모델 파라미터 수 계산"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_tokenizers(src_tokenizer, tgt_tokenizer, save_dir="./tokenizers"):
    """토크나이저 저장"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    src_tokenizer.save(f"{save_dir}/src_tokenizer.json")
    tgt_tokenizer.save(f"{save_dir}/tgt_tokenizer.json")
    print(f"토크나이저가 {save_dir}에 저장되었습니다.")

def load_tokenizers(save_dir="./tokenizers"):
    """토크나이저 로드"""
    from tokenizers import Tokenizer
    src_tokenizer = Tokenizer.from_file(f"{save_dir}/src_tokenizer.json")
    tgt_tokenizer = Tokenizer.from_file(f"{save_dir}/tgt_tokenizer.json")
    return src_tokenizer, tgt_tokenizer

# 실행부
if __name__ == "__main__":
    # 기본 실습 실행
    main()
    
    # 고급 실습을 원할 경우 주석 해제
    # main_advanced()
    
    print("\n=== 실습 완료 ===")
    print("주요 학습 내용:")
    print("1. Transformer 아키텍처의 핵심 구성 요소")
    print("2. 멀티헤드 어텐션 메커니즘")
    print("3. 위치 인코딩의 역할")
    print("4. 인코더-디코더 구조")
    print("5. 마스킹과 잔차 연결")
    print("6. 실제 번역 모델 구현 및 학습")
    print("\n추가 실험 아이디어:")
    print("- 다른 언어 쌍으로 실험")
    print("- 하이퍼파라미터 튜닝")
    print("- 더 큰 데이터셋으로 학습")
    print("- BLEU 스코어 등 정량적 평가 지표 추가")
    print("- 사전 훈련된 임베딩 사용")
