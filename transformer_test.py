!apt-get -qq install fonts-nanum
!fc-cache -fv
!rm -rf ~/.cache/matplotlib

import matplotlib.pyplot as plt
plt.rc('font', family='NanumGothic')
import matplotlib as mpl
mpl.rcParams['axes.unicode_minus'] = False

# ===============================================================================
# 🤖 Transformer 기반 간단한 언어모델 만들기 & 이해하기
# ===============================================================================
# 이 예제를 통해 배울 수 있는 것들:
# 1. 토크나이저와 인코딩/임베딩의 역할
# 2. Multi-Head Attention의 작동 원리
# 3. 온도(Temperature)가 생성에 미치는 영향
# 4. 어텐션 스코어 시각화를 통한 모델 이해
# 5. 멀티헤드 어텐션의 다양한 관점
# ===============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

# GPU 사용 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 사용 중인 디바이스: {device}")
print("=" * 80)

# ===============================================================================
# 📚 1단계: 학습 데이터 준비 및 토크나이저 구현
# ===============================================================================

# 텍스트 파일에서 학습 데이터를 로드
def load_training_data(file_path='training_data.txt'):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f.readlines() if line.strip()]
        print(f"📖 {len(texts)}개의 학습 문장을 로드했습니다.")
        return texts
    except FileNotFoundError:
        print(f"⚠️ 파일 '{file_path}'을 찾을 수 없습니다.")
        print("프로그램을 종료합니다.")
        exit()

# 토크나이저 클래스 - 텍스트를 숫자로 변환하는 역할
class SimpleTokenizer:
    def __init__(self, texts):
        print("🔤 토크나이저 생성 중...")
        
        # 모든 텍스트 합치기
        all_text = " ".join(texts)
        words = re.findall(r'\S+', all_text)
        
        # 단어 빈도 계산
        word_counts = Counter(words)
        print(f"📊 고유 단어 수: {len(word_counts)}")
        print(f"📈 가장 빈번한 단어들: {dict(word_counts.most_common(5))}")
        
        # 특수 토큰 + 어휘 구성
        self.vocab = ['<PAD>', '<UNK>', '<START>', '<END>'] + list(word_counts.keys())
        self.word_to_id = {word: i for i, word in enumerate(self.vocab)}
        self.id_to_word = {i: word for i, word in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        
        print(f"✅ 전체 어휘 크기: {self.vocab_size}")
        print(f"📚 어휘 예시: {self.vocab[:8]}")
    
    # 텍스트를 토큰 ID로 변환 (인코딩 과정)
    def encode(self, text, show_process=False):
        words = re.findall(r'\S+', text)
        ids = [self.word_to_id.get(word, self.word_to_id['<UNK>']) for word in words]
        
        if show_process:
            print(f"🔤 인코딩: '{text}'")
            print(f"   → 단어: {words}")
            print(f"   → ID: {ids}")
        
        return ids
    
    # 토큰 ID를 텍스트로 변환 (디코딩 과정)
    def decode(self, ids, show_process=False):
        words = [self.id_to_word[id] for id in ids if id not in [0, 1]]  # PAD, UNK 제외
        result = " ".join(words)
        
        if show_process:
            print(f"🔤 디코딩: {ids} → '{result}'")
        
        return result

# ===============================================================================
# 🧠 2단계: Multi-Head Attention 구현
# ===============================================================================

# Multi-Head Attention 메커니즘 - Transformer의 핵심
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        print(f"🧠 Multi-Head Attention 초기화:")
        print(f"   - 모델 차원: {d_model}")
        print(f"   - 헤드 수: {n_heads}")
        print(f"   - 헤드당 차원: {self.d_k}")
        
        # Query, Key, Value 변환 행렬
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x, return_attention=False):
        batch_size, seq_len, d_model = x.size()
        
        # 1️⃣ Query, Key, Value 계산
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 2️⃣ Attention Score 계산 (유사도 측정)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        # 3️⃣ Causal Masking (미래 토큰을 보지 못하도록)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        scores.masked_fill_(mask, -float('inf'))
        
        # 4️⃣ Softmax로 확률 변환
        attention_weights = F.softmax(scores, dim=-1)
        
        # 5️⃣ Value와 가중합
        attention_output = torch.matmul(attention_weights, V)
        
        # 6️⃣ 헤드들을 다시 합치기
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        output = self.W_o(attention_output)
        
        if return_attention:
            return output, attention_weights
        return output

# ===============================================================================
# 🏗️ 3단계: Transformer 블록 구현
# ===============================================================================

# Transformer 블록 - Attention + Feed Forward + Residual Connection
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed Forward Network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, return_attention=False):
        # 1️⃣ Self-Attention with Residual Connection
        if return_attention:
            attn_output, attn_weights = self.attention(x, return_attention=True)
            x = self.norm1(x + self.dropout(attn_output))
            
            # 2️⃣ Feed Forward with Residual Connection
            ff_output = self.feed_forward(x)
            x = self.norm2(x + self.dropout(ff_output))
            
            return x, attn_weights
        else:
            attn_output = self.attention(x)
            x = self.norm1(x + self.dropout(attn_output))
            
            ff_output = self.feed_forward(x)
            x = self.norm2(x + self.dropout(ff_output))
            
            return x

# ===============================================================================
# 🤖 4단계: 언어모델 구현 (Decoder-only 구조)
# ===============================================================================

# 간단한 언어모델 - GPT 스타일의 Decoder-only 구조
class SimpleLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2, max_seq_len=50):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.n_heads = n_heads
        
        print(f"🤖 언어모델 초기화:")
        print(f"   - 어휘 크기: {vocab_size}")
        print(f"   - 모델 차원: {d_model}")
        print(f"   - 헤드 수: {n_heads}")
        print(f"   - 레이어 수: {n_layers}")
        
        # 임베딩 레이어들
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer 블록들
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_model*4) 
            for _ in range(n_layers)
        ])
        
        # 출력 레이어
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, return_attention=False):
        batch_size, seq_len = x.size()
        
        # 1️⃣ 토큰 임베딩 + 위치 임베딩
        positions = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1).to(x.device)
        embeddings = self.token_embedding(x) + self.position_embedding(positions)
        x = self.dropout(embeddings)
        
        # 2️⃣ Transformer 블록들 통과
        attention_weights_list = []
        for i, transformer_block in enumerate(self.transformer_blocks):
            if return_attention:
                x, attn_weights = transformer_block(x, return_attention=True)
                attention_weights_list.append(attn_weights)
            else:
                x = transformer_block(x)
        
        # 3️⃣ 다음 토큰 예측을 위한 로짓 계산
        logits = self.output_projection(x)
        
        if return_attention:
            return logits, attention_weights_list
        return logits

# ===============================================================================
# 📊 5단계: 데이터 준비 및 학습 함수들
# ===============================================================================

# 학습 데이터 생성 함수
def create_training_data(texts, tokenizer, max_length=20):
    print(f"📊 학습 데이터 생성 중... (최대 길이: {max_length})")
    
    input_ids = []
    target_ids = []
    
    for i, text in enumerate(texts):
        # 시작/끝 토큰 추가
        tokens = [tokenizer.word_to_id['<START>']] + tokenizer.encode(text) + [tokenizer.word_to_id['<END>']]
        
        # 길이 조절
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        
        # 패딩
        while len(tokens) < max_length:
            tokens.append(tokenizer.word_to_id['<PAD>'])
        
        # 입력과 타겟 준비 (다음 토큰 예측)
        input_ids.append(tokens[:-1])
        target_ids.append(tokens[1:])
        
        if i < 3:  # 처음 3개 예시만 출력
            print(f"   예시 {i+1}: {tokenizer.decode(tokens[1:-1])}")
    
    print(f"   ... 총 {len(texts)}개 문장 처리 완료")
    return torch.tensor(input_ids), torch.tensor(target_ids)

# 모델 학습 함수
def train_model(model, train_inputs, train_targets, epochs=50, lr=0.001):
    print(f"🚀 모델 학습 시작 (에폭: {epochs}, 학습률: {lr})")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # PAD 토큰 무시
    
    model.train()
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 순전파
        logits = model(train_inputs)
        
        # 손실 계산
        loss = criterion(logits.reshape(-1, model.vocab_size), train_targets.reshape(-1))
        
        # 역전파
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 10 == 0:
            print(f"   Epoch {epoch:3d}, Loss: {loss.item():.4f}")
    
    print("✅ 학습 완료!")
    return losses

# ===============================================================================
# 🎨 6단계: 텍스트 생성 및 시각화 함수들
# ===============================================================================

# 온도를 이용한 텍스트 생성 - 다양성 조절의 핵심
def generate_text(model, tokenizer, prompt, max_length=20, temperature=1.0, top_k=5, show_process=False):
    if show_process:
        print(f"🎨 텍스트 생성:")
        print(f"   프롬프트: '{prompt}' | 온도: {temperature} | Top-K: {top_k}")
    
    model.eval()
    
    # 프롬프트 토큰화
    tokens = [tokenizer.word_to_id['<START>']] + tokenizer.encode(prompt)
    
    with torch.no_grad():
        for step in range(max_length - len(tokens)):
            # 입력 준비 (최근 토큰들만 사용)
            input_tensor = torch.tensor([tokens[-19:]]).to(device)
            
            # 예측
            logits = model(input_tensor)
            
            # 온도 적용 - 확률 분포의 뾰족함 조절
            next_token_logits = logits[0, -1] / temperature
            
            # Top-K 샘플링으로 다양성 증가
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, k=min(top_k, len(next_token_logits)))
                probs = F.softmax(top_k_logits, dim=-1)
                next_token_idx = torch.multinomial(probs, 1).item()
                next_token = top_k_indices[next_token_idx].item()
            else:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
            
            # 종료 조건
            if next_token == tokenizer.word_to_id['<END>']:
                break
                
            tokens.append(next_token)
            
            # 첫 2스텝만 확률 분포 출력
            if show_process and step < 2:  
                top_probs, top_indices = torch.topk(probs, k=3)
                top_words = [tokenizer.id_to_word[idx.item()] for idx in top_indices]
                print(f"   스텝 {step+1}: {list(zip(top_words, [f'{p:.2f}' for p in top_probs]))}")
    
    result = tokenizer.decode(tokens[1:])  # START 토큰 제외
    if show_process:
        print(f"   → 결과: '{result}'")
    return result

# 어텐션 가중치 시각화 - 모델이 어디에 주목하는지 확인
def visualize_attention(model, tokenizer, text, layer_idx=0, head_idx=0):
    print(f"👁️ 어텐션 시각화: '{text}'")
    print(f"   → 레이어 {layer_idx+1}, 헤드 {head_idx+1}에서 각 단어가 다른 단어들에 주목하는 정도")
    
    model.eval()
    
    # 토큰화
    tokens = [tokenizer.word_to_id['<START>']] + tokenizer.encode(text)
    input_tensor = torch.tensor([tokens]).to(device)
    
    with torch.no_grad():
        logits, attention_weights_list = model(input_tensor, return_attention=True)
    
    # 어텐션 가중치 추출
    if layer_idx >= len(attention_weights_list):
        layer_idx = 0
    if head_idx >= model.n_heads:
        head_idx = 0
        
    attn = attention_weights_list[layer_idx][0, head_idx].cpu().numpy()
    
    # 토큰 라벨 준비
    token_labels = [tokenizer.id_to_word[id] for id in tokens]
    
    # 히트맵 그리기
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn, 
                xticklabels=token_labels, 
                yticklabels=token_labels,
                cmap='Blues', 
                annot=True, 
                fmt='.2f',
                cbar_kws={'label': 'Attention Weight'})
    
    plt.title(f'🧠 어텐션 히트맵 (레이어 {layer_idx+1}, 헤드 {head_idx+1})\n각 행은 해당 단어가 다른 단어들에 주목하는 정도', 
              fontsize=12, pad=20)
    plt.xlabel('참조되는 단어 (Key)')
    plt.ylabel('참조하는 단어 (Query)')
    plt.tight_layout()
    plt.show()
    
    # 핵심 어텐션 패턴만 출력
    print("📊 주요 어텐션 패턴 (상위 2개):")
    for i, query_token in enumerate(token_labels):
        top_attention = np.argsort(attn[i])[-2:][::-1]  # 상위 2개
        top_tokens = [token_labels[j] for j in top_attention]
        top_weights = [attn[i][j] for j in top_attention]
        print(f"   '{query_token}' → {top_tokens[0]}({top_weights[0]:.2f}), {top_tokens[1]}({top_weights[1]:.2f})")

# 온도별 생성 다양성 실험
def experiment_with_temperature(model, tokenizer, prompt="인공지능은"):
    print("🌡️ 온도별 생성 다양성 비교")
    print("=" * 50)
    
    temperatures = [0.2, 0.5, 1.0, 2.0]
    
    for temp in temperatures:
        print(f"\n🌡️ 온도 {temp} ({'안정적' if temp < 1 else '창의적'})")
        
        # 한 번만 생성하되 과정을 보여줌
        generated = generate_text(model, tokenizer, prompt, 
                                temperature=temp, max_length=12, show_process=True)
        print()

# 다양한 레이어/헤드의 어텐션 패턴 비교
def compare_attention_patterns(model, tokenizer, text):
    print("🔍 멀티헤드 어텐션 이해하기")
    print("=" * 50)
    print(f"입력: '{text}'")
    print("→ 여러 헤드가 서로 다른 관점에서 단어 간 관계를 학습")
    
    model.eval()
    tokens = [tokenizer.word_to_id['<START>']] + tokenizer.encode(text)
    input_tensor = torch.tensor([tokens]).to(device)
    
    with torch.no_grad():
        logits, attention_weights_list = model(input_tensor, return_attention=True)
    
    token_labels = [tokenizer.id_to_word[id] for id in tokens]
    
    # 2x2 그리드로 헤드 비교 (처음 4개 헤드만)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    patterns = [
        (0, 0, "레이어1-헤드1"),
        (0, 1, "레이어1-헤드2") if model.n_heads > 1 else (0, 0, "레이어1-헤드1"),
        (0, 2, "레이어1-헤드3") if model.n_heads > 2 else (0, 0, "레이어1-헤드1"),
        (0, 3, "레이어1-헤드4") if model.n_heads > 3 else (0, 0, "레이어1-헤드1")
    ]
    
    for i, (layer_idx, head_idx, title) in enumerate(patterns):
        layer_idx = min(layer_idx, len(attention_weights_list) - 1)
        head_idx = min(head_idx, model.n_heads - 1)
        
        attn = attention_weights_list[layer_idx][0, head_idx].cpu().numpy()
        
        im = axes[i].imshow(attn, cmap='Blues', aspect='auto')
        axes[i].set_title(title, fontsize=11)
        axes[i].set_xticks(range(len(token_labels)))
        axes[i].set_yticks(range(len(token_labels)))
        axes[i].set_xticklabels(token_labels, rotation=45, ha='right', fontsize=9)
        axes[i].set_yticklabels(token_labels, fontsize=9)
        
        # 핵심 값만 표시
        for y in range(len(token_labels)):
            for x in range(len(token_labels)):
                if attn[y, x] > 0.3:  # 높은 값만 표시
                    axes[i].text(x, y, f'{attn[y, x]:.2f}', 
                               ha='center', va='center', 
                               color='white', fontsize=8)
    
    plt.suptitle(f'🧠 멀티헤드 어텐션: 각 헤드의 서로 다른 관점\n"{text}"', fontsize=14, y=0.95)
    plt.tight_layout()
    plt.show()
    
    print("\n💡 멀티헤드 어텐션의 장점:")
    print("   • 각 헤드가 다른 종류의 관계를 학습 (문법적, 의미적 등)")
    print("   • 더 풍부하고 다양한 표현 학습 가능")
    print("   • 병렬 처리로 효율성 향상")

# 인코딩/임베딩 과정 시각화
def visualize_encoding_process(model, tokenizer, text):
    print("🔤 토크나이저와 임베딩 이해하기")
    print("=" * 50)
    
    # 간단한 예시들로 토크나이저 설명
    examples = [
        "안녕하세요",
        "파이썬 프로그래밍",
        "인공지능 기술",
        "좋은 하루",
        "컴퓨터 과학"
    ]
    
    print("1️⃣ 토크나이저 동작 예시:")
    for i, example in enumerate(examples):
        if i >= 5:  # 5개만 출력
            break
        encoded = tokenizer.encode(example, show_process=True)
        print()
    
    # 2단계: 임베딩 과정
    print("2️⃣ 임베딩 과정:")
    model.eval()
    tokens = tokenizer.encode(text)
    input_tensor = torch.tensor([tokens]).to(device)
    
    # 임베딩 추출
    token_embeddings = model.token_embedding(input_tensor)
    position_embeddings = model.position_embedding(torch.arange(len(tokens)).unsqueeze(0).to(device))
    final_embeddings = token_embeddings + position_embeddings
    
    print(f"   입력 텍스트: '{text}'")
    print(f"   토큰 수: {len(tokens)}")
    print(f"   임베딩 차원: {final_embeddings.shape[-1]}")
    print(f"   최종 임베딩 모양: {final_embeddings.shape}")
    
    # 간단한 임베딩 시각화
    embeddings_np = final_embeddings[0].detach().cpu().numpy()
    words = text.split()
    
    plt.figure(figsize=(10, 6))
    plt.imshow(embeddings_np.T, cmap='coolwarm', aspect='auto')
    plt.colorbar(label='임베딩 값')
    plt.title(f'🎯 임베딩 벡터: "{text}"\n각 단어가 {final_embeddings.shape[-1]}차원 벡터로 표현됨', fontsize=12)
    plt.xlabel('단어 위치')
    plt.ylabel('임베딩 차원')
    plt.xticks(range(len(words)), words, rotation=45)
    plt.tight_layout()
    plt.show()

# 어텐션 메커니즘 쉽게 이해하기
def analyze_attention_mechanism(model, tokenizer, text="안녕하세요 반갑습니다"):
    print("🔍 어텐션 메커니즘 쉽게 이해하기")
    print("=" * 50)
    print("💡 어텐션 = '각 단어가 다른 단어들을 얼마나 참고하는가?'")
    
    model.eval()
    tokens = [tokenizer.word_to_id['<START>']] + tokenizer.encode(text)
    input_tensor = torch.tensor([tokens]).to(device)
    
    with torch.no_grad():
        # 임베딩 추출
        embeddings = model.token_embedding(input_tensor) + model.position_embedding(
            torch.arange(len(tokens)).unsqueeze(0).to(device))
        
        # 첫 번째 어텐션 블록에서 Q, K, V 추출
        first_attention = model.transformer_blocks[0].attention
        Q = first_attention.W_q(embeddings)
        K = first_attention.W_k(embeddings)
        V = first_attention.W_v(embeddings)
        
        print(f"\n📝 입력: '{text}'")
        token_words = [tokenizer.id_to_word[id] for id in tokens]
        print(f"🔤 토큰들: {token_words}")
        
        print(f"\n🧮 어텐션 계산 단계:")
        print(f"   1. Query(Q): '누가 질문하는가?' - 크기: {Q.shape}")
        print(f"   2. Key(K): '누구를 참고할 수 있는가?' - 크기: {K.shape}")  
        print(f"   3. Value(V): '실제 정보 내용' - 크기: {V.shape}")
        
        # 어텐션 스코어 계산
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(first_attention.d_k)
        
        print(f"\n🎯 핵심: Causal Masking")
        print(f"   → 미래 단어는 볼 수 없음 (자기회귀 생성을 위해)")
        
        # 마스킹 적용
        mask = torch.triu(torch.ones(len(tokens), len(tokens)), diagonal=1).bool().to(device)
        scores_masked = scores.clone()
        scores_masked.masked_fill_(mask, -float('inf'))
        
        # Softmax 적용
        attention_weights = F.softmax(scores_masked, dim=-1)
        
        print(f"\n📊 어텐션 가중치 예시 (첫 번째 헤드):")
        attn_np = attention_weights[0, 0, :, :].cpu().numpy()
        for i, word in enumerate(token_words):
            # 자기 자신 제외하고 가장 높은 어텐션 찾기
            attn_row = attn_np[i].copy()
            attn_row[i] = 0  # 자기 자신 제외
            max_idx = np.argmax(attn_row)
            max_weight = attn_row[max_idx]
            if max_weight > 0.1:  # 의미있는 어텐션만
                print(f"   '{word}' → '{token_words[max_idx]}' ({max_weight:.2f})")
            else:
                print(f"   '{word}' → 주로 자기 자신에 집중")

# ===============================================================================
# 🚀 7단계: 메인 실행 코드
# ===============================================================================

print("🚀 Transformer 언어모델 학습 시작!")
print("=" * 80)

# 학습 데이터 로드
sample_texts = load_training_data()

# 토크나이저 생성
tokenizer = SimpleTokenizer(sample_texts)
print()

# 인코딩/디코딩 예시
print("🔤 인코딩/디코딩 예시:")
sample_text = "안녕하세요 좋은 하루"
encoded = tokenizer.encode(sample_text, show_process=True)
decoded = tokenizer.decode(encoded, show_process=True)
print()

# 학습 데이터 준비
train_inputs, train_targets = create_training_data(sample_texts, tokenizer)
print(f"📊 학습 데이터 크기: {train_inputs.shape}")
print()

# 모델 생성
model = SimpleLM(tokenizer.vocab_size, d_model=64, n_heads=4, n_layers=2)
model = model.to(device)

param_count = sum(p.numel() for p in model.parameters())
print(f"🔧 모델 파라미터 수: {param_count:,}")
print()

# 모델 학습
losses = train_model(model, train_inputs.to(device), train_targets.to(device), epochs=100)

# 학습 곡선 시각화
plt.figure(figsize=(12, 6))
plt.plot(losses, 'b-', linewidth=2)
plt.title('📈 학습 손실 곡선', fontsize=14)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)
plt.show()

print("\n✅ 기본 학습 완료! 이제 다양한 실험을 시작합니다!")
print("=" * 80)

# ===============================================================================
# 🎯 8단계: 다양한 실험들
# ===============================================================================

# 1. 기본 텍스트 생성 테스트
print("\n🎨 기본 텍스트 생성 테스트")
print("=" * 50)

test_prompts = ["안녕하세요", "파이썬을", "인공지능은"]

for prompt in test_prompts:
    print(f"\n📝 프롬프트: '{prompt}'")
    generated = generate_text(model, tokenizer, prompt, max_length=10, temperature=1.0)
    print(f"   → {generated}")

# 2. 온도별 생성 다양성 실험
print("\n")
experiment_with_temperature(model, tokenizer, "컴퓨터는")

# 3. 토크나이저와 임베딩 이해
print("\n")
visualize_encoding_process(model, tokenizer, "인공지능 기술")

# 4. 어텐션 메커니즘 이해
print("\n")
analyze_attention_mechanism(model, tokenizer, "안녕하세요 반갑습니다")

# 5. 어텐션 가중치 시각화
print("\n")
visualize_attention(model, tokenizer, "좋은 하루", layer_idx=0, head_idx=0)

# 6. 멀티헤드 어텐션 이해
print("\n")
compare_attention_patterns(model, tokenizer, "파이썬 프로그래밍")

# ===============================================================================
# 🎯 9단계: 간소화된 추가 실험들
# ===============================================================================

print("\n" + "="*60)
print("🔬 핵심 개념 정리")
print("="*60)

print("\n🎯 학습한 핵심 개념:")
print("   1. 토크나이저: 텍스트 ↔ 숫자 변환의 다리 역할")
print("   2. 임베딩: 단어를 컴퓨터가 이해할 수 있는 벡터로 변환")
print("   3. 어텐션: 각 단어가 다른 단어들과의 관계를 학습")
print("   4. 멀티헤드: 여러 관점에서 동시에 관계를 파악")
print("   5. 온도: 생성의 창의성과 안정성을 조절하는 핵심 파라미터")

print("\n🛠️ 실험해볼 수 있는 함수들:")
print("   • generate_text(model, tokenizer, '프롬프트', temperature=1.5)")
print("   • visualize_attention(model, tokenizer, '텍스트')")
print("   • experiment_with_temperature(model, tokenizer, '시작말')")

print("\n💡 추가 실험 아이디어:")
print("   1. 다른 온도값으로 창의성 변화 관찰")
print("   2. 긴 텍스트로 어텐션 패턴 분석")  
print("   3. 새로운 프롬프트로 생성 품질 테스트")
print("   4. 다른 레이어/헤드의 어텐션 비교")

print("\n✨ 축하합니다! Transformer 언어모델의 핵심을 이해했습니다!")
print("="*60)
