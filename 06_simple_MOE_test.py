# MoE 실습 - 파라미터 활성화 및 가중합 상세 분석
# 3개 분야 Expert: '수학', '과학', '프로그래밍'

# !pip install --upgrade sentence-transformers huggingface-hub

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import numpy as np
import time

# 전문가 신경망 (각 도메인에 특화)
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, domain_name):
        super(Expert, self).__init__()
        self.domain_name = domain_name
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)
        self.activation_count = 0
    
    def forward(self, x):
        self.activation_count += 1
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    # 전문가의 파라미터 수 계산
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# Gating Network (전문가 선택)
class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_experts)
        self.dropout = nn.Dropout(0.2)
        
        # 로드 밸런싱을 위한 보조 손실
        self.load_balance_loss = 0
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        gates = F.softmax(self.fc3(x), dim=-1)
        
        # 로드 밸런스 손실 계산 (전문가 균등 사용 유도)
        self.load_balance_loss = self._compute_load_balance_loss(gates)
        
        return gates
    
    # 전문가 사용 균형을 위한 손실 계산
    def _compute_load_balance_loss(self, gates):
        avg_gates = gates.mean(dim=0)
        uniform_dist = torch.ones_like(avg_gates) / len(avg_gates)
        return F.mse_loss(avg_gates, uniform_dist)

# 전문가 응답 생성기
class ExpertResponseGenerator:
    def __init__(self):
        self.response_templates = {
            'math': [
                "수학 전문가: {}에 대해 단계별 풀이를 제공합니다.",
                "수학적 접근: {}를 공식과 계산으로 해결합니다.",
                "수학 분석: {}의 수치적 해법을 제시합니다."
            ],
            'history': [
                "역사 전문가: {}의 역사적 배경을 설명합니다.",
                "역사적 분석: {}를 시대적 맥락으로 접근합니다.",
                "역사 해설: {}의 역사적 의미를 설명합니다."
            ],
            'software': [
                "소프트웨어 전문가: {}를 코드로 구현합니다.",
                "기술적 해결: {}의 알고리즘을 설계합니다.",
                "소프트웨어 솔루션: {}를 효율적으로 처리합니다."
            ]
        }
    
    def generate_response(self, expert_idx, question):
        domain_map = {0: 'math', 1: 'history', 2: 'software'}
        domain = domain_map[expert_idx]
        template = np.random.choice(self.response_templates[domain])
        return template.format(f'"{question}"')

# MoE 모델 (Mixture of Experts)
class TextMoE(nn.Module):
    def __init__(self, embedding_dim, num_experts, hidden_dim, output_dim, top_k=1):
        super(TextMoE, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 텍스트를 벡터로 변환하는 인코더
        self.text_encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        
        # 전문가들 생성
        self.expert_names = ['수학', '역사', '소프트웨어']
        self.experts = nn.ModuleList([
            Expert(embedding_dim, hidden_dim, output_dim, name) 
            for name in self.expert_names
        ])
        
        # 게이팅 네트워크
        self.gate = GatingNetwork(embedding_dim, num_experts)
        
        # 응답 생성기
        self.response_generator = ExpertResponseGenerator()
        
        # 전문가 사용 통계
        self.expert_usage_stats = {i: 0 for i in range(num_experts)}
    
    # 파라미터 수 계산
    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        gate_params = sum(p.numel() for p in self.gate.parameters() if p.requires_grad)
        expert_params = self.experts[0].count_parameters()
        
        return {
            'total': total_params,  # 모델 전체 파라미터 수
            'gate': gate_params,  # 게이팅 네트워크 파라미터 수
            'single_expert': expert_params,  # 전문가 1개의 파라미터 수
            'all_experts': expert_params * self.num_experts  # 모든 전문가의 파라미터 수
        }
    
    # 활성화 상태 통계 계산
    def get_activation_stats(self):
        params = self.count_parameters()
        active_params = params['gate'] + params['single_expert'] * self.top_k  # 활성화된 파라미터
        inactive_params = params['single_expert'] * (self.num_experts - self.top_k)  # 비활성화된 파라미터
        usage_ratio = (active_params / params['total']) * 100  # 사용률
        
        return {
            'total_params': params['total'],  # 전체 파라미터
            'gate_params': params['gate'],  # 게이트 파라미터
            'single_expert_params': params['single_expert'],  # 전문가 1개 파라미터
            'active_expert_count': self.top_k,  # 활성화된 전문가 수
            'inactive_expert_count': self.num_experts - self.top_k,  # 비활성화된 전문가 수
            'active_params': active_params,  # 활성 파라미터
            'inactive_params': inactive_params,  # 비활성 파라미터
            'usage_ratio': usage_ratio,  # 사용률
            'savings_ratio': 100 - usage_ratio  # 절감률
        }
    
    # 순전파 (학습 시 사용)
    def forward(self, texts):
        # 텍스트를 임베딩으로 변환
        with torch.no_grad():
            embeddings = self.text_encoder.encode(texts, convert_to_tensor=True)
        
        embeddings = embeddings.to(self.device).clone().detach().requires_grad_(True)
        
        # 게이팅 네트워크로 전문가 선택 확률 계산
        gate_scores = self.gate(embeddings)
        
        # Top-K 전문가 선택
        top_k_scores, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        
        # 선택된 전문가들의 확률 합이 1이 되도록 정규화
        top_k_scores = top_k_scores / top_k_scores.sum(dim=-1, keepdim=True)
        
        # 출력 초기화
        batch_size = embeddings.shape[0]
        output = torch.zeros(batch_size, self.experts[0].fc2.out_features).to(self.device)
        
        # 선택된 전문가만 활성화하여 출력 계산 (Sparse Activation)
        for i in range(self.top_k):
            for batch_idx in range(batch_size):
                expert_idx = top_k_indices[batch_idx, i].item()
                expert_weight = top_k_scores[batch_idx, i].unsqueeze(0)
                expert_output = self.experts[expert_idx](embeddings[batch_idx:batch_idx+1])
                output[batch_idx] += expert_weight * expert_output.squeeze(0)
                self.expert_usage_stats[expert_idx] += 1
        
        return output, gate_scores
    
    # 텍스트 응답 생성 (추론 시 사용)
    def generate_text_response(self, question, verbose=True):
        if verbose:
            print("\n" + "="*80)
            print("MoE 추론 과정 상세 분석")
            print("="*80)
            print(f"\n질문: {question}")
            print("-"*80)
        
        start_time = time.time()
        
        with torch.no_grad():
            # [1단계] 임베딩 생성
            if verbose:
                print("\n[1단계] 질문 임베딩 생성")
                embed_start = time.time()
            
            embedding = self.text_encoder.encode([question], convert_to_tensor=True)
            embedding = embedding.to(self.device)
            
            if verbose:
                embed_time = time.time() - embed_start
                print(f"  - 임베딩 차원: {embedding.shape}")
                print(f"  - 생성 시간: {embed_time*1000:.2f}ms")
            
            # [2단계] Gating Network 결정
            if verbose:
                print("\n[2단계] Gating Network 라우팅 결정")
                gate_start = time.time()
            
            gate_scores = self.gate(embedding)
            
            if verbose:
                gate_time = time.time() - gate_start
                print(f"  - 처리 시간: {gate_time*1000:.2f}ms")
                print(f"  - 라우팅 확률:")
                for idx, score in enumerate(gate_scores[0]):
                    print(f"      전문가 {idx} ({self.expert_names[idx]}): {score.item():.4f}")
            
            # Top-K 선택
            top_k_scores, top_k_indices = torch.topk(gate_scores[0], self.top_k)
            top_k_scores_normalized = top_k_scores / top_k_scores.sum()
            
            if verbose:
                print(f"\n  - Top-{self.top_k} 선택:")
                for k in range(self.top_k):
                    idx = top_k_indices[k].item()
                    score = top_k_scores[k].item()
                    norm_score = top_k_scores_normalized[k].item()
                    print(f"      {k+1}위: 전문가 {idx} ({self.expert_names[idx]}) - "
                          f"원점수: {score:.4f}, 정규화: {norm_score:.4f}")
            
            # [3단계] 파라미터 활성화 분석
            if verbose:
                print("\n[3단계] 파라미터 활성화 분석 (MoE의 핵심 효율성)")
                stats = self.get_activation_stats()
                
                print(f"\n  전체 모델 구조:")
                print(f"    - 전체 파라미터: {stats['total_params']:,}개")
                print(f"    - Gating Network: {stats['gate_params']:,}개")
                print(f"    - 단일 전문가: {stats['single_expert_params']:,}개")
                print(f"    - 전체 전문가 ({self.num_experts}개): "
                      f"{stats['single_expert_params'] * self.num_experts:,}개")
                
                print(f"\n  현재 활성화 상태 (Top-{self.top_k}):")
                print(f"    - 활성화된 전문가: {stats['active_expert_count']}개")
                print(f"    - 비활성화된 전문가: {stats['inactive_expert_count']}개")
                
                print(f"\n  파라미터 사용 현황:")
                print(f"    - 활성 파라미터: {stats['active_params']:,}개")
                print(f"      = Gate ({stats['gate_params']:,}) + "
                      f"전문가 {stats['active_expert_count']}개 "
                      f"({stats['single_expert_params']:,} x {stats['active_expert_count']})")
                print(f"    - 비활성 파라미터: {stats['inactive_params']:,}개")
                print(f"      = 전문가 {stats['inactive_expert_count']}개 "
                      f"({stats['single_expert_params']:,} x {stats['inactive_expert_count']})")
                
                print(f"\n  효율성 지표:")
                print(f"    - 파라미터 사용률: {stats['usage_ratio']:.1f}%")
                print(f"    - 연산 절감률: {stats['savings_ratio']:.1f}%")
                
                print(f"\n  Dense 모델과 비교:")
                dense_params = stats['gate_params'] + stats['single_expert_params'] * self.num_experts
                print(f"    - Dense 모델 (모든 전문가 활성): {dense_params:,}개")
                print(f"    - MoE 모델 (Top-{self.top_k} 활성): {stats['active_params']:,}개")
                print(f"    - 절감량: {dense_params - stats['active_params']:,}개 "
                      f"({stats['savings_ratio']:.1f}%)")
            
            # [4단계] 선택된 전문가들의 추론 및 가중합
            if verbose:
                print("\n[4단계] 선택된 전문가들의 추론 및 가중합")
                expert_start = time.time()
            
            expert_outputs = []
            expert_infos = []
            
            for k in range(self.top_k):
                expert_idx = top_k_indices[k].item()
                expert_weight = top_k_scores_normalized[k].item()
                
                expert_output = self.experts[expert_idx](embedding)
                expert_outputs.append(expert_output)
                expert_infos.append((expert_idx, expert_weight))
                
                if verbose:
                    print(f"\n  전문가 {expert_idx} ({self.expert_names[expert_idx]}) 실행:")
                    print(f"    - 가중치: {expert_weight:.4f}")
                    print(f"    - 출력 shape: {expert_output.shape}")
                    print(f"    - 출력 샘플 (처음 5개): {expert_output[0, :5].cpu().numpy()}")
            
            if verbose:
                print(f"\n  가중합 계산:")
                print(f"    최종_출력 = ", end="")
                terms = []
                for idx, weight in expert_infos:
                    terms.append(f"({weight:.3f} x 전문가{idx})")
                print(" + ".join(terms))
            
            # 가중합 수행
            final_output = torch.zeros_like(expert_outputs[0])
            for k, (expert_idx, expert_weight) in enumerate(expert_infos):
                contribution = expert_weight * expert_outputs[k]
                final_output += contribution
                
                if verbose and k == 0:
                    print(f"\n    가중합 과정:")
                    print(f"      단계 {k+1}: {expert_weight:.4f} x 전문가{expert_idx} = "
                          f"{contribution[0, :3].cpu().numpy()}")
            
            if verbose:
                expert_time = time.time() - expert_start
                print(f"\n  처리 시간: {expert_time*1000:.2f}ms")
                print(f"  최종 출력 shape: {final_output.shape}")
                print(f"  최종 출력 샘플 (처음 5개): {final_output[0, :5].cpu().numpy()}")
            
            selected_expert = top_k_indices[0].item()
            
            # [5단계] 최종 응답 생성
            if verbose:
                print("\n[5단계] 최종 응답 생성")
            
            response = self.response_generator.generate_response(selected_expert, question)
            
            if verbose:
                print(f"  - 응답: {response}")
        
        total_time = time.time() - start_time
        stats = self.get_activation_stats()
        
        if verbose:
            print("\n" + "="*80)
            print("[성능 요약]")
            print("="*80)
            print(f"전체 처리 시간: {total_time*1000:.2f}ms")
            print(f"파라미터 사용률: {stats['usage_ratio']:.1f}%")
            print(f"연산 절감률: {stats['savings_ratio']:.1f}%")
            print(f"활성 파라미터: {stats['active_params']:,} / {stats['total_params']:,}")
            print("="*80 + "\n")
        
        return response, selected_expert, gate_scores[0]

# 모델 학습
def train_model():
    # 하이퍼파라미터 설정
    embedding_dim = 384
    num_experts = 3
    hidden_dim = 512
    output_dim = 3
    epochs = 400
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 장치: {device}\n")
    
    # 모델 초기화
    model = TextMoE(embedding_dim, num_experts, hidden_dim, output_dim, top_k=1)
    model = model.to(device)
    
    params = model.count_parameters()
    stats = model.get_activation_stats()
    
    print("="*80)
    print("MoE 모델 구조 및 효율성 분석")
    print("="*80)
    print("\n[파라미터 구성]")
    print(f"  전체 파라미터: {params['total']:,}개")
    print(f"    - Gating Network: {params['gate']:,}개")
    print(f"    - 단일 전문가: {params['single_expert']:,}개")
    print(f"    - 전체 전문가: {params['all_experts']:,}개 ({num_experts}개)")
    
    print(f"\n[MoE 효율성 (Top-{model.top_k})]")
    print(f"  활성 파라미터: {stats['active_params']:,}개 ({stats['usage_ratio']:.1f}%)")
    print(f"  비활성 파라미터: {stats['inactive_params']:,}개 ({stats['savings_ratio']:.1f}%)")
    print(f"  연산 절감: {stats['savings_ratio']:.1f}%")
    
    print(f"\n[로드 밸런싱]")
    print(f"  목적: 학습 초기부터 모든 전문가가 균등하게 학습되도록 조정: 3개 전문가 각각 약 33%씩 학습")
    print(f"  효과: 특정 전문가만 과도하게 학습되는 것을 방지")
    print(f"  방법: 배치마다 전문가 선택 확률 분포를 균등 분포에 가깝게 유도")
    print("="*80 + "\n")
    
    # 옵티마이저 설정
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    
    print("학습 시작...\n")
    
    # 학습 루프
    for epoch in range(epochs):
        texts, labels, domains = generate_training_data(100)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs, gate_scores = model(texts)
        
        # 메인 손실 (예측 오차)
        loss = F.mse_loss(outputs, labels)
        
        # 전문화 손실 (올바른 전문가 선택 유도)
        domain_map = {'math': 0, 'history': 1, 'software': 2}
        specialization_loss = 0
        for i, domain in enumerate(domains):
            target_expert = domain_map[domain]
            specialization_loss += -torch.log(gate_scores[i, target_expert] + 1e-8)
        specialization_loss = specialization_loss / len(domains)
        
        # 로드 밸런스 손실 (전문가 균등 사용 유도)
        load_balance_loss = model.gate.load_balance_loss
        
        # 전체 손실 계산 및 역전파
        total_loss = loss + 0.5 * specialization_loss + 0.1 * load_balance_loss
        
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Main Loss: {loss.item():.4f}")
            print(f"  Specialization Loss: {specialization_loss.item():.4f}")
            print(f"  Load Balance Loss: {load_balance_loss.item():.4f}")
            print(f"  Total Loss: {total_loss.item():.4f}\n")
    
    print("학습 완료!\n")
    print("="*80)
    print("전문가 사용 통계 (로드 밸런싱 결과)")
    print("="*80)
  
    total_usage = sum(model.expert_usage_stats.values())
  
    for idx, count in model.expert_usage_stats.items():
        ratio = (count / total_usage * 100) if total_usage > 0 else 0
        print(f"전문가 {idx} ({model.expert_names[idx]}): {count}회 ({ratio:.1f}%)")
      
    print("="*80 + "\n")
    
    return model

# 학습 데이터 생성 (도메인별 질문 샘플)
def generate_training_data(num_samples=50):
    domains = {
        'math': [
            "방정식을 풀어주세요", "이차방정식 풀이", "미분을 계산해주세요", 
            "적분 문제 풀이", "삼각함수 값 계산", "행렬 곱셈을 계산",
            "극한값을 구해주세요", "도함수 구하기", "함수 그래프 그리기",
            "수열의 일반항", "확률을 계산", "통계 평균 구하기",
            "수학 공식 유도", "기하학 문제", "대수 문제 풀이",
            "로그 계산", "지수 함수", "벡터 내적 계산"
        ],
        'history': [
            "프랑스 혁명 설명", "고려 시대 역사", "제1차 세계대전의 원인",
            "조선 왕조 건국 과정", "르네상스 시대", "냉전의 시작과 끝",
            "삼국시대 통일 과정", "로마 제국 멸망", "산업혁명의 영향",
            "대항해 시대", "독립운동가들", "고대 이집트 문명",
            "한국전쟁 배경", "명나라와 청나라", "비잔틴 제국",
            "임진왜란 경과", "몽골 제국", "십자군 전쟁"
        ],
        'software': [
            "알고리즘 설명해주세요", "파이썬 코드 작성 방법", "버그를 수정",
            "자료구조 설명", "시간복잡도 분석", "디버깅 방법 알려주세요",
            "리스트 정렬 구현", "이진 탐색 구현", "정렬 알고리즘",
            "재귀함수 작성", "객체지향 프로그래밍", "API 설계 방법",
            "데이터베이스 쿼리", "웹 개발 방법", "React 컴포넌트",
            "자바스크립트 함수", "SQL 쿼리 작성", "Git 사용법"
        ]
    }
    
    texts = []
    labels = []
    domain_names = []
    
    for _ in range(num_samples):
        domain_list = list(domains.keys())
        domain_idx = np.random.randint(len(domain_list))
        domain = domain_list[domain_idx]
        
        question = np.random.choice(domains[domain])
        
        # 원-핫 인코딩 레이블
        label = np.zeros(len(domain_list))
        label[domain_idx] = 1.0
        
        texts.append(question)
        labels.append(label)
        domain_names.append(domain)
    
    return texts, torch.FloatTensor(labels), domain_names

# 모델 테스트
def test_model(model):
    print("\n" + "="*80)
    print("MoE 모델 테스트")
    print("="*80)
    
    test_cases = [
        ("이차방정식 x^2 + 5x + 6 = 0을 풀어주세요", "수학", 0),
        ("조선시대 세종대왕의 업적을 설명해주세요", "역사", 1),
        ("파이썬 리스트 정렬 알고리즘을 구현해주세요", "소프트웨어", 2),
        ("미분 계산 방법을 알려주세요", "수학", 0),
        ("프랑스 혁명의 원인은 무엇인가요", "역사", 1),
        ("재귀 함수로 팩토리얼을 계산하는 코드", "소프트웨어", 2),
        ("삼각함수 sin과 cos의 관계", "수학", 0),
        ("제2차 세계대전의 배경을 설명해주세요", "역사", 1),
        ("이진 탐색 트리 구현 방법", "소프트웨어", 2),
        ("행렬의 역행렬 구하는 방법", "수학", 0),
        ("고려시대 무신정변에 대해 알려주세요", "역사", 1),
        ("React 컴포넌트 작성 방법", "소프트웨어", 2),
    ]
    
    print("\n첫 번째 질문만 상세 출력, 나머지는 요약\n")
    
    correct = 0
    for idx, (question, expected_domain, expected_idx) in enumerate(test_cases):
        verbose = (idx == 0)
        
        response, selected_idx, gate_probs = model.generate_text_response(
            question, verbose=verbose
        )
        
        is_correct = selected_idx == expected_idx
        if is_correct:
            correct += 1
        
        if not verbose:
            print(f"\n질문: {question}")
            print(f"선택: 전문가 {selected_idx} ({model.expert_names[selected_idx]}) "
                  f"[{'정답' if is_correct else '오답'}]")
            print(f"확률: ", end="")
            for i, prob in enumerate(gate_probs):
                print(f"{model.expert_names[i]}={prob.item():.3f} ", end="")
            print(f"\n응답: {response}")
    
    print(f"\n정확도: {correct}/{len(test_cases)} = "
          f"{correct/len(test_cases)*100:.1f}%\n")

if __name__ == "__main__":
    model = train_model()
    test_model(model)
