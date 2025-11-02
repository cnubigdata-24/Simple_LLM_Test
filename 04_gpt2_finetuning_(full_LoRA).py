# 콜센터 고객 서비스 AI 자동 응답 시스템 파인튜닝 실습 (full fine tuning, LoRA fine tuning)

!pip install -q transformers datasets torch

#-------------------------------------------------

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("시스템 정보")
print("="*60)
print(f"PyTorch 버전: {torch.__version__}")
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB")
else:
    print("CPU 모드 - 학습이 다소 느릴 수 있습니다")

#-------------------------------------------------

# 토크나이저: GPT-2는 Byte-Pair Encoding (BPE) 사용
# 어휘사전: 50,257개
# Causal Language Model: 왼쪽에서 오른쪽으로 순차적으로 텍스트를 생성
# 대화형 AI, 텍스트 완성 등에 적합한 구조입니다

# Part 1: 기본 파인튜닝 #######################################################################################
model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(model_name)

# GPT-2는 pad_token 없음 => 배치 처리를 위해 문장 길이를 맞춰야 하므로 eos_token을 pad_token으로 사용
tokenizer.pad_token = tokenizer.eos_token
base_model.config.pad_token_id = tokenizer.eos_token_id

print(f"\n모델: {model_name}")
print(f"전체 파라미터 수: {base_model.num_parameters():,}")
print(f"토큰 어휘 크기: {len(tokenizer):,}")
print(f"최대 시퀀스 길이: {tokenizer.model_max_length}")

#-------------------------------------------------

# 파인튜닝용 콜센터 문의-응답 데이터
training_data = {
    "text": [
        # 배송 (2개)
        "질문: 주문한 상품이 언제 도착하나요?\n답변: 일반 배송은 주문 후 2-3일 소요되며, 빠른 배송은 익일 배송입니다. 주문번호로 배송 현황을 확인하실 수 있습니다.",
        "질문: 배송 조회는 어떻게 하나요?\n답변: 마이페이지에서 주문 내역을 클릭하시면 운송장 번호로 실시간 배송 추적이 가능합니다. 또는 고객센터로 주문번호를 알려주시면 확인해드립니다.",

        # 반품/교환 (2개)
        "질문: 반품하고 싶은데 어떻게 하나요?\n답변: 상품 수령 후 7일 이내 반품 가능합니다. 마이페이지에서 반품 신청하시거나 고객센터로 연락 주시면 수거 접수해드립니다.",
        "질문: 사이즈가 안 맞아서 교환하고 싶어요.\n답변: 상품 수령 후 7일 이내 교환 가능합니다. 마이페이지에서 교환 신청 시 원하시는 사이즈를 선택해주시면 상품 회수 후 새 제품을 발송해드립니다.",

        # 결제/환불 (2개)
        "질문: 결제가 두 번 처리된 것 같아요.\n답변: 중복 결제 확인을 위해 주문번호와 결제 내역을 고객센터로 알려주시면 확인 후 즉시 환불 처리해드리겠습니다. 환불은 3-5 영업일 소요됩니다.",
        "질문: 환불은 언제 되나요?\n답변: 반품 상품 확인 후 3-5 영업일 이내 결제하신 수단으로 환불됩니다. 카드 결제는 카드사 일정에 따라 추가 시일이 소요될 수 있습니다.",
    ]
}

# Hugging Face Dataset 형식으로 변환, transformers 라이브러리와 호환
dataset = Dataset.from_dict(training_data)

print(f"학습 데이터 개수: {len(dataset)}개")
print(f"\n[샘플 데이터 1]")
print(dataset[0]['text'])
print(f"\n[샘플 데이터 2]")
print(dataset[3]['text'])

#-------------------------------------------------

# 토큰화: 텍스트를 토큰(숫자)으로 변환하는 과정
def tokenize_function(examples):

    return tokenizer(
        examples["text"],
        padding="max_length", #모든 시퀀스를 같은 길이로 맞춤, 짧은 문장은 pad_token으로 채우고, 긴 문장은 자름
        truncation=True, #max_length보다 긴 문장은 잘라냄
        max_length=150 # 최대 150개 토큰까지만 사용
    )

print("데이터 토큰화 중...")

# 전체 데이터셋에 토큰화 적용
tokenized_dataset = dataset.map(
    tokenize_function,
    remove_columns=["text"], # 원본 텍스트 컬럼은 제거하고 토큰만 남김
    batched=True # 여러 샘플을 한 번에 처리하여 속도 향상
)

print(f"완료! 토큰화된 데이터 수: {tokenized_dataset}")
print(f"\n첫 번째 샘플의 토큰 개수: {len(tokenized_dataset[0]['input_ids'])}개")
print(f"첫 번째 샘플의 토큰 예시 (처음 20개): {tokenized_dataset[0]['input_ids'][:20]}")

#-------------------------------------------------

# 데이터 콜레이터 설정: 배치 단위로 데이터를 묶어주는 역할
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False # GPT-2는 Causal LM, Masked Language Modeling을 사용하지 않음 (MLM은 BERT 같은 양방향 모델에서 사용)
)

print("데이터 콜레이터 설정 완료")

#-------------------------------------------------

# 기본 파인튜닝 실행

# 하이퍼 파라미터 설정
training_args = TrainingArguments(
    output_dir="./callcenter_basic_model", # 학습 결과 저장 디렉토리

    num_train_epochs=50,

    per_device_train_batch_size=2, # 한 번에 몇 개씩 학습할지 (2로 설정 => 6개 데이터를 3번에 나눠서 학습)
    learning_rate=5e-5, # 5e-5 = 0.00005는 파인튜닝에 적합한 작은 값

    logging_steps=2, # 2 step마다 loss 값 출력
    save_strategy="no", # 체크포인트 저장 X
    fp16=torch.cuda.is_available(), # fp16: 16비트 연산 사용 (GPU에서만 가능 => 메모리 사용량 절반, 학습속도 2배)

    report_to="none" # 학습 로그 기록 위치
)

# 학습 객체
trainer = Trainer(
    model=base_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
)

print("="*60)
print("기본 파인튜닝 시작")
print("="*60)
print(f"데이터 개수: {len(dataset)}개")
print(f"Epochs: {training_args.num_train_epochs}")
print(f"Batch Size: {training_args.per_device_train_batch_size}")
print(f"Learning Rate: {training_args.learning_rate}")
print(f"총 학습 스텝: {len(dataset) // training_args.per_device_train_batch_size * training_args.num_train_epochs}")
print("\n학습 중...\n")

# 실제 학습 시작
# 모델의 모든 파라미터(약 124M개)가 업데이트됩니다
trainer.train()

print("\n파인튜닝 완료!")

#-------------------------------------------------

# 파인튜닝 모델 테스트

# 텍스트 생성 함수
def generate_response(model, prompt, max_tokens=100):

    # 프롬프트를 토큰으로 변환, 모델이 있는 디바이스(GPU/CPU)로 이동
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 텍스트 생성
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.7,
        do_sample=True, # 확률 기반 샘플링 사용 여부
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id #패딩 토큰 ID 지정
    )

    # 생성된 토큰을 텍스트로 변환, skip_special_tokens=True: <pad>, <eos> 같은 특수 토큰 제거
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 파인튜닝에 학습한 질문으로 테스트
test_prompts = [
    "질문: 주문한 상품이 언제 도착하나요?\n답변:",
    "질문: 반품하고 싶은데 어떻게 하나요?\n답변:",
    "질문: 결제가 두 번 처리된 것 같아요.\n답변:",
]

print("="*60)
print("기본 파인튜닝 모델 테스트 결과")
print("="*60)

for i, prompt in enumerate(test_prompts, 1):
    print(f"\n[테스트 {i}]")
    result = generate_response(base_model, prompt)
    print(result)
    print("-"*60)
    
#-------------------------------------------------

# Part 2: LoRA 파인튜닝 #######################################################################################
!pip install -q peft

#-------------------------------------------------

# LoRA 모델 준비
from peft import get_peft_model, LoraConfig, TaskType

print("LoRA용 새 모델 로딩 중...")
lora_base_model = AutoModelForCausalLM.from_pretrained(model_name)
lora_base_model.config.pad_token_id = tokenizer.eos_token_id

# LoRA 설정
lora_config = LoraConfig(
    r=8,  # rank (어댑터의 차원),  낮을수록 파라미터 적음, 8은 일반적으로 좋은 성능
    lora_alpha=32, # LoRA scaling factor, 학습된 어댑터의 영향력 조절, 보통 r의 2-4배
    # target_modules: LoRA를 적용할 레이어
    target_modules=["c_attn"], # LoRA를 적용할 레이어, "c_attn"은 GPT-2의 attention 레이어
    lora_dropout=0.1,
    bias="none", # 바이어스 파라미터 학습 여부
    task_type=TaskType.CAUSAL_LM # Task 유형 => CAUSAL_LM = 다음 토큰 예측 (GPT 스타일)
)

# 원본 모델에 LoRA 어댑터 추가: 원본 파라미터는 고정(frozen)되고 어댑터만 학습
lora_model = get_peft_model(lora_base_model, lora_config)

print("\n" + "="*60)
print("LoRA 모델 파라미터 정보")
print("="*60)

# 학습 가능한 파라미터 비율 출력
lora_model.print_trainable_parameters()

#-------------------------------------------------

# LoRA 파인튜닝 실행

# LoRA 학습 설정
lora_training_args = TrainingArguments(
    output_dir="./callcenter_lora_model",

    num_train_epochs=100,   # LoRA는 파라미터가 적어서 더 많은 epoch 가능
    per_device_train_batch_size=2,

    learning_rate=3e-4, # LoRA는 보통 더 큰 학습률 사용 (3e-4 = 0.0003)
    logging_steps=3,
    save_strategy="no",
    fp16=torch.cuda.is_available(),
    report_to="none"
)

# LoRA Trainer
lora_trainer = Trainer(
    model=lora_model,
    args=lora_training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
)

print("="*60)
print("LoRA 파인튜닝 시작")
print("="*60)

print(f"Epochs: {lora_training_args.num_train_epochs}")
print(f"Learning Rate: {lora_training_args.learning_rate}")

print("\n학습 중...\n")

lora_trainer.train()

print("\nLoRA 파인튜닝 완료!")

#-------------------------------------------------

# LoRA 모델 테스트
print("="*60)
print("LoRA 파인튜닝 모델 테스트 결과")
print("="*60)

for i, prompt in enumerate(test_prompts, 1):
    print(f"\n[테스트 {i}]")

    result = generate_response(lora_model, prompt)

    print(result)

    print("-"*60)

#-------------------------------------------------
# 모델 비교: 학습에 없던 새로운 질문들로 일반화 성능 테스트

new_test_prompts = [
    "질문: 배송지 변경이 가능한가요?\n답변:",
    "질문: 상품이 파손되어 왔어요.\n답변:",
    "질문: 카드 할부가 가능한가요?\n답변:",
]

print("="*60)
print("새로운 질문에 대한 두 모델 비교")
print("="*60)

for i, prompt in enumerate(new_test_prompts, 1):
    question = prompt.split("답변:")[0]
    print(f"\n{'='*60}")
    print(f"[질문 {i}] {question}")
    print(f"{'='*60}")

    print("\n[기본 파인튜닝 응답]")
    result1 = generate_response(base_model, prompt, max_tokens=80)
    print(result1)

    print("\n[LoRA 파인튜닝 응답]")
    result2 = generate_response(lora_model, prompt, max_tokens=80)
    print(result2)
    print("-"*60)
