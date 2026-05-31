## 실습 개요: GPU 사용 필수 (CPU 단독 실행 불가)
# Llama 3.2 1B 모델을 Unsloth와 QLoRA를 사용하여 고객 문의를 5개 카테고리로 자동 분류

## 분류 카테고리 (5개)
# 1.배송 (Shipping): 배송 관련 문의
# 2.반품/교환 (Return): 반품, 교환 관련 문의
# 3.결제 (Payment): 결제, 환불 관련 문의
# 4.제품 (Product): 제품 정보, 재고 문의
# 5.계정 (Account): 로그인, 회원 정보 관련 문의

## 학습 데이터
# 총 13개 (카테고리별 2-3개)
# 실제 콜센터 문의 기반

## 기술 스택 및 하드웨어 제약
# 모델: Llama 3.2 1B Instruct (가볍고 빠름)
# 최적화: Unsloth (NVIDIA GPU 전용 CUDA 커널 가속 기술 - CPU 미지원)
# 파인튜닝: QLoRA (4-bit quantization - bitsandbytes 라이브러리가 GPU 연산 필수 요구)
# 메모리: 약 2-3GB VRAM 필요 (Google Colab 무료 버전인 T4 GPU에서 충분히 작동)

#----------------------------------------------------

# [GPU 필수] Unsloth 및 종속 패키지(xformers, bitsandbytes 등)는 GPU가 인식되어야만 정상 설치/로드
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "xformers==0.0.27" trl peft accelerate bitsandbytes

#----------------------------------------------------

import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import TrainingArguments, TextStreamer
from trl import SFTTrainer
from datasets import Dataset
import pandas as pd

# 시스템 정보 및 GPU 활성화 여부 확인
print("="*60)
print("시스템 하드웨어 정보 점검")
print("="*60)
print(f"PyTorch 버전: {torch.__version__}")
print(f"CUDA(GPU) 사용 가능 여부: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"사용 중인 GPU: {torch.cuda.get_device_name(0)}")
    print(f"사용 가능한 VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB")
    print("\n[성공] GPU 모드가 활성화, 실습 진행...")
else:
    print("\n[경고/에러] 현재 CPU 모드, 모델 로드 및 학습 불가능 !!!")
    print("▶ 코랩 상단 메뉴에서 [런타임] > [런타임 유형 변경] > 하드웨어 가속기를 [T4 GPU]로 선택 후 다시 실행.")
    
    assert torch.cuda.is_available(), "GPU가 없습니다. 런타임 유형을 GPU로 변경해 주세요."

#----------------------------------------------------

# 하이퍼파라미터 설정
max_seq_length = 512  # 문의 텍스트는 짧으므로 512면 충분
load_in_4bit = True   # QLoRA: 4-bit quantization 사용 (bitsandbytes 가 작동하기 위해 GPU 필수)
dtype = None          # 자동으로 최적 dtype 선택 (bfloat16 or float16)

print("\n원격 저장소로부터 모델 로딩 중...")

# FastLanguageModel: Unsloth 최적화 커널을 사용해 4-bit 양자화 모델을 GPU VRAM에 직접 올립니다.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_your_token_here",  # Hugging Face 토큰 (필요시)
)

print(f"\n모델 로드 완료!")
print(f"모델: Llama 3.2 1B Instruct")
print(f"Quantization: 4-bit (QLoRA 연산 준비 완료)")

#----------------------------------------------------

# LoRA 적용: 원본 모델에 LoRA 어댑터 추가
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,  # LoRA rank: 16이면 적당한 성능과 속도 보장
    lora_alpha = 16,  # Scaling factor: r과 같은 값 사용
    lora_dropout = 0,  # 비활성화 (작은 데이터셋: 0)
    bias = "none",  # Bias 파라미터는 학습하지 않음

    # target_modules: Llama의 주요 attention 레이어에 LoRA 적용
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],

    use_gradient_checkpointing = "unsloth",  # Unsloth의 GPU 메모리 최적화 기법 적용

    random_state = 42,  # 재현성을 위한 시드 고정
)

print("\nLoRA 설정 완료!")
print(f"LoRA rank (r): 16")
print(f"학습 파라미터: 전체의 약 1% 미만 (빠른 자원 업데이트 가능)")

#----------------------------------------------------

# 고객 문의 분류 데이터셋: 카테고리 (배송/반품/결제/제품/계정)
training_data = [
    # 배송 관련 (3개)
    {
        "instruction": "다음 고객 문의를 카테고리로 분류하세요. 카테고리는 배송, 반품, 결제, 제품, 계정 중 하나입니다.",
        "input": "주문한 상품이 언제 도착하나요?",
        "output": "배송"
    },
    {
        "instruction": "다음 고객 문의를 카테고리로 분류하세요. 카테고리는 배송, 반품, 결제, 제품, 계정 중 하나입니다.",
        "input": "배송 조회는 어떻게 하나요?",
        "output": "배송"
    },
    {
        "instruction": "다음 고객 문의를 카테고리로 분류하세요. 카테고리는 배송, 반품, 결제, 제품, 계정 중 하나입니다.",
        "input": "배송지 변경이 가능한가요?",
        "output": "배송"
    },

    # 반품/교환 관련 (3개)
    {
        "instruction": "다음 고객 문의를 카테고리로 분류하세요. 카테고리는 배송, 반품, 결제, 제품, 계정 중 하나입니다.",
        "input": "반품하고 싶은데 어떻게 하나요?",
        "output": "반품"
    },
    {
        "instruction": "다음 고객 문의를 카테고리로 분류하세요. 카테고리는 배송, 반품, 결제, 제품, 계정 중 하나입니다.",
        "input": "사이즈가 안 맞아서 교환하고 싶어요.",
        "output": "반품"
    },
    {
        "instruction": "다음 고객 문의를 카테고리로 분류하세요. 카테고리는 배송, 반품, 결제, 제품, 계정 중 하나입니다.",
        "input": "상품이 파손되어 왔어요. 교환 가능한가요?",
        "output": "반품"
    },

    # 결제 관련 (3개)
    {
        "instruction": "다음 고객 문의를 카테고리로 분류하세요. 카테고리는 배송, 반품, 결제, 제품, 계정 중 하나입니다.",
        "input": "결제가 두 번 처리된 것 같아요.",
        "output": "결제"
    },
    {
        "instruction": "다음 고객 문의를 카테고리로 분류하세요. 카테고리는 배송, 반품, 결제, 제품, 계정 중 하나입니다.",
        "input": "환불은 언제 되나요?",
        "output": "결제"
    },
    {
        "instruction": "다음 고객 문의를 카테고리로 분류하세요. 카테고리는 배송, 반품, 결제, 제품, 계정 중 하나입니다.",
        "input": "카드 할부가 가능한가요?",
        "output": "결제"
    },

    # 제품 관련 (2개)
    {
        "instruction": "다음 고객 문의를 카테고리로 분류하세요. 카테고리는 배송, 반품, 결제, 제품, 계정 중 하나입니다.",
        "input": "이 제품 재입고는 언제 되나요?",
        "output": "제품"
    },
    {
        "instruction": "다음 고객 문의를 카테고리로 분류하세요. 카테고리는 배송, 반품, 결제, 제품, 계정 중 하나입니다.",
        "input": "이 상품 사이즈표를 알고 싶어요.",
        "output": "제품"
    },

    # 계정 관련 (2개)
    {
        "instruction": "다음 고객 문의를 카테고리로 분류하세요. 카테고리는 배송, 반품, 결제, 제품, 계정 중 하나입니다.",
        "input": "비밀번호를 잊어버렸어요.",
        "output": "계정"
    },
    {
        "instruction": "다음 고객 문의를 카테고리로 분류하세요. 카테고리는 배송, 반품, 결제, 제품, 계정 중 하나입니다.",
        "input": "회원 탈퇴하고 싶어요.",
        "output": "계정"
    },
]

#----------------------------------------------------

# 데이터프레임으로 변환
df = pd.DataFrame(training_data)

print("\n학습 데이터셋:")
print(f"총 {len(df)}개 샘플")
print("\n카테고리별 분포:")
print(df['output'].value_counts())
print("\n샘플 데이터:")
print(df.head(3))

#----------------------------------------------------

# Alpaca 프롬프트 템플릿
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token

# 데이터를 Alpaca 포맷으로 변환
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]

    texts = []
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input_text, output) + EOS_TOKEN
        texts.append(text)

    return {"text": texts}

# Hugging Face Dataset으로 변환
dataset = Dataset.from_pandas(df)

# Alpaca 포맷 적용
dataset = dataset.map(
    formatting_prompts_func,
    batched=True,
)

print("\nAlpaca 포맷 적용 완료!")
print("\n[포맷팅된 샘플 예시]")
print(dataset[0]['text'][:300] + "...")

#----------------------------------------------------

# 학습 설정
training_args = TrainingArguments(
    output_dir = "./outputs_classification", 

    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    warmup_ratio = 0.1,
    num_train_epochs = 3,
    learning_rate = 2e-4,

    # GPU 가용 하드웨어 가속 정밀도 설정
    fp16 = not is_bfloat16_supported(),
    bf16 = is_bfloat16_supported(),

    logging_steps = 1,
    optim = "adamw_8bit", # GPU 효율 증대를 위한 8비트 AdamW 옵티마이저
    weight_decay = 0.01,
    lr_scheduler_type = "cosine",
    seed = 42,
    report_to = "none", 
)

print("학습 설정 완료!")
print(f"\n배치 크기: {training_args.per_device_train_batch_size}")
print(f"그래디언트 누적: {training_args.gradient_accumulation_steps}")
print(f"실제 배치 크기: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"Epochs: {training_args.num_train_epochs}")
print(f"학습률: {training_args.learning_rate}")

#----------------------------------------------------

# SFTTrainer 초기화
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",  
    max_seq_length = max_seq_length,
    args = training_args,
    packing = False,  
)

print("="*60)
print("GPU 연산 가속 학습 시작")
print("="*60)

print(f"학습 데이터: {len(dataset)}개")
print(f"Epochs: {training_args.num_train_epochs}")
print(f"예상 소요 시간: 약 1~3분 내외 (코랩 T4 GPU 적용 기준)")
print("\n학습 중...\n")

# 학습 실행 (GPU 내부에서 병렬 연산 처리)
trainer.train()

print("\n" + "="*60)
print("학습 완료!")
print("="*60)

#----------------------------------------------------

# 추론 모드로 전환 (학습 상태 구조를 해제하고 연산 최적화)
FastLanguageModel.for_inference(model)

# 고객 문의 분류 함수
def classify_inquiry(inquiry_text):
    instruction = "다음 고객 문의를 카테고리로 분류하세요. 카테고리는 배송, 반품, 결제, 제품, 계정 중 하나입니다."
    prompt = alpaca_prompt.format(instruction, inquiry_text, "")

    # 모델이 GPU에 할당되어 있으므로 입력 토큰 역시 반드시 .to("cuda")로 GPU에 전달되어야 합니다.
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    # 텍스트 생성 연산
    outputs = model.generate(
        **inputs,
        max_new_tokens=10,  
        temperature=0.1,    
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "### Response:" in result:
        prediction = result.split("### Response:")[1].strip()
    else:
        prediction = result

    return prediction

print("분류 추론 함수 준비 완료!")

#----------------------------------------------------

# 학습 데이터 테스트 1
print("="*60)
print("테스트 1: 학습 데이터 검증")
print("="*60)

test_cases_trained = [
    "주문한 상품이 언제 도착하나요?",
    "반품하고 싶은데 어떻게 하나요?",
    "결제가 두 번 처리된 것 같아요.",
    "이 제품 재입고는 언제 되나요?",
    "비밀번호를 잊어버렸어요.",
]

expected_categories = ["배송", "반품", "결제", "제품", "계정"]

for i, (inquiry, expected) in enumerate(zip(test_cases_trained, expected_categories), 1):
    prediction = classify_inquiry(inquiry)

    print(f"\n[테스트 {i}]")
    print(f"문의: {inquiry}")
    print(f"정답: {expected}")
    print(f"예측: {prediction}")
    print(f"결과: {'O 정확' if expected in prediction else 'X 오답'}")
    print("-"*60)

#----------------------------------------------------

# 미학습 데이터 테스트 (일반화 기능 검증)
print("\n" + "="*60)
print("테스트 2: 새로운 문의 검증 (학습에 쓰이지 않은 실전 텍스트)")
print("="*60)

test_cases_new = [
    {"inquiry": "택배가 아직 안 왔어요.", "expected": "배송"},
    {"inquiry": "색상을 바꾸고 싶은데 교환 되나요?", "expected": "반품"},
    {"inquiry": "신용카드로 결제했는데 취소하려면?", "expected": "결제"},
    {"inquiry": "이 제품 블랙 색상 있나요?", "expected": "제품"},
    {"inquiry": "로그인이 안 돼요.", "expected": "계정"},
    {"inquiry": "배송 날짜를 변경하고 싶어요.", "expected": "배송"},
    {"inquiry": "포인트 환불 받을 수 있나요?", "expected": "결제"},
]

correct = 0
total = len(test_cases_new)

for i, test_case in enumerate(test_cases_new, 1):
    inquiry = test_case["inquiry"]
    expected = test_case["expected"]

    prediction = classify_inquiry(inquiry)
    is_correct = expected in prediction

    if is_correct:
        correct += 1

    print(f"\n[테스트 {i}]")
    print(f"문의: {inquiry}")
    print(f"정답: {expected}")
    print(f"예측: {prediction}")
    print(f"결과: {'O 정확' if is_correct else 'X 오답'}")
    print("-"*60)

# 정확도 산출
accuracy = (correct / total) * 100
print(f"\n" + "="*60)
print(f"새로운 문의 테스트 결과 정확도: {correct}/{total} = {accuracy:.1f}%")
print("="*60)

#----------------------------------------------------

# 모델 가중치(LoRA 가중치만 격리 추출) 로컬 저장
model.save_pretrained("classification_lora_model")
tokenizer.save_pretrained("classification_lora_model")

print("\n파인튜닝된 어댑터 모델 저장 완료!")
print("저장 위치: ./classification_lora_model")
print("\n생성 파일 정보:")
print("- LoRA 레이어 가중치 (adapter_model.safetensors, 약 10~50MB 내외)")
print("- Tokenizer 환경 메타데이터")
