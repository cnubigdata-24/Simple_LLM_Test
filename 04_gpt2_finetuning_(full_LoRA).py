# ============================================================
# 패키지 설치, torchao 비활성화
# ============================================================
!pip install -q transformers datasets torch peft

import peft.import_utils
peft.import_utils.is_torchao_available = lambda: False

import peft
import pkgutil
import importlib

for importer, modname, ispkg in pkgutil.walk_packages(peft.__path__, prefix="peft."):
    try:
        mod = importlib.import_module(modname)
        if hasattr(mod, 'is_torchao_available'):
            mod.is_torchao_available = lambda: False
    except Exception:
        pass

print("torchao 비활성화 패치 완료!")

# ============================================================
# 콜센터 고객 서비스: Full Fine-Tuning vs LoRA 비교 실습
# ============================================================

import time
import torch
import warnings
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType

warnings.filterwarnings('ignore')

# ─── 1. 시스템 정보 ───
print("=" * 60)
print("시스템 정보")
print("=" * 60)
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"VRAM: {vram:.1f}GB")

# ─── 2. 모델 & 토크나이저 ───
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token   # GPT-2는 pad_token이 없으므로 eos로 대체

# ─── 3. 학습 데이터 (콜센터 FAQ 10건) ───
training_data = {
    "text": [
        # 배송 관련 (3건)
        "질문: 주문한 상품이 언제 도착하나요?\n답변: 일반 배송은 주문 후 2-3일 소요되며, 빠른 배송은 익일 배송입니다. 주문번호로 배송 현황을 확인하실 수 있습니다.",
        "질문: 배송 조회는 어떻게 하나요?\n답변: 마이페이지에서 주문 내역을 클릭하시면 운송장 번호로 실시간 배송 추적이 가능합니다. 또는 고객센터로 주문번호를 알려주시면 확인해드립니다.",
        "질문: 해외 배송도 가능한가요?\n답변: 네, 해외 배송 가능합니다. 해외 배송은 국가에 따라 7-15일 소요되며, 배송비는 주문 시 자동 계산됩니다. 통관 지연 시 별도 안내드립니다.",

        # 반품/교환 관련 (3건)
        "질문: 반품하고 싶은데 어떻게 하나요?\n답변: 상품 수령 후 7일 이내 반품 가능합니다. 마이페이지에서 반품 신청하시거나 고객센터로 연락 주시면 수거 접수해드립니다.",
        "질문: 사이즈가 안 맞아서 교환하고 싶어요.\n답변: 상품 수령 후 7일 이내 교환 가능합니다. 마이페이지에서 교환 신청 시 원하시는 사이즈를 선택해주시면 상품 회수 후 새 제품을 발송해드립니다.",
        "질문: 상품이 불량이에요. 어떻게 하나요?\n답변: 불량 상품은 수령일과 관계없이 교환 또는 환불이 가능합니다. 상품 사진을 고객센터로 보내주시면 확인 후 즉시 처리해드리겠습니다.",

        # 결제/환불 관련 (2건)
        "질문: 결제가 두 번 처리된 것 같아요.\n답변: 중복 결제 확인을 위해 주문번호와 결제 내역을 고객센터로 알려주시면 확인 후 즉시 환불 처리해드리겠습니다. 환불은 3-5 영업일 소요됩니다.",
        "질문: 환불은 언제 되나요?\n답변: 반품 상품 확인 후 3-5 영업일 이내 결제하신 수단으로 환불됩니다. 카드 결제는 카드사 일정에 따라 추가 시일이 소요될 수 있습니다.",

        # 회원/계정 관련 (2건)
        "질문: 비밀번호를 잊어버렸어요.\n답변: 로그인 화면에서 '비밀번호 찾기'를 클릭하시면 가입하신 이메일로 재설정 링크가 발송됩니다. 이메일이 오지 않으면 스팸함을 확인해주세요.",
        "질문: 회원 탈퇴는 어떻게 하나요?\n답변: 마이페이지 > 설정 > 회원 탈퇴에서 신청하실 수 있습니다. 탈퇴 시 적립금과 쿠폰은 소멸되며, 처리 완료까지 최대 3일 소요됩니다.",
    ]
}


dataset = Dataset.from_dict(training_data)

def tokenize_fn(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=150)

tokenized_dataset = dataset.map(tokenize_fn, remove_columns=["text"], batched=True)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
print(f"\n학습 데이터: {len(dataset)}개")

# ─── 4. 공통 함수 ───
def generate_response(model, prompt, max_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs, max_new_tokens=max_tokens,
        temperature=0.7, do_sample=True, top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def train_model(model, output_dir, epochs, lr, label):
    """학습 실행 후 소요 시간(초) 반환"""
    args = TrainingArguments(
        output_dir=output_dir, num_train_epochs=epochs,
        per_device_train_batch_size=2, learning_rate=lr,
        logging_steps=10, save_strategy="no",
        fp16=torch.cuda.is_available(), report_to="none"
    )
    trainer = Trainer(
        model=model, args=args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        processing_class=tokenizer
    )
    print(f"\n{'=' * 60}")
    print(f"  {label} 시작 (epochs={epochs}, lr={lr})")
    print(f"{'=' * 60}\n")

    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time

    print(f"\n {label} 완료! (소요 시간: {elapsed:.1f}초)")
    return elapsed


# ─── Part 1: Full Fine-Tuning (모델의 124M개 파라미터 "전부"를 학습) ───

full_model = AutoModelForCausalLM.from_pretrained(model_name)
full_model.config.pad_token_id = tokenizer.eos_token_id

full_total_params = full_model.num_parameters()
full_trainable_params = sum(p.numel() for p in full_model.parameters() if p.requires_grad)

full_time = train_model(full_model, "./full_ft_model", epochs=50, lr=5e-5, label="Full Fine-Tuning")


# ─── Part 2: LoRA Fine-Tuning (원본 가중치는 동결(freeze)하고, 저랭크 행렬(r=8)만 학습) ───

# - 100 에폭 (학습 가능한 파라미터가 적으므로 더 많이 반복)
# - 학습률 3e-4 (적은 파라미터를 더 적극적으로 업데이트)

lora_base_model = AutoModelForCausalLM.from_pretrained(model_name)
lora_base_model.config.pad_token_id = tokenizer.eos_token_id

lora_config = LoraConfig(
    r=8,                          # 저랭크 행렬의 랭크 (작을수록 경량)
    lora_alpha=32,                # 스케일링 팩터
    target_modules=["c_attn"],    # GPT-2의 어텐션 레이어에만 적용
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

lora_model = get_peft_model(lora_base_model, lora_config)

lora_total_params = sum(p.numel() for p in lora_model.parameters())
lora_trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)

lora_model.print_trainable_parameters()

lora_time = train_model(lora_model, "./lora_ft_model", epochs=100, lr=3e-4, label="LoRA Fine-Tuning")


# ─── Part 3: 테스트 - 학습한 질문 (Seen Questions) ───

seen_prompts = [
    "질문: 주문한 상품이 언제 도착하나요?\n답변:",
    "질문: 반품하고 싶은데 어떻게 하나요?\n답변:",
    "질문: 결제가 두 번 처리된 것 같아요.\n답변:",
]

print("\n" + "=" * 60)
print("  > 테스트 1: 학습한 질문 (Seen Questions)")
print("=" * 60)

for i, prompt in enumerate(seen_prompts, 1):
    question = prompt.split("답변:")[0].strip()
    print(f"\n{'─' * 60}")
    print(f"  [질문 {i}] {question}")
    print(f"{'─' * 60}")
    print(f"\n  > Full Fine-Tuning:")
    print(f"  {generate_response(full_model, prompt, 80)}")
    print(f"\n  > LoRA Fine-Tuning:")
    print(f"  {generate_response(lora_model, prompt, 80)}")

# ─── Part 4: Test (Unseen Questions) ───

unseen_prompts = [
    "질문: 배송지 변경이 가능한가요?\n답변:",
    "질문: 상품이 파손되어 왔어요.\n답변:",
    "질문: 카드 할부가 가능한가요?\n답변:",
]

print("\n\n" + "=" * 60)
print("  > 테스트 2: 새로운 질문 (Unseen Questions)")
print("=" * 60)

for i, prompt in enumerate(unseen_prompts, 1):
    question = prompt.split("답변:")[0].strip()
    print(f"\n{'─' * 60}")
    print(f"  [질문 {i}] {question}")
    print(f"{'─' * 60}")
    print(f"\n  > Full Fine-Tuning:")
    print(f"  {generate_response(full_model, prompt, 80)}")
    print(f"\n  > LoRA Fine-Tuning:")
    print(f"  {generate_response(lora_model, prompt, 80)}")

# ─── Part 5: 비교 요약 ───

lora_ratio = (lora_trainable_params / lora_total_params) * 100
time_per_epoch_full = full_time / 50
time_per_epoch_lora = lora_time / 100

print("\n")
print("=" * 60)
print("  Full Fine-Tuning vs LoRA 비교 요약")
print("=" * 60)

print(f"\n  [파라미터 비교]")
print(f"    Full FT 전체 파라미터:  {full_total_params:>15,}개")
print(f"    Full FT 학습 파라미터:  {full_trainable_params:>15,}개 (100%)")
print(f"    LoRA 전체 파라미터:     {lora_total_params:>15,}개")
print(f"    LoRA 학습 파라미터:     {lora_trainable_params:>15,}개 ({lora_ratio:.2f}%)")

print(f"\n  [학습 설정 비교]")
print(f"    Full FT:  50 에폭,  학습률 5e-5")
print(f"    LoRA:    100 에폭,  학습률 3e-4")

print(f"\n  [속도 비교]")
print(f"    Full FT 총 학습 시간:   {full_time:.1f}초  (에폭당 {time_per_epoch_full:.2f}초)")
print(f"    LoRA 총 학습 시간:      {lora_time:.1f}초  (에폭당 {time_per_epoch_lora:.2f}초)")
print(f"    에폭당 속도:  LoRA가 약 {time_per_epoch_full / time_per_epoch_lora:.1f}배 빠름")

print(f"\n  [핵심 차이점]")
print(f"    - LoRA는 전체의 {lora_ratio:.2f}%만 학습해도 유사한 성능 달성")
print(f"    - 메모리 사용량이 크게 절감되어 큰 모델에 유리")
print(f"    - 여러 태스크용 어댑터를 따로 저장/교체 가능")
print(f"    - 원본 모델이 보존되어 재학습 없이 복원 가능")

print("\n" + "=" * 60)
