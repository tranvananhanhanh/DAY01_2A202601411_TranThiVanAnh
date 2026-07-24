"""
K3 — Ngày 1: Khám Phá LLM API (9h00–13h00)
AICB-P1: AI Practical Competency Program, Phase 1

Hướng dẫn:
    1. Làm theo LAB_GUIDE.md — mỗi block có các bước chi tiết và checkpoint.
    2. Điền vào tất cả các chỗ đánh dấu TODO.
    3. KHÔNG đổi chữ ký hàm (tên hàm, tham số).
    4. Import OpenAI BÊN TRONG hàm (xem gợi ý) — nếu import ở đầu file,
       các bài test mock sẽ không hoạt động.
    5. Kiểm tra tiến độ:  pytest tests/test_part1.py -v  (từng phần)
       Chấm điểm tổng:    python grade.py
"""

import os
import time
from typing import Any, Callable

from dotenv import load_dotenv

# Nạp OPENAI_API_KEY từ file .env (copy .env.example thành .env và dán key vào)
load_dotenv()

# ---------------------------------------------------------------------------
# Bảng giá ước tính (USD / 1K token) — cập nhật cho cả GPT và Gemini (Chuyển sang dùng GEMINI FLASH & LITE vì em k có key OpenAI)
# ---------------------------------------------------------------------------
PRICING_PER_1K_TOKENS = {
    "gpt-4o": {"input": 0.0025, "output": 0.010},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gemini-flash-latest": {"input": 0.000075, "output": 0.0003},
    "gemini-flash-lite-latest": {"input": 0.0000375, "output": 0.00015},
}

# Tên model có thể đổi qua .env — ví dụ khi dùng Gemini OpenAI-compatible endpoint.
OPENAI_MODEL = os.getenv("LAB_MODEL", "gemini-flash-latest")
OPENAI_MINI_MODEL = os.getenv("LAB_MINI_MODEL", "gemini-flash-lite-latest")

# Endpoint tương thích chuẩn OpenAI của Gemini
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


# ===========================================================================
# PART 1 — API CƠ BẢN (Block 1: 10h00–10h40)
# ===========================================================================

# ---------------------------------------------------------------------------
# Task 1.1 — Gọi GPT-4o
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Gọi OpenAI Chat Completions API, trả về nội dung phản hồi + độ trễ.

    Args:
        prompt:      Tin nhắn của người dùng.
        model:       Model sử dụng.
        temperature: Độ ngẫu nhiên khi lấy mẫu (0.0 – 2.0).
        top_p:       Ngưỡng nucleus sampling.
        max_tokens:  Số token tối đa được sinh ra.

    Returns:
        Tuple (response_text: str, latency_seconds: float).
    """
    from openai import OpenAI
    import os
    import time

    base_url = os.getenv("OPENAI_BASE_URL", GEMINI_BASE_URL)
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=base_url
    )

    try:
        start_time = time.perf_counter()

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        latency_seconds = time.perf_counter() - start_time
        response_text = response.choices[0].message.content

        return response_text, latency_seconds

    except Exception as e:
        print(f"API Error: {e}")
        return "", 0.0


# ---------------------------------------------------------------------------
# Task 1.2 — Gọi GPT-4o-mini
# ---------------------------------------------------------------------------
def call_openai_mini(
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Gọi API với model gpt-4o-mini hoặc gemini-flash-lite-latest — nhanh hơn và rẻ hơn.

    Returns:
        Tuple (response_text: str, latency_seconds: float).
    """
    return call_openai(
        prompt=prompt,
        model=OPENAI_MINI_MODEL,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# Task 1.3 — So sánh GPT-4o vs GPT-4o-mini
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    """
    Gọi cả hai model với cùng một prompt và trả về dict so sánh.

    Returns:
        Dict với các key:
            - "gpt4o_response":      str
            - "mini_response":       str
            - "gpt4o_latency":       float
            - "mini_latency":        float
            - "gpt4o_cost_estimate": float  (USD ước tính cho phản hồi)
    """
    gpt4o_response, gpt4o_latency = call_openai(prompt)
    mini_response, mini_latency = call_openai_mini(prompt)

    # Ước tính chi phí output
    estimated_tokens = len(gpt4o_response.split()) / 0.75
    
    pricing_model = OPENAI_MODEL if OPENAI_MODEL in PRICING_PER_1K_TOKENS else "gpt-4o"
    gpt4o_cost_estimate = (
        estimated_tokens / 1000
        * PRICING_PER_1K_TOKENS[pricing_model]["output"]
    )

    return {
        "gpt4o_response": gpt4o_response,
        "mini_response": mini_response,
        "gpt4o_latency": gpt4o_latency,
        "mini_latency": mini_latency,
        "gpt4o_cost_estimate": gpt4o_cost_estimate,
    }


# ===========================================================================
# PART 2 — SYSTEM PROMPT & TOKEN (Block 2: 10h40–11h20)
# ===========================================================================

# ---------------------------------------------------------------------------
# Task 2.1 — Chat với system prompt (persona)
# ---------------------------------------------------------------------------
def chat_with_system_prompt(
    system_prompt: str,
    user_prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Gọi API với MESSAGES gồm 2 phần: system prompt (định hình vai trò/persona
    của model) và user prompt (câu hỏi thật).

    Args:
        system_prompt: Chỉ dẫn vai trò, ví dụ "Bạn là giáo viên tiểu học,
                       giải thích mọi thứ thật đơn giản."
        user_prompt:   Tin nhắn của người dùng.

    Returns:
        Tuple (response_text: str, latency_seconds: float).
    """
    from openai import OpenAI
    import os
    import time

    base_url = os.getenv("OPENAI_BASE_URL", GEMINI_BASE_URL)
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=base_url
    )

    try:
        start_time = time.perf_counter()

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        latency_seconds = time.perf_counter() - start_time
        response_text = response.choices[0].message.content

        return response_text, latency_seconds

    except Exception as e:
        print(f"API Error: {e}")
        return "", 0.0


# ---------------------------------------------------------------------------
# Task 2.2 — Đếm token bằng tiktoken
# ---------------------------------------------------------------------------
def count_tokens(text: str, model: str = OPENAI_MODEL) -> int:
    """
    Đếm số token của một đoạn text bằng thư viện tiktoken.

    Args:
        text:  Đoạn text cần đếm.
        model: Model dùng để chọn bộ mã hóa (encoding).

    Returns:
        Số token (int).
    """
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        # Fallback khi offline hoặc sử dụng model không được tiktoken trực tiếp hỗ trợ (ví dụ Gemini)
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Task 2.3 — Ước tính chi phí chính xác
# ---------------------------------------------------------------------------
def estimate_cost(prompt: str, response: str, model: str = OPENAI_MODEL) -> dict:
    """
    Tính chi phí một lượt gọi API dựa trên số token THẬT (đếm bằng
    count_tokens) và bảng giá PRICING_PER_1K_TOKENS.

    Returns:
        Dict với các key:
            - "input_tokens":  int
            - "output_tokens": int
            - "input_cost":    float  (USD)
            - "output_cost":   float  (USD)
            - "total_cost":    float  (USD)
    """
    input_tokens = count_tokens(prompt, model)
    output_tokens = count_tokens(response, model)

    pricing = PRICING_PER_1K_TOKENS.get(model, PRICING_PER_1K_TOKENS["gpt-4o"])
    
    input_cost = input_tokens / 1000 * pricing["input"]
    output_cost = output_tokens / 1000 * pricing["output"]
    total_cost = input_cost + output_cost

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }


# ===========================================================================
# PART 3 — STREAMING & ĐỘ BỀN (Block 3: 11h30–12h10)
# ===========================================================================

# ---------------------------------------------------------------------------
# Task 3.1 — Chatbot streaming có lịch sử hội thoại
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Chatbot dòng lệnh tương tác dùng streaming.
    """
    from openai import OpenAI
    import os

    base_url = os.getenv("OPENAI_BASE_URL", GEMINI_BASE_URL)
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=base_url
    )

    history = []

    while True:
        try:
            user_msg = input("You: ")
        except (KeyboardInterrupt, EOFError):
            print("\nChatbot terminated.")
            break

        if user_msg.strip().lower() in ("quit", "exit"):
            print("Chatbot terminated.")
            break

        messages = history + [{"role": "user", "content": user_msg}]

        try:
            stream = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                stream=True,
            )

            print("Assistant: ", end="", flush=True)
            reply = ""
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                print(delta, end="", flush=True)
                reply += delta
            print()

            history.append({"role": "user", "content": user_msg})
            history.append({"role": "assistant", "content": reply})
            
            # Giữ tối đa 3 lượt hội thoại (6 messages)
            history = history[-6:]
            
        except Exception as e:
            print(f"\nAPI Error: {e}")


# ---------------------------------------------------------------------------
# Task 3.2 — Retry với exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable,
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Gọi fn(). Nếu ném exception, thử lại tối đa max_retries lần với
    exponential backoff (delay = base_delay * 2^attempt).
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_exception = e
            if attempt == max_retries:
                raise last_exception
            
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)


# ===========================================================================
# PART 4 — MINI-PROJECT: TRỢ LÝ CLI HOÀN CHỈNH (Block 4: 12h10–12h50)
# ===========================================================================
def run_assistant(
    persona: str,
    get_input: Callable[[], str] = None,
    max_turns: int = None,
) -> dict:
    """
    Trợ lý CLI hoàn chỉnh — ghép mọi thứ bạn đã xây trong Part 1–3.
    """
    from openai import OpenAI
    import os

    if get_input is None:
        get_input = input

    base_url = os.getenv("OPENAI_BASE_URL", GEMINI_BASE_URL)
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=base_url
    )

    history = []
    num_turns = 0
    total_tokens = 0
    total_cost = 0.0

    while True:
        if max_turns is not None and num_turns >= max_turns:
            break

        try:
            user_msg = get_input()
        except (KeyboardInterrupt, EOFError):
            break

        if user_msg.strip().lower() in ("quit", "exit"):
            break

        messages = (
            [{"role": "system", "content": persona}]
            + history
            + [{"role": "user", "content": user_msg}]
        )

        try:
            stream = retry_with_backoff(
                lambda: client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=messages,
                    stream=True,
                )
            )

            print("Assistant: ", end="", flush=True)
            reply = ""
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                print(delta, end="", flush=True)
                reply += delta
            print()

            history.append({"role": "user", "content": user_msg})
            history.append({"role": "assistant", "content": reply})
            history = history[-6:]

            num_turns += 1
            total_tokens += count_tokens(user_msg, OPENAI_MODEL) + count_tokens(reply, OPENAI_MODEL)
            total_cost += estimate_cost(user_msg, reply, OPENAI_MODEL)["total_cost"]

        except Exception as e:
            print(f"\nError: {e}")
            break

    return {
        "num_turns": num_turns,
        "total_tokens": total_tokens,
        "total_cost": total_cost,
        "history": history,
    }


# ===========================================================================
# BONUS (không bắt buộc — cho bạn nào xong sớm)
# ===========================================================================
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Chạy compare_models cho từng prompt trong list.
    """
    results = []
    for prompt in prompts:
        result = compare_models(prompt)
        result["prompt"] = prompt
        results.append(result)
    return results


def format_comparison_table(results: list[dict]) -> str:
    """
    Định dạng kết quả batch_compare thành bảng text dễ đọc.
    """
    headers = [
        "Prompt",
        "Gemini Flash Response",
        "Lite Response",
        "Flash Latency",
        "Lite Latency",
    ]

    table = [
        f"{headers[0]:<42}| {headers[1]:<42}| {headers[2]:<42}| {headers[3]:<15}| {headers[4]:<15}"
    ]
    table.append("-" * len(table[0]))

    for result in results:
        prompt = result["prompt"][:40]
        gpt4o_response = result["gpt4o_response"][:40]
        mini_response = result["mini_response"][:40]
        gpt4o_latency = f"{result['gpt4o_latency']:.4f}"
        mini_latency = f"{result['mini_latency']:.4f}"

        table.append(
            f"{prompt:<42}| "
            f"{gpt4o_response:<42}| "
            f"{mini_response:<42}| "
            f"{gpt4o_latency:<15}| "
            f"{mini_latency:<15}"
        )

    return "\n".join(table)


# ---------------------------------------------------------------------------
# Entry point — demo chạy thật (cần OPENAI_API_KEY)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== So sánh model ===")
    result = compare_models(
        "Giải thích khác biệt giữa temperature và top_p trong một câu."
    )
    for key, value in result.items():
        print(f"{key}: {value}")

    print("\n=== Trợ lý CLI (gõ 'quit' để thoát) ===")
    stats = run_assistant(
        persona="Bạn là trợ giảng thân thiện của khóa AI, "
                "trả lời ngắn gọn bằng tiếng Việt.",
    )
    print("\n--- Thống kê phiên chat ---")
    for key, value in stats.items():
        if key != "history":
            print(f"{key}: {value}")
