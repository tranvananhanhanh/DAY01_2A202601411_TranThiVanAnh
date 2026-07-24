# run_experiments.py
import os
import time
from template import call_openai, chat_with_system_prompt, count_tokens

from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

models = client.models.list()

for model in models.data:
    print(model.id)

def run_block_1_experiment():
    print("=" * 60)
    print("THỬ NGHIỆM BLOCK 1: NHIỆT ĐỘ (TEMPERATURE)")
    print("=" * 60)
    
    prompt = "Hãy kể cho tôi một sự thật thú vị về Việt Nam."
    temperatures = [0.0, 0.5, 1.0, 1.5]
    
    for temp in temperatures:
        print(f"\n[Đang gọi với Temperature = {temp}...]")
        response, latency = call_openai(prompt, temperature=temp)
        print(f"--- Kết quả (Temperature {temp}) | Độ trễ: {latency:.2f} giây ---")
        print(response.strip() if response else "[Không có phản hồi do lỗi API]")
        print("-" * 60)
        # Chờ 5 giây để tránh bị giới hạn Requests Per Minute (RPM)
        time.sleep(5)

def run_block_2_experiment_personas():
    print("\n" + "=" * 60)
    print("THỬ NGHIỆM BLOCK 2.1: SỨC MẠNH CỦA PERSONA")
    print("=" * 60)
    
    user_prompt = "Giải thích blockchain là gì?"
    
    # Persona 1: Giáo viên tiểu học
    sys_prompt_1 = "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
    print("\n[Đang gọi Persona 1: Giáo viên tiểu học...]")
    resp_1, lat_1 = chat_with_system_prompt(sys_prompt_1, user_prompt)
    print(f"--- Kết quả Persona 1 | Độ trễ: {lat_1:.2f} giây ---")
    print(resp_1.strip() if resp_1 else "[Không có phản hồi]")
    
    time.sleep(5) # Chờ tránh trùng rate limit
    
    # Persona 2: Chuyên gia tài chính
    sys_prompt_2 = "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."
    print("\n[Đang gọi Persona 2: Chuyên gia tài chính...]")
    resp_2, lat_2 = chat_with_system_prompt(sys_prompt_2, user_prompt)
    print(f"--- Kết quả Persona 2 | Độ trễ: {lat_2:.2f} giây ---")
    print(resp_2.strip() if resp_2 else "[Không có phản hồi]")
    print("-" * 60)

def run_block_2_experiment_tokens():
    print("\n" + "=" * 60)
    print("THỬ NGHIỆM BLOCK 2.2: TIKTOKEN VS ĐẾM TỪ")
    print("=" * 60)
    
    # Đoạn văn tiếng Việt mẫu (khoảng 100 từ)
    vietnamese_text = (
        "Tiếng Việt là ngôn ngữ chính thức tại Việt Nam. Đây là tiếng mẹ đẻ của khoảng tám mươi lăm "
        "phần trăm dân số Việt Nam, cùng với hơn bốn triệu người Việt kiều sinh sống ở nước ngoài. "
        "Tiếng Việt còn là ngôn ngữ thứ hai của nhiều dân tộc thiểu số tại Việt Nam. Về mặt phân loại học, "
        "tiếng Việt được đặt vào ngữ hệ Nam Á, có quan hệ gần gũi nhất với tiếng Mường. Tiếng Việt là một "
        "ngôn ngữ đơn âm tiết và có thanh điệu, với hệ thống sáu thanh điệu tương đối phức tạp, đóng vai "
        "trò quan trọng trong việc biểu đạt ngữ nghĩa của từ ngữ khi giao tiếp hàng ngày."
    )
    
    word_count = len(vietnamese_text.split())
    token_count = count_tokens(vietnamese_text)
    estimated_tokens = word_count / 0.75
    
    # Tính phần trăm chênh lệch
    diff_percent = abs(token_count - estimated_tokens) / token_count * 100
    
    print(f"Đoạn văn kiểm thử: \"{vietnamese_text[:60]}...\"")
    print(f"- Số từ đếm được: {word_count} từ")
    print(f"- Số token ước lượng (Số từ / 0.75): {estimated_tokens:.2f}")
    print(f"- Số token thực tế (count_tokens): {token_count}")
    print(f"- Chênh lệch: {diff_percent:.2f}%")
    print("-" * 60)

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("Cảnh báo: Không tìm thấy biến môi trường OPENAI_API_KEY.")
    else:
        run_block_1_experiment()
        run_block_2_experiment_personas()
        run_block_2_experiment_tokens()
