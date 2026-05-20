"""
LangGraph 에이전트
- LLM_PROVIDER 환경변수로 모델 전환 (gemini / claude)
- Claude/Gemini가 의도 분류 + 도구 호출 둘 다 수행
- 온디바이스 LLM과 같은 JSON 포맷으로 응답
"""
import os
import json
import re
from pathlib import Path
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

load_dotenv()

SYSTEM_PROMPT = """당신은 디지털 취약계층(어르신 등)을 위한 도우미입니다.

【중요】 반드시 JSON 형식으로만 응답하세요. 다른 텍스트는 절대 포함하지 마세요.
코드블록(```)도 쓰지 마세요. 순수 JSON만 출력하세요.

응답 형식:
{"intent": "...", "query": "...", "msg": "한국어 답변"}

intent 종류:
- "chat": 일반 대화, 날씨 안내, 검색 결과 설명 등 (도구 사용 후 답변할 때도 chat)
- "buy_product": 쇼핑 요청 (예: "라면 사줘", "휴지 주문해줘")
- "book_ktx": 기차 예매 요청 (예: "기차 예매해줘", "KTX 타고 싶어")

규칙:
1. 날씨를 물으면 반드시 get_weather 도구를 먼저 호출하세요. 절대 도구 없이 날씨를 답변하지 마세요.
   도구 결과를 받은 후 intent="chat"으로 자연스럽게 한국어로 답변하세요.
2. 쇼핑/기차 요청이면 도구를 호출하지 말고 바로 해당 intent를 반환하세요.
   - query에는 검색할 상품명을 넣으세요 (기차는 빈 문자열).
3. msg는 항상 한국어로, 짧고 친절하게 작성하세요.

예시 응답:

사용자: "안녕"
{"intent": "chat", "query": "", "msg": "안녕하세요! 무엇을 도와드릴까요?"}

사용자: "라면 사줘"
{"intent": "buy_product", "query": "라면", "msg": "네, 라면을 찾아드릴게요."}

사용자: "기차 예매해줘"
{"intent": "book_ktx", "query": "", "msg": "네, 기차표 예매 도와드릴게요."}

사용자: "오늘 서울 날씨 어때?"
(반드시 먼저 get_weather("서울") 도구를 호출하고, 결과를 받은 후 답변)
{"intent": "chat", "query": "", "msg": "지금 서울은 맑고 18도예요. 가벼운 외투면 충분하겠어요."}
"""


def build_llm():
    """환경변수에 따라 LLM 선택"""
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()

    if provider == "claude":
        print("🤖 Using Claude (Opus 4.7)")
        return ChatAnthropic(
            model="claude-opus-4-7",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0.2,
        )
    elif provider == "gemini":
        print("🤖 Using Gemini 2.5 Flash")
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.2,
        )
    else:
        raise ValueError(f"알 수 없는 LLM_PROVIDER: {provider}")


_agent = None


async def get_agent():
    global _agent
    if _agent is not None:
        return _agent

    weather_script = str(Path(__file__).parent / "weather_mcp.py")
    print(f"🔌 MCP weather 서버 경로: {weather_script}")

    client = MultiServerMCPClient({
        "weather": {
            "command": "python",
            "args": [weather_script],
            "transport": "stdio",
        }
    })
    tools = await client.get_tools()

    # 🌟 등록된 도구 확인
    print(f"🛠️ 등록된 도구 수: {len(tools)}")
    for tool in tools:
        print(f"   - {tool.name}: {tool.description[:80]}")

    llm = build_llm()

    _agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=SYSTEM_PROMPT,
    )
    return _agent


def parse_json_response(text: str) -> dict:
    """LLM 응답에서 JSON 추출"""
    try:
        # 코드블록 제거 (Gemini가 가끔 ```json 으로 감쌈)
        text = re.sub(r'```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```', '', text)

        # 첫 { 부터 매칭되는 } 까지
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return json.loads(text)
    except Exception as e:
        print(f"⚠️ JSON 파싱 실패: {e}")
        print(f"원본 텍스트: {text}")
        return {
            "intent": "chat",
            "query": "",
            "msg": text.strip() or "응답을 이해하지 못했어요.",
        }


async def run_agent(user_message: str) -> dict:
    agent = await get_agent()
    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": user_message}]
    })

    messages = result["messages"]

    # 🌟 디버그: 메시지 흐름 출력
    print("=" * 60)
    print(f"📨 사용자 입력: {user_message}")
    print("-" * 60)
    for i, msg in enumerate(messages):
        msg_type = msg.__class__.__name__
        content = msg.content if hasattr(msg, 'content') else ''
        if isinstance(content, list):
            content_str = json.dumps(content, ensure_ascii=False)[:300]
        else:
            content_str = str(content)[:300]
        print(f"[{i}] {msg_type}: {content_str}")

        # 도구 호출 정보 출력
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"    🔧 도구 호출: {tc.get('name')}({tc.get('args')})")

        # ToolMessage의 name
        if msg_type == 'ToolMessage':
            tool_name = getattr(msg, 'name', '?')
            print(f"    ↩️ 도구 결과 ({tool_name})")
    print("=" * 60)

    final_text = messages[-1].content if messages else ""
    if isinstance(final_text, list):
        final_text = "".join(
            b.get("text", "") if isinstance(b, dict) else str(b)
            for b in final_text
        )

    print(f"📩 LLM 원본 응답: {final_text}")
    parsed = parse_json_response(final_text)

    return {
        "intent": parsed.get("intent", "chat"),
        "query": parsed.get("query", ""),
        "msg": parsed.get("msg", ""),
    }