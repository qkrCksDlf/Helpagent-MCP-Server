"""
LangGraph 에이전트 (대화 히스토리 지원)
- LLM_PROVIDER 환경변수로 모델 전환 (gemini / claude)
- 세션별 대화 히스토리 유지
- 온디바이스 LLM과 같은 JSON 포맷으로 응답
"""
import os
import json
import re
from pathlib import Path
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

load_dotenv()

SYSTEM_PROMPT = """당신은 디지털 취약계층(어르신 등)을 위한 도우미입니다.

【중요】 반드시 JSON 형식으로만 응답하세요. 다른 텍스트는 절대 포함하지 마세요.
코드블록(```)도 쓰지 마세요. 순수 JSON만 출력하세요.

응답 형식:
{"intent": "...", "query": "...", "msg": "한국어 답변"}

⚠️ intent는 반드시 다음 셋 중 하나만 사용하세요:
- "chat"  (대화, 정보 안내, 날씨/길찾기/검색 결과 안내, 추가 정보 요청)
- "buy_product"  (쇼핑 요청)
- "book_ktx"  (기차 예매)

도구 이름(find_transit_route, get_current_weather, search_blog 등)을 절대 intent에 넣지 마세요.
도구는 호출만 하고, 결과를 받은 후 항상 intent="chat"으로 답변하세요.

잘못된 예: {"intent": "find_transit_route", ...}  ❌ 절대 금지!
올바른 예: {"intent": "chat", "msg": "강남역에서 홍대까지 2호선으로 18분이에요"}  ✅

【대화 맥락 유지】
이전 대화 내용을 기억하고 활용하세요.
사용자가 이전 질문에 대한 답변을 이어서 하는 경우 자연스럽게 연결하세요.
예시:
- 이전 AI: "어디에서 출발하세요?"
- 현재 사용자: "인하대학교에서요"
- → 이전 도착지와 합쳐서 find_transit_route 호출

규칙:
1. 날씨를 물으면 적절한 날씨 도구 호출:
   - "지금 날씨", "오늘 날씨" → get_current_weather
   - "내일 날씨", "이번 주", "주말" → get_daily_forecast
   - "오후", "저녁", "몇 시쯤" → get_hourly_forecast

2. 길찾기는 find_transit_route(start, end) 호출:
   - "어떻게 가?", "가는 길", "타고 가" 등
   - 출발지나 도착지가 부족하면 도구 호출 말고 intent="chat"으로 물어보세요.
   - 이전 대화에서 정보가 있으면 활용하세요.

3. 검색이 필요한 질문:
   - 맛집, 후기, 리뷰, 여행 → search_blog
   - 뉴스, 사건, 최신 소식 → search_news
   - 상품, 가격, 쇼핑 정보 → search_shop
   - 일반 정보, 정의, 지식 → search_web

4. 도구 결과를 받으면 intent="chat"으로 자연스럽게 답변:
   - 답변은 짧고 친절하게, 어르신도 이해하기 쉽게
   - 검색 결과가 많으면 2-3개만 골라서 안내
   - 길찾기는 가장 빠른 경로 하나만 단계별로 안내

5. 쇼핑/기차 요청이면 도구 호출 없이 바로 해당 intent 반환:
   - 쇼핑 → intent="buy_product", query=상품명
   - 기차 → intent="book_ktx"
"""


def build_llm():
    """환경변수에 따라 LLM 선택"""
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()

    if provider == "claude":
        print("🤖 Using Claude Haiku 4.5")
        return ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0.2,
        )
    elif provider == "gemini":
        print("🤖 Using Gemini 3.5 Flash")
        return ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.2,
        )
    else:
        raise ValueError(f"알 수 없는 LLM_PROVIDER: {provider}")


_agent = None

# 🌟 세션별 대화 히스토리 저장소 (메모리)
_sessions: dict[str, list[BaseMessage]] = {}

# 🌟 한 세션당 보관할 최대 메시지 수 (사용자+AI 합쳐서)
# 너무 길면 토큰 폭주, 너무 짧으면 맥락 손실
MAX_HISTORY = 10


async def get_agent():
    global _agent
    if _agent is not None:
        return _agent

    weather_script = str(Path(__file__).parent / "weather_mcp.py")
    search_script = str(Path(__file__).parent / "search_mcp.py")
    transit_script = str(Path(__file__).parent / "transit_mcp.py")

    print(f"🔌 MCP weather: {weather_script}")
    print(f"🔌 MCP search: {search_script}")
    print(f"🔌 MCP transit: {transit_script}")

    client = MultiServerMCPClient({
        "weather": {
            "command": "python",
            "args": [weather_script],
            "transport": "stdio",
        },
        "search": {
            "command": "python",
            "args": [search_script],
            "transport": "stdio",
        },
        "transit": {
            "command": "python",
            "args": [transit_script],
            "transport": "stdio",
        },
    })
    tools = await client.get_tools()

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


def reset_session(session_id: str = "default"):
    """세션 초기화 (대화 기록 삭제)"""
    if session_id in _sessions:
        del _sessions[session_id]
        print(f"🗑️ 세션 '{session_id}' 초기화됨")


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


async def run_agent(user_message: str, session_id: str = "default") -> dict:
    try:
        agent = await get_agent()

        # 🌟 세션 히스토리 가져오기
        history = _sessions.get(session_id, [])

        # 새 사용자 메시지 추가
        new_message = HumanMessage(content=user_message)
        messages_to_send = history + [new_message]

        print(f"📚 세션 '{session_id}' 히스토리 길이: {len(history)}")

        result = await agent.ainvoke({"messages": messages_to_send})

        all_messages = result["messages"]

        # 디버그 출력
        print("=" * 60)
        print(f"📨 사용자 입력: {user_message}")
        print(f"🔑 세션 ID: {session_id}")
        print("-" * 60)
        for i, msg in enumerate(all_messages):
            msg_type = msg.__class__.__name__
            content = msg.content if hasattr(msg, 'content') else ''
            if isinstance(content, list):
                content_str = json.dumps(content, ensure_ascii=False)[:200]
            else:
                content_str = str(content)[:200]
            print(f"[{i}] {msg_type}: {content_str}")

            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"    🔧 도구 호출: {tc.get('name')}({tc.get('args')})")
            if msg_type == 'ToolMessage':
                tool_name = getattr(msg, 'name', '?')
                print(f"    ↩️ 도구 결과 ({tool_name})")
        print("=" * 60)

        final_text = all_messages[-1].content if all_messages else ""
        if isinstance(final_text, list):
            final_text = "".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in final_text
            )

        print(f"📩 최종 응답: {final_text}")
        parsed = parse_json_response(final_text)

        # 🌟 히스토리 업데이트: 사용자 메시지 + AI 응답(msg)만 저장
        # 도구 호출/결과는 다음 대화의 컨텍스트로 부담스러우니 제외
        ai_msg_text = parsed.get("msg", "")
        final_ai_message = AIMessage(content=ai_msg_text)
        new_history = history + [new_message, final_ai_message]

        # 너무 길면 오래된 거 잘라내기 (최근 MAX_HISTORY개만 유지)
        if len(new_history) > MAX_HISTORY:
            new_history = new_history[-MAX_HISTORY:]

        _sessions[session_id] = new_history
        print(f"💾 세션 '{session_id}' 저장됨 (총 {len(new_history)}개 메시지)")

        return {
            "intent": parsed.get("intent", "chat"),
            "query": parsed.get("query", ""),
            "msg": parsed.get("msg", ""),
        }

    except Exception as e:
        error_str = str(e)
        print(f"❌ 에이전트 에러: {error_str[:300]}")

        # 한도 초과 감지
        if "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower() or "429" in error_str:
            return {
                "intent": "chat",
                "query": "",
                "msg": "⏰ 죄송해요, 오늘 사용량이 다 됐어요. 잠시 후 다시 시도해 주세요.",
            }
        # 기타 에러
        return {
            "intent": "chat",
            "query": "",
            "msg": "⚠️ 잠시 문제가 생겼어요. 다시 한 번 말씀해 주세요.",
        }
