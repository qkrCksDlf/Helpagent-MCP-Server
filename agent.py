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

⚠️ 길찾기 질문에는 절대 당신의 지식으로 경로를 지어내지 마세요.
출발지·도착지가 있으면 무조건 find_transit_route 도구를 먼저 호출해야 합니다.
도구 없이 역 이름이나 버스 번호를 직접 답하면 안 됩니다.

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

2. 길찾기 질문(가는 법, 어떻게 가, 경로, 가려면 등):
   - 출발지와 도착지가 **둘 다 있으면 반드시 find_transit_route(start, end) 도구를 호출하세요.**
   - 절대 당신의 지식으로 경로를 지어내지 마세요. 역 이름, 버스 번호, 노선, 소요시간을
     추측해서 답하는 것은 금지입니다. 반드시 도구 결과만 사용하세요.
   - 출발지나 도착지가 없을 때만 도구 없이 intent="chat"으로 되물어보세요.
   - 이전 대화에 출발지/도착지 정보가 있으면 합쳐서 도구를 호출하세요.
   - 장소 이름이 모호하면(예: "롯데월드") 사용자가 말한 그대로 도구에 넘기세요.
   - 출발지가 "@현재위치@..." 형식으로 주어지면 그 값을 **그대로** start 에 넣어 도구를 호출하세요.
     (이것은 GPS 현재 위치 좌표이므로 절대 수정하거나 다른 장소명으로 바꾸지 마세요.)

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


def _extract_tool_result(msg, tool_name: str) -> dict | None:
    """ToolMessage 에서 특정 도구의 결과(dict)를 꺼낸다."""
    if msg.__class__.__name__ != "ToolMessage":
        return None
    if getattr(msg, "name", "") != tool_name:
        return None

    content = msg.content
    # ToolMessage.content 는 문자열이거나 [{"type":"text","text":"..."}] 형태
    if isinstance(content, list):
        text = "".join(
            b.get("text", "") if isinstance(b, dict) else str(b)
            for b in content
        )
    else:
        text = str(content)

    try:
        return json.loads(text)
    except Exception:
        return None


_MODE_LABEL = {"subway": "지하철", "bus": "버스", "walk": "도보"}


def _summarize_route(route: dict) -> str:
    """길찾기 구조 데이터를 어르신용 짧은 한 줄 요약으로 (음성 안내/말풍선 백업용)."""
    routes = route.get("routes", [])
    if not routes:
        return f"{route.get('from','출발지')}에서 {route.get('to','도착지')}까지 경로를 찾지 못했어요."

    best = routes[0]
    frm = route.get("from", "출발지")
    to = route.get("to", "도착지")
    total = best.get("total_time_min")
    transfers = best.get("transfer_count", 0)

    parts = [f"{frm}에서 {to}까지"]
    if total:
        parts.append(f"약 {total}분")
    if transfers:
        parts.append(f"환승 {transfers}번")
    head = ", ".join(parts) + " 걸려요."

    # 교통수단만 순서대로 (도보 제외하고 핵심 탈것만)
    rides = [s for s in best.get("steps", []) if s.get("mode") in ("subway", "bus")]
    if rides:
        seq = " → ".join(
            f"{s.get('line','')} {_MODE_LABEL.get(s.get('mode'),'')}".strip()
            for s in rides
        )
        head += f" {seq} 순서로 타시면 돼요."
    return head


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

        # 🌟 길찾기 결과 가로채기:
        #    find_transit_route 가 성공(routes 존재)했으면 LLM 요약을 버리고
        #    구조화 데이터를 그대로 안드로이드로 패스스루한다.
        #    (LLM 이 경로를 줄글로 뭉개는 걸 방지 → 카드 UI 로 렌더 가능)
        for msg in all_messages:
            route = _extract_tool_result(msg, "find_transit_route")
            if route and route.get("routes"):
                summary = _summarize_route(route)
                print(f"🗺️ 길찾기 구조화 응답으로 직접 반환 (LLM 요약 건너뜀)")

                # 히스토리에는 요약 한 줄만 남겨 맥락 유지 (구조 데이터는 부담되니 제외)
                final_ai_message = AIMessage(content=summary)
                new_history = history + [new_message, final_ai_message]
                if len(new_history) > MAX_HISTORY:
                    new_history = new_history[-MAX_HISTORY:]
                _sessions[session_id] = new_history

                return {
                    "intent": "transit_route",
                    "query": f"{route.get('from','')} → {route.get('to','')}",
                    "msg": summary,
                    "route": route,
                }
            # 길찾기는 호출했는데 실패(error)한 경우는 가로채지 않고
            # LLM 이 웹검색 fallback 등으로 답하도록 그대로 둔다.

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
            "route": None,
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
