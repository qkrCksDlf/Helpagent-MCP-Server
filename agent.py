"""
LangGraph 에이전트 (대화 히스토리 지원)
- LLM_PROVIDER 환경변수로 모델 전환 (gemini / claude)
- 세션별 대화 히스토리 유지
- 온디바이스 LLM과 같은 JSON 포맷으로 응답
"""
import os
import json
import re
import contextvars
from pathlib import Path
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

load_dotenv()

SYSTEM_PROMPT = """당신은 디지털 기기가 낯선 어르신을 돕는 친절한 AI 도우미입니다. 어려운 IT 용어와 외래어를 피하고, 짧고 다정한 말투(~해요)로 답하세요.
 
【도구 사용】
- 기본은 도구 호출입니다. 사실을 확인해야 하는 질문(날씨, 길찾기, 요리법, 용어, 상식, 최신 정보, 제품·장소)은 내장 지식으로 추측하지 말고 반드시 도구를 먼저 부른 뒤 결과를 받아 답하세요.
- 아래 네 가지만 도구 없이 바로 답합니다. 목록에 없으면 도구를 씁니다.
  인사·감사·감정 응대 / 도우미 자신에 대한 질문 / 산수 / 농담·이야기 창작
 
【출력 형식: 절대 규칙】
JSON 한 줄만 출력하세요. 코드블록이나 부연 설명 금지.
{"intent": "...", "query": "...", "msg": "한국어 답변"}
intent 는 셋 중 하나만: "chat" | "buy_product" | "book_ktx"
도구 이름을 intent 에 적지 마세요.
 
【intent 판정】
- "사줘/주문해줘/시켜줘/구매해줘" 또는 "떨어졌다/다 썼다" → intent="buy_product". 되묻지 말고, query 에는 상품명만 담으세요. (예: "쿠팡에서 햄버거 사줘" → query="햄버거")
- 기차·KTX·SRT 예매/예약 → intent="book_ktx". 출발지·날짜가 없어도 되묻지 마세요.
- 가격이나 리뷰가 궁금한 것뿐이면 구매가 아닙니다. search_shop 호출 후 intent="chat".
- 구매·예매가 다른 요청과 섞이면 구매·예매를 우선하세요. 화면이 전환되어 다른 답변은 보이지 않으므로 msg 에는 짧은 안내만 담습니다.
 
【도구 선택】
- 길찾기: 출발지와 도착지가 모두 있을 때만 find_transit_route(start, end) 호출. 사용자가 말한 장소 이름을 그대로 넣고 근처 상호로 바꾸지 마세요. 안내는 가장 빠른 경로 하나만 단계별로.
- 날씨: 지금·오늘 → get_current_weather / 내일·주말·이번 주 → get_daily_forecast / 오후·저녁·특정 시간 → get_hourly_forecast
- 검색: 맛집·후기 → search_blog / 뉴스 → search_news / 상품 가격 → search_shop / 그 외 모든 사실 질문 → search_web. 검색어에 사용자가 말한 핵심 키워드를 반드시 포함하고, 결과는 2~3개만 요약하세요.
 
【길찾기 출발지 예외】
길찾기에 한해, 출발지를 모르면 도구를 부르지 말고 intent="chat" 으로 "어디에서 출발하실까요?" 하고 먼저 여쭤보세요.
- "@현재위치@위도,경도" 가 메시지에 실제로 있으면 그 문자열을 한 글자도 고치지 말고 그대로 start 에 넣습니다.
- 태그가 없으면 좌표를 지어내거나 "현재 위치"·"여기"·"내 위치" 같은 말을 넣지 마세요.
- 구매·예매의 되묻지 않기 규칙은 길찾기에 적용되지 않습니다.
- 이 예외는 길찾기에만 적용됩니다. 다른 질문은 【도구 사용】 기본 규칙대로 도구를 부르세요.
 
【맥락】
이전 대화의 장소·주제를 기억하고 단답형 발화도 문맥에 연결하세요. 직전이 구매·예매 흐름이면 "그걸로 해줘" 같은 후속도 같은 intent 로 이어갑니다."""


# ─────────────────────────────────────────────────────────────
# 길찾기 출발지 가드
#
# 배경: 홀드아웃 평가에서 "인하대 갈라믄 어떻게 가?" 처럼 출발지가 없는 질문에
#       모델이 start="@현재위치@37.5665,126.9780" 을 스스로 만들어 넣는 사례가 나왔다.
#       (해당 좌표는 서울시청. 앱이 준 GPS 가 아니라 모델이 지어낸 값)
#       형식이 완벽해서 하위 로직은 정상 GPS 로 인식하고, 엉뚱한 출발지의 경로가
#       아무 경고 없이 안내된다. 사용자는 틀린 줄 알 수 없다.
#
# 프롬프트로 금지해도 모델이 어긴 사례이므로, 코드에서 결정적으로 막는다.
# 프롬프트는 1차 방어, 이 가드가 최종 방어다.
# ─────────────────────────────────────────────────────────────
GPS_TAG = "@현재위치@"

# 출발지 자리에 들어오면 안 되는 자리표시자 (공백 제거 후 비교)
_PLACEHOLDER_ORIGINS = {
    "현재위치", "현위치", "지금위치", "내위치", "여기", "현재장소", "출발지", "현재지",
}

# 이번 요청의 사용자 메시지에 실제 GPS 태그가 있었는지
_gps_available = contextvars.ContextVar("gps_available", default=False)


def set_gps_context(user_message: str) -> bool:
    """
    요청 시작 시 호출. 사용자 메시지에 실제 GPS 태그가 있었는지 기록한다.
    호출하지 않으면 기본값 False(가드 활성)이라 안전한 쪽으로 동작한다.
    """
    has_tag = GPS_TAG in (user_message or "")
    _gps_available.set(has_tag)
    return has_tag


def _origin_rejection(start: str, reason: str) -> str:
    print(f"🛑 [출발지 가드] 차단: start={start!r} ({reason})", file=__import__("sys").stderr)
    return json.dumps({
        "error": "NO_ORIGIN",
        "reason": reason,
        "instruction": (
            "출발지를 확인할 수 없습니다. 도구를 다시 부르지 말고, "
            "사용자에게 어디에서 출발하시는지 여쭤보세요. "
            "현재 위치나 좌표를 임의로 지어내면 안 됩니다."
        ),
    }, ensure_ascii=False)


def _check_origin(start: str) -> str | None:
    """차단해야 하면 도구 결과로 돌려줄 JSON 문자열, 통과면 None."""
    raw = str(start or "")
    if GPS_TAG in raw and not _gps_available.get():
        return _origin_rejection(raw, "사용자 메시지에 없던 GPS 태그를 모델이 생성함")
    if not raw.startswith(GPS_TAG):
        squashed = re.sub(r"\s+", "", raw)
        if squashed in _PLACEHOLDER_ORIGINS:
            return _origin_rejection(raw, "실제 장소가 아닌 자리표시자")
        if not squashed:
            return _origin_rejection(raw, "출발지가 비어 있음")
    return None


def _wrap_transit_guard(tool):
    """find_transit_route 를 감싸 출발지를 검증한다. 실패하면 원본을 그대로 쓴다."""
    try:
        async def _guarded(**kwargs):
            blocked = _check_origin(kwargs.get("start"))
            if blocked:
                return blocked
            return await tool.ainvoke(kwargs)

        return StructuredTool(
            name=tool.name,
            description=tool.description,
            args_schema=tool.args_schema,
            coroutine=_guarded,
        )
    except Exception as e:
        print(f"⚠️ 출발지 가드 적용 실패, 원본 도구 사용: {e}")
        return tool


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
        model_name = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
        print(f"🤖 Using {model_name}")
        return ChatGoogleGenerativeAI(
            model=model_name,
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

    # 길찾기 도구에만 출발지 가드를 씌운다
    tools = [_wrap_transit_guard(t) if t.name == "find_transit_route" else t for t in tools]

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

        # 🌟 이번 요청에 실제 GPS 태그가 있었는지 기록 (출발지 가드가 참조)
        has_gps = set_gps_context(user_message)
        print(f"📍 GPS 태그 {'있음' if has_gps else '없음'}")

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
