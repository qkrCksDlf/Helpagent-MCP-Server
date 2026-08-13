#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HelpAgent 클라우드 에이전트 — 단일 라우팅 평가 하네스 (v2)

이전 하네스와의 차이:
  - 복합 요청(coverage) / 멀티턴 제거. 단일 요청만 측정한다.
  - 라우팅 정확도 외에 '슬롯 정확도'를 추가로 측정한다.
      · coupang : 구매 검색어가 올바르게 추출됐는가
      · search  : 네이버 검색 도구에 넘긴 질의어가 올바른가
      · transit : 출발지/도착지가 올바르게 추출됐는가 (순서 포함)
      · weather : 지역명이 올바르게 추출됐는가
      · korail  : 슬롯 없음. 라우팅만 맞으면 통과
  - 세 가지 지표를 분리해서 낸다.
      routing : 올바른 capability 로 갔는가
      slot    : (라우팅 성공 케이스 중) 슬롯까지 맞았는가
      e2e     : 라우팅 AND 슬롯 둘 다 맞았는가  ← 실사용 기준

사용법 (agent.py 와 같은 폴더에서):
  python eval_harness_single.py                        # testset_single.json
  python eval_harness_single.py testset_holdout.json   # 홀드아웃
  python eval_harness_single.py testset_single.json --tag before
  python eval_harness_single.py --stop-on-error        # 첫 에러에서 중단

결과 파일명은 테스트셋 이름에서 자동으로 갈라진다.
  testset_single.json  → eval_single_results_single.json / .png
  testset_holdout.json → eval_single_results_holdout.json / .png
따라서 서로 덮어쓸 일이 없다.
"""
import os
import re
import sys
import json
import time
import asyncio
import platform
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# langchain / agent.py 는 무거우므로 실제 실행 직전에 import 한다.
# 덕분에 --help 나 테스트셋 경로 오타는 환경 없이도 바로 확인된다.
HumanMessage = None
get_agent = None
parse_json_response = None
_extract_tool_result = None


def _load_agent_deps():
    """agent.py 와 langchain 을 지연 로드한다."""
    global HumanMessage, get_agent, parse_json_response, _extract_tool_result
    from langchain_core.messages import HumanMessage as _HM
    from agent import (
        get_agent as _ga,
        parse_json_response as _pjr,
        _extract_tool_result as _etr,
    )
    HumanMessage = _HM
    get_agent = _ga
    parse_json_response = _pjr
    _extract_tool_result = _etr

# ── capability ↔ 도구 매핑 ────────────────────────────────────
WEATHER_TOOLS = {"get_current_weather", "get_daily_forecast", "get_hourly_forecast"}
SEARCH_TOOLS = {"search_blog", "search_news", "search_shop", "search_web"}
TRANSIT_TOOLS = {"find_transit_route"}
KNOWN_TOOLS = WEATHER_TOOLS | SEARCH_TOOLS | TRANSIT_TOOLS

CAPS = ["transit", "weather", "search", "coupang", "korail", "direct"]
# 라벨 충돌 시 우선순위 — 도구 호출이 intent 보다 강한 신호
PRIORITY = CAPS

# 슬롯 검증 대상 capability. 여기서 빼면 해당 카테고리는 라우팅만 본다.
SLOT_CHECK_CAPS = {"transit", "weather", "search", "coupang"}

SLEEP_BETWEEN = 0.5   # 케이스 간 대기(초). rate limit 완화용
TAG = ""              # 결과 파일 접미사. main() 에서 테스트셋 이름으로 채워진다
FAIL_FAST = False     # True 면 첫 에러에서 전체 트레이스를 찍고 중단


def _parse_args():
    import argparse
    ap = argparse.ArgumentParser(
        description="HelpAgent 단일 라우팅 평가 하네스",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("testset", nargs="?", default="testset_single.json",
                    help="테스트셋 JSON 경로 (기본: testset_single.json)")
    ap.add_argument("--tag", default=None,
                    help="결과 파일 접미사. 생략하면 테스트셋 파일명에서 자동 생성")
    ap.add_argument("--sleep", type=float, default=0.5,
                    help="케이스 간 대기 초 (기본 0.5)")
    ap.add_argument("--stop-on-error", action="store_true",
                    help="첫 에러에서 중단 (기본은 끝까지 진행)")
    return ap.parse_args()


def _normalize_testset(raw, path):
    """
    테스트셋을 리스트로 정규화한다. 아래 형태를 모두 받는다.
      1) [ {...}, {...} ]                          — 표준
      2) { "cases": [ {...} ] }                    — 래핑된 형태
      3) { "transit": [ {...} ], "weather": [...] } — 카테고리별 묶음
    3번은 키를 expected 로 자동 채운다.
    """
    if isinstance(raw, list):
        cases = list(raw)
    elif isinstance(raw, dict):
        cases = None
        for key in ("cases", "testset", "items", "data"):
            if isinstance(raw.get(key), list):
                cases = list(raw[key])
                break
        if cases is None:
            cases = []
            for cap, group in raw.items():
                if not isinstance(group, list):
                    print(f"❌ 테스트셋 형식을 알 수 없습니다: {os.path.basename(path)}")
                    print(f"   최상위 키: {list(raw)[:10]}")
                    print(f"   '{cap}' 의 값이 리스트가 아닙니다 ({type(group).__name__}).")
                    return None
                for c in group:
                    if isinstance(c, dict):
                        c.setdefault("expected", cap)
                        cases.append(c)
            print(f"ℹ️  카테고리별 dict 로 저장돼 있어 {len(raw)}개 그룹을 평탄화했습니다.")
    else:
        print(f"❌ 테스트셋이 리스트도 dict 도 아닙니다: {type(raw).__name__}")
        return None

    if not cases:
        print(f"❌ 테스트셋이 비어 있습니다: {os.path.basename(path)}")
        return None

    bad = []
    for i, c in enumerate(cases):
        if not isinstance(c, dict):
            bad.append(f"  [{i}] dict 가 아님 ({type(c).__name__})")
            continue
        c.setdefault("id", f"case{i + 1:03d}")
        for field in ("query", "expected"):
            if not c.get(field):
                bad.append(f"  [{c['id']}] '{field}' 없음")
    if bad:
        print(f"❌ 테스트셋에 형식 오류 {len(bad)}건:")
        for line in bad[:10]:
            print(line)
        if len(bad) > 10:
            print(f"  ... 외 {len(bad) - 10}건")
        return None

    unknown = sorted({c["expected"] for c in cases} - set(CAPS))
    if unknown:
        print(f"⚠️  알 수 없는 expected 값: {unknown}  (허용: {CAPS})")

    dup = [k for k, n in __import__("collections").Counter(c["id"] for c in cases).items() if n > 1]
    if dup:
        print(f"⚠️  중복 id: {dup[:10]}")

    return cases


def _derive_tag(path):
    base = os.path.splitext(os.path.basename(path))[0]
    if base.startswith("testset_"):
        base = base[len("testset_"):]
    return base or "run"

_PUNCT = re.compile(r"[\s\-_·,./()\[\]{}'\"“”‘’!?~:;+]+")


def norm(s):
    """공백·문장부호 제거 + 소문자화. 슬롯 비교는 전부 이 위에서 한다."""
    return _PUNCT.sub("", str(s or "")).lower()


def _iter_strings(obj):
    """중첩 dict/list 안의 모든 문자열 값을 뽑는다 (도구 인자 이름을 몰라도 되게)."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _iter_strings(v)


def args_haystack(tool_calls, names):
    """지정한 도구들의 인자에 들어간 문자열을 하나로 합친다."""
    out = []
    for tc in tool_calls:
        if tc.get("name") in names:
            out.extend(_iter_strings(tc.get("args")))
    return " | ".join(out)


# find_transit_route 인자 키 후보 (MCP 서버 구현에 따라 다를 수 있음)
START_KEYS = ("start", "from", "origin", "departure", "src")
END_KEYS = ("end", "to", "destination", "arrival", "dest")


def transit_args(tool_calls):
    """LLM 이 find_transit_route 에 넘긴 출발/도착 원문을 꺼낸다.
    키 이름을 못 찾으면 인자 순서(삽입 순서)로 fallback."""
    for tc in tool_calls:
        if tc.get("name") not in TRANSIT_TOOLS:
            continue
        args = tc.get("args") or {}
        if not isinstance(args, dict):
            continue
        low = {str(k).lower(): v for k, v in args.items()}
        start = next((low[k] for k in START_KEYS if isinstance(low.get(k), str)), None)
        end = next((low[k] for k in END_KEYS if isinstance(low.get(k), str)), None)
        if start is None or end is None:
            strs = [v for v in args.values() if isinstance(v, str)]
            if len(strs) >= 2:
                start = start if start is not None else strs[0]
                end = end if end is not None else strs[1]
        return start or "", end or ""
    return "", ""


def slot_hit(actual, spec):
    """spec: {"any_of":[...]} 또는 {"all_of":[...]}"""
    a = norm(actual)
    if not a:
        return False
    if "any_of" in spec:
        return any(norm(x) in a for x in spec["any_of"])
    if "all_of" in spec:
        return all(norm(x) in a for x in spec["all_of"])
    return False


def caps_from_run(intent, tools_called):
    t = {tc.get("name") for tc in tools_called}
    caps = set()
    if t & TRANSIT_TOOLS:
        caps.add("transit")
    if t & WEATHER_TOOLS:
        caps.add("weather")
    if t & SEARCH_TOOLS:
        caps.add("search")
    if intent == "buy_product":
        caps.add("coupang")
    if intent == "book_ktx":
        caps.add("korail")
    if not caps:
        caps.add("direct")
    return caps


def primary_cap(caps):
    for c in PRIORITY:
        if c in caps:
            return c
    return "direct"


async def run_traced(agent, user_message):
    """단일 턴 실행. 호출 도구를 인자까지 함께 캡처한다."""
    t0 = time.perf_counter()
    result = await agent.ainvoke({"messages": [HumanMessage(content=user_message)]})
    latency = time.perf_counter() - t0
    all_messages = result["messages"]

    tools_called = []
    for m in all_messages:
        for tc in (getattr(m, "tool_calls", None) or []):
            if tc.get("name"):
                tools_called.append({"name": tc["name"], "args": tc.get("args") or {}})

    # transit 가로채기 우선 (run_agent 와 동일)
    intent, query, route = None, "", None
    for m in all_messages:
        r = _extract_tool_result(m, "find_transit_route")
        if r and r.get("routes"):
            route, intent = r, "transit_route"
            query = f"{r.get('from', '')} → {r.get('to', '')}"
            break
    if intent is None:
        final = all_messages[-1].content if all_messages else ""
        if isinstance(final, list):
            final = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in final)
        parsed = parse_json_response(final)
        intent = parsed.get("intent", "chat")
        query = parsed.get("query", "")

    return {
        "intent": intent,
        "query": query,
        "route": route,
        "tools_called": tools_called,
        "latency": latency,
    }


def check_slots(cap, expected_slots, out):
    """
    반환: (slot_ok, detail)
      slot_ok = None  → 검증 대상 아님 (기대 슬롯 없음 / 대상 capability 아님)
    """
    if not expected_slots or cap not in SLOT_CHECK_CAPS:
        return None, {}

    tools = out["tools_called"]
    detail = {}

    if cap == "transit":
        # 슬롯 = LLM 이 도구에 넘긴 원문 (프롬프트/추출 능력의 지표)
        arg_start, arg_end = transit_args(tools)
        ok = True
        for key, actual in (("from", arg_start), ("to", arg_end)):
            spec = expected_slots.get(key)
            if not spec:
                continue
            hit = slot_hit(actual, spec)
            detail[key] = {"expected": spec, "actual": actual, "ok": hit}
            ok = ok and hit

        # 지오코딩 = 도구가 그 원문을 어떤 장소로 해석했는지 (MCP/ODsay 의 지표)
        route = out["route"] or {}
        for key, arg_val in (("from", arg_start), ("to", arg_end)):
            resolved = route.get(key)
            if not resolved:
                continue
            detail[f"resolved_{key}"] = {
                "sent": arg_val,
                "resolved": resolved,
                "ok": slot_hit(resolved, expected_slots.get(key, {"any_of": [arg_val]})),
                "geocode_only": True,
            }
        return ok, detail

    if cap == "search":
        actual = args_haystack(tools, SEARCH_TOOLS)
    elif cap == "weather":
        actual = args_haystack(tools, WEATHER_TOOLS)
    elif cap == "coupang":
        actual = out["query"]  # 앱 전환용 검색어는 응답 JSON 의 query 필드
    else:
        return None, {}

    ok = True
    for key, spec in expected_slots.items():
        hit = slot_hit(actual, spec)
        detail[key] = {"expected": spec, "actual": actual, "ok": hit}
        ok = ok and hit
    return ok, detail


def score_row(case, out):
    caps = caps_from_run(out["intent"], out["tools_called"])
    primary = primary_cap(caps)
    expected = case["expected"]
    routing_ok = (primary == expected)

    slot_ok, slot_detail = (None, {})
    if routing_ok:
        slot_ok, slot_detail = check_slots(expected, case.get("slots"), out)

    if slot_ok is None:
        e2e_ok = routing_ok
    else:
        e2e_ok = routing_ok and slot_ok

    names = [tc["name"] for tc in out["tools_called"]]
    return {
        "id": case["id"],
        "query": case["query"],
        "expected": expected,
        "predicted": primary,
        "predicted_caps": sorted(caps),
        "intent": out["intent"],
        "tools_called": names,
        "tool_args": [tc["args"] for tc in out["tools_called"]],
        "unknown_tools": [n for n in names if n not in KNOWN_TOOLS],
        "extracted_query": out["query"],
        "routing_ok": routing_ok,
        "slot_ok": slot_ok,
        "slot_detail": slot_detail,
        "e2e_ok": e2e_ok,
        "latency_s": round(out["latency"], 2),
    }


def _print_row(r):
    if not r["routing_ok"]:
        mark = "❌ route"
    elif r["slot_ok"] is False:
        mark = "⚠️  slot "
    else:
        mark = "✅       "
    tools = ",".join(r["tools_called"]) or "-"
    line = f"  {mark} [{r['id']}] {r['expected']} → {r['predicted']}  tools=[{tools}]  {r['latency_s']}s"
    if r["slot_ok"] is False:
        bad = [f"{k}: 기대={v['expected']} 실제='{v['actual']}'"
               for k, v in r["slot_detail"].items()
               if not v.get("geocode_only") and not v["ok"]]
        line += "\n           슬롯 불일치 → " + " / ".join(bad)
    print(line)


async def main(args):
    global TAG, FAIL_FAST, SLEEP_BETWEEN
    SLEEP_BETWEEN = args.sleep
    FAIL_FAST = args.stop_on_error

    here = os.path.dirname(os.path.abspath(__file__))
    ts_path = args.testset
    if not os.path.isabs(ts_path):
        ts_path = os.path.join(here, ts_path)

    if not os.path.exists(ts_path):
        print(f"❌ 테스트셋을 찾을 수 없습니다: {ts_path}")
        found = sorted(f for f in os.listdir(here) if f.startswith("testset") and f.endswith(".json"))
        if found:
            print("   이 폴더에 있는 테스트셋: " + ", ".join(found))
        return

    try:
        with open(ts_path, encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ JSON 파싱 실패: {os.path.basename(ts_path)}")
        print(f"   {e}")
        return

    testset = _normalize_testset(raw, ts_path)
    if testset is None:
        return

    TAG = args.tag or _derive_tag(ts_path)

    print("=" * 60)
    print(f"📄 테스트셋 : {os.path.basename(ts_path)}  ({len(testset)}개)")
    print(f"🏷️  결과 태그 : {TAG}  → eval_single_results_{TAG}.json / .png")
    print(f"🆔 첫 케이스 : {testset[0]['id']}   마지막: {testset[-1]['id']}")
    print(f"⚙️  LLM_PROVIDER = {os.getenv('LLM_PROVIDER', 'gemini')}")
    print(f"🔍 슬롯 검증 대상: {', '.join(sorted(SLOT_CHECK_CAPS))}")
    print("=" * 60)
    print("🔌 에이전트 초기화 중...")
    _load_agent_deps()
    agent = await get_agent()
    print("✅ 준비 완료. 평가 시작.\n")

    rows = []
    for case in testset:
        try:
            out = await run_traced(agent, case["query"])
            r = score_row(case, out)
        except Exception as e:
            import traceback
            full = f"{type(e).__name__}: {e}"
            r = {
                "id": case["id"], "query": case["query"], "expected": case["expected"],
                "error": full, "traceback": traceback.format_exc(),
                "routing_ok": False, "slot_ok": None,
                "e2e_ok": False, "latency_s": None, "tools_called": [],
                "unknown_tools": [], "slot_detail": {},
            }
            print(f"  ❌ err  [{case['id']}]")
            print("-" * 60)
            traceback.print_exc()
            print("-" * 60)
            if FAIL_FAST:
                print("\n⛔ --stop-on-error → 첫 에러에서 중단합니다.")
                print("   원인을 고친 뒤 옵션 없이 다시 실행하세요.")
                rows.append(r)
                _report(rows, here, len(testset))
                return
        rows.append(r)
        if "error" not in r:
            _print_row(r)
        await asyncio.sleep(SLEEP_BETWEEN)

    _report(rows, here, len(testset))


def _rate(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return None, 0, 0
    ok = sum(1 for v in vals if v)
    return ok / len(vals), ok, len(vals)


def _report(rows, here, n_cases):
    errors = [r for r in rows if r.get("error")]

    print("\n" + "=" * 60)
    print("📈 결과 요약  [%s]  단일 라우팅 하네스, n=%d" % (TAG, n_cases))
    print("=" * 60)

    for label, key in (("라우팅 정확도", "routing_ok"),
                       ("슬롯 정확도  ", "slot_ok"),
                       ("종합(e2e)   ", "e2e_ok")):
        rate, ok, tot = _rate(rows, key)
        if rate is None:
            print(f"{label}: 대상 없음")
        else:
            print(f"{label}: {rate * 100:5.1f}%  ({ok}/{tot})")

    by_cap = defaultdict(list)
    for r in rows:
        by_cap[r["expected"]].append(r)

    print("\n[카테고리별]")
    print(f"  {'capability':10s} {'라우팅':>12s} {'슬롯':>12s} {'종합':>12s}")
    cat_stats = {}
    for cap in CAPS:
        g = by_cap.get(cap)
        if not g:
            continue
        rr, ro, rt = _rate(g, "routing_ok")
        sr, so, st = _rate(g, "slot_ok")
        er, eo, et = _rate(g, "e2e_ok")
        cat_stats[cap] = {
            "routing": rr, "slot": sr, "e2e": er,
            "n": len(g), "n_slot_checked": st,
        }
        s_txt = f"{sr * 100:5.1f}% ({so}/{st})" if sr is not None else "         -"
        print(f"  {cap:10s} {rr * 100:5.1f}% ({ro}/{rt}) {s_txt:>12s} {er * 100:5.1f}% ({eo}/{et})")

    mis = [r for r in rows if r.get("routing_ok") is False and not r.get("error")]
    if mis:
        print(f"\n[라우팅 오분류 {len(mis)}건]")
        for r in mis:
            tools = ",".join(r["tools_called"]) or "-"
            print(f"  {r['id']}: '{r['query']}'")
            print(f"       기대={r['expected']} → 예측={r['predicted']} (intent={r.get('intent')}, tools=[{tools}])")

    slot_fail = [r for r in rows if r.get("slot_ok") is False]
    if slot_fail:
        print(f"\n[슬롯 불일치 {len(slot_fail)}건] — 라우팅은 맞았으나 인자 추출 실패")
        for r in slot_fail:
            print(f"  {r['id']}: '{r['query']}'")
            for k, v in r["slot_detail"].items():
                if not v.get("geocode_only") and not v["ok"]:
                    print(f"       {k}: 기대={v['expected']}  실제='{v['actual']}'")

    geo = []
    for r in rows:
        for k, v in (r.get("slot_detail") or {}).items():
            if v.get("geocode_only") and not v["ok"]:
                geo.append((r["id"], r["query"], k.replace("resolved_", ""), v["sent"], v["resolved"]))
    if geo:
        print(f"\n[지오코딩 불일치 {len(geo)}건] — LLM 은 맞게 넘겼으나 도구가 다른 장소로 해석")
        print("  → 시스템 프롬프트가 아니라 장소 검색(ODsay POI) 쪽 문제입니다.")
        for cid, q, key, sent, resolved in geo:
            print(f"  {cid} [{key}]: '{q}'")
            print(f"       LLM 전달='{sent}'  →  도구 해석='{resolved}'")

    unk = [(r["id"], r["unknown_tools"]) for r in rows if r.get("unknown_tools")]
    if unk:
        print(f"\n[미등록 도구 호출 {len(unk)}건]")
        for cid, tools in unk:
            print(f"  {cid}: {tools}")

    lats = [r["latency_s"] for r in rows if r.get("latency_s") is not None]
    if lats:
        s = sorted(lats)
        p50 = s[len(s) // 2]
        p95 = s[min(len(s) - 1, int(len(s) * 0.95))]
        print(f"\n[지연] 평균 {sum(lats) / len(lats):.2f}s  p50 {p50:.2f}s  p95 {p95:.2f}s")

    if errors:
        print(f"\n⚠️ 실행 에러 {len(errors)}건: " + ", ".join(r["id"] for r in errors))

    rr, _, _ = _rate(rows, "routing_ok")
    sr, _, _ = _rate(rows, "slot_ok")
    er, _, _ = _rate(rows, "e2e_ok")
    out = {
        "harness": "single-routing-v2",
        "n_cases": n_cases,
        "slot_check_caps": sorted(SLOT_CHECK_CAPS),
        "summary": {
            "routing_accuracy": rr,
            "slot_accuracy": sr,
            "e2e_accuracy": er,
            "n_geocode_mismatch": len(geo),
            "n_errors": len(errors),
            "latency_mean_s": (sum(lats) / len(lats)) if lats else None,
        },
        "category": cat_stats,
        "rows": rows,
    }
    suffix = f"_{TAG}" if TAG else ""
    jpath = os.path.join(here, f"eval_single_results{suffix}.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n💾 {os.path.basename(jpath)} 저장")

    _plot(cat_stats, rr, sr, er, here, suffix)


# ── 그래프 ────────────────────────────────────────────────────
KR_LABELS = {
    "transit": "대중교통", "weather": "날씨", "search": "검색",
    "coupang": "쿠팡", "korail": "코레일", "direct": "직접응답",
}


def _pick_korean_font():
    """한글 폰트를 못 찾으면 None 을 돌려주고, 그래프는 영문 라벨로 그린다."""
    try:
        from matplotlib import font_manager
    except Exception:
        return None
    prefer = {
        "Windows": ["Malgun Gothic", "NanumGothic", "Batang"],
        "Darwin": ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic"],
    }.get(platform.system(), ["NanumGothic", "Noto Sans CJK KR", "Noto Sans KR", "UnDotum"])
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in prefer:
        if name in installed:
            return name
    for f in font_manager.fontManager.ttflist:
        if any(k in f.name for k in ("Gothic", "Nanum", "CJK", "Malgun")):
            return f.name
    return None


def _plot(cat_stats, rr, sr, er, here, suffix):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        print("ℹ️ matplotlib 없음 → 그래프 생략")
        return

    font = _pick_korean_font()
    if font:
        matplotlib.rcParams["font.family"] = font
        labeler = lambda c: KR_LABELS.get(c, c)
        titles = ("라우팅", "슬롯", "종합")
        ylab, title = "정확도 (%)", "capability 별 라우팅 · 슬롯 정확도"
    else:
        print("ℹ️ 한글 폰트를 못 찾아 영문 라벨로 그립니다 (NanumGothic 설치 권장)")
        labeler = lambda c: c
        titles = ("routing", "slot", "e2e")
        ylab, title = "accuracy (%)", "Routing and slot accuracy by capability"
    matplotlib.rcParams["axes.unicode_minus"] = False

    caps = [c for c in CAPS if c in cat_stats]
    labels = [labeler(c) for c in caps] + [labeler("OVERALL") if not font else "전체"]

    def series(key, overall):
        vals = [cat_stats[c][key] for c in caps]
        vals = [(v * 100 if v is not None else np.nan) for v in vals]
        vals.append(overall * 100 if overall is not None else np.nan)
        return vals

    data = [series("routing", rr), series("slot", sr), series("e2e", er)]
    colors = ["#1A1A1A", "#DCDCDC", "#B4B4B4"]

    x = np.arange(len(labels))
    w = 0.27
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, (vals, name, col) in enumerate(zip(data, titles, colors)):
        pos = x + (i - 1) * w
        bars = ax.bar(pos, vals, w, label=name, color=col, edgecolor="none")
        for b, v in zip(bars, vals):
            if v == v:  # NaN 아님
                ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.0f}",
                        ha="center", fontsize=8.5, color="#1A1A1A")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=12)
    ax.axhline(100, ls="--", lw=0.7, color="#B4B4B4")
    ax.legend(frameon=False, ncol=3, loc="lower center",
              bbox_to_anchor=(0.5, -0.22), fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", lw=0.5, color="#EBEBEB")
    ax.set_axisbelow(True)
    plt.tight_layout()
    path = os.path.join(here, f"eval_single_results{suffix}.png")
    plt.savefig(path, dpi=160)
    print(f"🖼️ {os.path.basename(path)} 저장")


if __name__ == "__main__":
    asyncio.run(main(_parse_args()))
