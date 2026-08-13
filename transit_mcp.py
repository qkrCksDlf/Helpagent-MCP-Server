"""
ODsay 대중교통 길찾기 MCP 서버 (네이버 지역검색 + ODsay 조합)
"""
import os
import re
import sys
import httpx
from urllib.parse import unquote, quote
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

# 🌟 윈도우(cp949) 콘솔에서 한글/이모지 로그가 깨지거나 터지는 것 방지.
#    stdout 은 MCP 프로토콜 전용이라 건드리지 않고, 로그용 stderr 만 utf-8 로 맞춤.
try:
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

mcp = FastMCP("transit")

# 🌟 인코딩된 키가 들어와도 원본으로 되돌림 (원본 키면 %가 없어 그대로 통과)
#    httpx 가 params 로 넘길 때 한 번 인코딩하므로, 여기서 미리 디코딩해 두면
#    "일반 키 / URL 인코딩 키" 어느 쪽을 .env 에 넣어도 이중 인코딩이 안 생김.
ODSAY_API_KEY = unquote(os.getenv("ODSAY_API_KEY", ""))
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

ODSAY_BASE = "https://api.odsay.com/v1/api"
NAVER_LOCAL_URL = "https://openapi.naver.com/v1/search/local.json"


async def _search_place(name: str) -> dict | None:
    """
    네이버 지역검색으로 장소 → 좌표 변환.
    네이버 지역검색 v1 은 mapx/mapy 를 WGS84(경위도) * 1e7 정수로 반환하므로
    1e7 로 나누면 경도/위도가 된다. (예: 경복궁 mapx '1269770162' → 126.977)
    """
    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        return None

    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
    }

    # 🌟 "~역" 으로 끝나면 지하철역을 원하는 것이므로 검색어에 "지하철역" 을 덧붙여
    #    롯데마트/헤어샵 같은 엉뚱한 상호가 1등으로 잡히는 걸 막는다.
    is_station_query = name.endswith("역")
    query = f"{name} 지하철역" if is_station_query else name

    # 후보를 여러 개 받아서 이름·카테고리로 더 그럴듯한 장소를 고른다.
    params = {"query": query, "display": 5, "sort": "random"}

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(NAVER_LOCAL_URL, headers=headers, params=params)
        if resp.status_code != 200:
            return None
        data = resp.json()

    items = data.get("items", [])
    if not items:
        return None

    def _score(it: dict) -> int:
        title = re.sub(r"<[^>]+>", "", it.get("title", ""))
        cat = it.get("category", "")
        s = 0
        # 검색어가 제목에 그대로 들어가면 가산
        if name in title:
            s += 2
        # 제목이 검색어로 시작하면 더 가산 (예: "부평" → "부평역")
        if title.startswith(name):
            s += 2

        # 🌟 지하철역 검색일 때: 역/교통 카테고리 강하게 우대, 상호류 강하게 감점
        if is_station_query:
            if "지하철" in cat or "전철" in cat or "교통" in cat or "역" in cat:
                s += 6
            # 제목이 "부평역" 처럼 딱 역 이름이면 최우선
            if title == name or title.startswith(name) and "역" in title:
                s += 4
            # 마트·미용·음식 등 상호류는 강하게 감점
            if any(k in cat for k in ("음식", "카페", "쇼핑", "마트", "미용", "헤어", "병원", "매장", "상점", "학원")):
                s -= 6
        else:
            # 일반 장소: 관광·명소·공원 우대, 음식·쇼핑 감점
            if any(k in cat for k in ("관광", "명소", "공원", "유원지", "테마파크")):
                s += 3
            if any(k in cat for k in ("음식", "카페", "쇼핑", "매장", "상점")):
                s -= 2
        return s

    first = max(items, key=_score)

    # 후보들과 최종 선택을 로그로 (발표 전 검증용)
    cand_log = ", ".join(
        f"{re.sub(r'<[^>]+>', '', it.get('title',''))}({it.get('category','')})"
        for it in items
    )
    print(f"[지역검색] '{name}' 후보: {cand_log}", file=sys.stderr)
    # 네이버 지역검색: mapx, mapy = 경도/위도 * 1e7
    mapx = first.get("mapx", "0")
    mapy = first.get("mapy", "0")
    lng = float(mapx) / 1e7
    lat = float(mapy) / 1e7

    # HTML 태그 제거
    title = re.sub(r"<[^>]+>", "", first.get("title", name))
    print(f"[지역검색] '{name}' → '{title}' (category={first.get('category','')})", file=sys.stderr)

    return {
        "name": title,
        "x": lng,
        "y": lat,
    }


async def _resolve_location(value: str) -> dict | None:
    """
    출발/도착 위치를 좌표로 변환.
    - "@현재위치@위도,경도" 형식이면 GPS 좌표를 바로 사용 (네이버 검색 생략)
    - 그 외에는 장소명으로 네이버 지역검색
    """
    if value.startswith("@현재위치@"):
        try:
            coords = value.split("@현재위치@", 1)[1]
            lat_str, lng_str = coords.split(",")
            lat = float(lat_str.strip())
            lng = float(lng_str.strip())
            print(f"[GPS] 현재 위치 좌표 사용: lat={lat}, lng={lng}", file=sys.stderr)
            return {"name": "현재 위치", "x": lng, "y": lat}
        except Exception as e:
            print(f"[GPS] 좌표 파싱 실패: {e}", file=sys.stderr)
            return None
    return await _search_place(value)


@mcp.tool()
async def find_transit_route(start: str, end: str) -> dict:
    """
    대중교통(지하철/버스) 길찾기.
    출발지에서 도착지까지의 경로, 환승 정보, 소요 시간을 알려줍니다.

    Args:
        start: 출발지 (예: "강남역", "인하대학교"). 현재 위치는 "@현재위치@위도,경도" 형식 가능.
        end: 도착지 (예: "홍대입구역", "롯데월드")
    """
    if not ODSAY_API_KEY:
        return {"error": "ODsay API 키가 설정되어 있지 않습니다."}

    start_poi = await _resolve_location(start)
    if not start_poi:
        return {"error": f"'{start}' 위치를 찾을 수 없습니다."}

    end_poi = await _resolve_location(end)
    if not end_poi:
        return {"error": f"'{end}' 위치를 찾을 수 없습니다."}

    # 🌟 ODsay 키는 직접 인코딩해서 URL 에 붙인다.
    #    httpx params 에 맡기면 키 안의 '+', '/', '=' 같은 문자를 ODsay 가 기대하는 형태로
    #    안 보내주는 경우가 있어(특히 '+' 가 공백처럼 처리됨) ApiKeyAuthFailed 가 난다.
    encoded_key = quote(ODSAY_API_KEY, safe="")
    url = (
        f"{ODSAY_BASE}/searchPubTransPathT"
        f"?apiKey={encoded_key}"
        f"&SX={start_poi['x']}&SY={start_poi['y']}"
        f"&EX={end_poi['x']}&EY={end_poi['y']}"
        f"&OPT=0"
    )
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(url)
        if resp.status_code != 200:
            return {"error": f"ODsay API 오류: {resp.status_code}"}
        data = resp.json()

    # 🌟 원인 확인용: ODsay 원본 응답 로그 (stdout 은 MCP 프로토콜용이라 stderr 로 출력)
    print(f"[ODsay] 원본 응답: {str(data)[:300]}", file=sys.stderr)

    # ODsay 에러 체크 — 에러는 리스트로 오고 키는 'message' (msg 아님)
    if "error" in data:
        err = data["error"]
        if isinstance(err, list) and err:
            code = err[0].get("code", "")
            msg = err[0].get("message", "알 수 없음")
            return {"error": f"길찾기 실패: [{code}] {msg}"}
        return {"error": f"길찾기 실패: {err}"}

    result = data.get("result", {})
    paths = result.get("path", [])
    if not paths:
        return {"error": "경로를 찾을 수 없습니다."}

    routes = []
    for path in paths[:3]:
        info = path.get("info", {})
        sub_paths = path.get("subPath", [])

        steps = []
        for sp in sub_paths:
            traffic_type = sp.get("trafficType")
            if traffic_type == 1:
                # 지하철
                lane = sp.get("lane", [{}])[0]
                steps.append({
                    "mode": "subway",
                    "line": lane.get("name", "지하철"),
                    "from": sp.get("startName", ""),
                    "to": sp.get("endName", ""),
                    "stations": sp.get("stationCount", 0),
                    "time_min": sp.get("sectionTime"),
                })
            elif traffic_type == 2:
                # 버스
                lane = sp.get("lane", [{}])[0]
                steps.append({
                    "mode": "bus",
                    "line": f"{lane.get('busNo', '버스')}번",
                    "from": sp.get("startName", ""),
                    "to": sp.get("endName", ""),
                    "stations": sp.get("stationCount", 0),
                    "time_min": sp.get("sectionTime"),
                })
            elif traffic_type == 3:
                # 도보 (0분짜리 자투리 구간은 노이즈라 제외)
                walk_time = sp.get("sectionTime", 0)
                if walk_time and walk_time > 0:
                    steps.append({
                        "mode": "walk",
                        "time_min": walk_time,
                    })

        routes.append({
            "total_time_min": info.get("totalTime"),
            "transfer_count": info.get("subwayTransitCount", 0) + info.get("busTransitCount", 0),
            "walking_time_min": info.get("totalWalk"),
            "fare": info.get("payment"),
            "steps": steps,
        })

    return {
        "from": start_poi["name"],
        "to": end_poi["name"],
        "routes": routes,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
