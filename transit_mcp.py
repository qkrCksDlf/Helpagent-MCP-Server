"""
ODsay 대중교통 길찾기 MCP 서버 (네이버 지역검색 + ODsay 조합)
"""
import os
import re
import sys
import httpx
from urllib.parse import unquote
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
    params = {"query": name, "display": 1, "sort": "random"}

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(NAVER_LOCAL_URL, headers=headers, params=params)
        if resp.status_code != 200:
            return None
        data = resp.json()

    items = data.get("items", [])
    if not items:
        return None

    first = items[0]
    # 네이버 지역검색: mapx, mapy = 경도/위도 * 1e7
    mapx = first.get("mapx", "0")
    mapy = first.get("mapy", "0")
    lng = float(mapx) / 1e7
    lat = float(mapy) / 1e7

    # HTML 태그 제거
    title = re.sub(r"<[^>]+>", "", first.get("title", name))

    return {
        "name": title,
        "x": lng,
        "y": lat,
    }


@mcp.tool()
async def find_transit_route(start: str, end: str) -> dict:
    """
    대중교통(지하철/버스) 길찾기.
    출발지에서 도착지까지의 경로, 환승 정보, 소요 시간을 알려줍니다.

    Args:
        start: 출발지 (예: "강남역", "인하대학교", "서울대학교")
        end: 도착지 (예: "홍대입구역", "인천공항", "인하대병원")
    """
    if not ODSAY_API_KEY:
        return {"error": "ODsay API 키가 설정되어 있지 않습니다."}

    start_poi = await _search_place(start)
    if not start_poi:
        return {"error": f"'{start}' 위치를 찾을 수 없습니다."}

    end_poi = await _search_place(end)
    if not end_poi:
        return {"error": f"'{end}' 위치를 찾을 수 없습니다."}

    # 🌟 ODsay 키는 직접 인코딩해서 URL 에 붙인다.
    #    httpx params 에 맡기면 키 안의 '+', '/', '=' 같은 문자를 ODsay 가 기대하는 형태로
    #    안 보내주는 경우가 있어(특히 '+' 가 공백처럼 처리됨) ApiKeyAuthFailed 가 난다.
    params = {
    "apiKey": ODSAY_API_KEY,
    "SX": start_poi["x"],
    "SY": start_poi["y"],
    "EX": end_poi["x"],
    "EY": end_poi["y"],
    "OPT": 0,
}

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
        f"{ODSAY_BASE}/searchPubTransPathT",
        params=params
    )
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
                lane = sp.get("lane", [{}])[0]
                line_name = lane.get("name", "지하철")
                start_st = sp.get("startName", "")
                end_st = sp.get("endName", "")
                stations = sp.get("stationCount", 0)
                steps.append(f"{line_name} {start_st}→{end_st} ({stations}정거장)")
            elif traffic_type == 2:
                lane = sp.get("lane", [{}])[0]
                bus_no = lane.get("busNo", "버스")
                start_st = sp.get("startName", "")
                end_st = sp.get("endName", "")
                stations = sp.get("stationCount", 0)
                steps.append(f"{bus_no}번 버스 {start_st}→{end_st} ({stations}정거장)")

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