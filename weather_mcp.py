"""
날씨 MCP 서버
- Open-Meteo API 사용 (무료, 키 불필요)
- Geocoding API로 도시명 → 좌표 변환
"""
import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather")

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

# Open-Meteo WMO weather code 일부 매핑
WEATHER_CODES = {
    0: "맑음 ☀️",
    1: "대체로 맑음 🌤️",
    2: "부분적으로 흐림 ⛅",
    3: "흐림 ☁️",
    45: "안개 🌫️",
    48: "짙은 안개 🌫️",
    51: "약한 이슬비 🌦️",
    53: "이슬비 🌦️",
    55: "강한 이슬비 🌧️",
    61: "약한 비 🌧️",
    63: "비 🌧️",
    65: "강한 비 🌧️",
    71: "약한 눈 🌨️",
    73: "눈 ❄️",
    75: "많은 눈 ❄️",
    80: "소나기 🌦️",
    81: "강한 소나기 🌧️",
    82: "매우 강한 소나기 ⛈️",
    95: "천둥번개 ⛈️",
}


@mcp.tool()
async def get_weather(city: str) -> dict:
    """
    한국 도시의 현재 날씨를 가져옵니다.
    
    Args:
        city: 도시명 (한글 또는 영문). 예: "서울", "Seoul", "부산"
    
    Returns:
        현재 기온, 체감 온도, 날씨 상태, 풍속, 습도
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        # 1. 도시명 → 좌표
        geo_resp = await client.get(
            GEOCODE_URL,
            params={"name": city, "count": 1, "language": "ko"}
        )
        geo_data = geo_resp.json()
        if not geo_data.get("results"):
            return {"error": f"'{city}' 도시를 찾을 수 없습니다."}

        loc = geo_data["results"][0]
        lat, lon = loc["latitude"], loc["longitude"]
        resolved_name = loc.get("name", city)

        # 2. 날씨 조회
        weather_resp = await client.get(
            WEATHER_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m",
                "timezone": "Asia/Seoul",
            },
        )
        weather_data = weather_resp.json()
        current = weather_data.get("current", {})

        code = current.get("weather_code", -1)
        condition = WEATHER_CODES.get(code, "알 수 없음")

        return {
            "city": resolved_name,
            "temperature": current.get("temperature_2m"),
            "apparent_temperature": current.get("apparent_temperature"),
            "humidity": current.get("relative_humidity_2m"),
            "wind_speed": current.get("wind_speed_10m"),
            "condition": condition,
            "weather_code": code,
        }


if __name__ == "__main__":
    mcp.run(transport="stdio")