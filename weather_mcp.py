"""
날씨 MCP 서버
- 현재 날씨 + 시간별 예보 + 일별 예보
"""
import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather")

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

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


async def _geocode(city: str):
    """도시명 → 좌표"""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            GEOCODE_URL,
            params={"name": city, "count": 1, "language": "ko"},
        )
        data = resp.json()
        if not data.get("results"):
            return None
        loc = data["results"][0]
        return {
            "lat": loc["latitude"],
            "lon": loc["longitude"],
            "name": loc.get("name", city),
        }


@mcp.tool()
async def get_current_weather(city: str) -> dict:
    """
    한국 도시의 *현재* 날씨를 가져옵니다.
    "지금 날씨", "오늘 날씨 어때" 같은 질문에 사용하세요.

    Args:
        city: 도시명 (예: "서울", "부산", "Seoul")
    """
    loc = await _geocode(city)
    if not loc:
        return {"error": f"'{city}' 도시를 찾을 수 없습니다."}

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            WEATHER_URL,
            params={
                "latitude": loc["lat"],
                "longitude": loc["lon"],
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m",
                "timezone": "Asia/Seoul",
            },
        )
        current = resp.json().get("current", {})

    code = current.get("weather_code", -1)
    return {
        "city": loc["name"],
        "temperature": current.get("temperature_2m"),
        "apparent_temperature": current.get("apparent_temperature"),
        "humidity": current.get("relative_humidity_2m"),
        "wind_speed": current.get("wind_speed_10m"),
        "condition": WEATHER_CODES.get(code, "알 수 없음"),
    }


@mcp.tool()
async def get_daily_forecast(city: str, days: int = 7) -> dict:
    """
    한국 도시의 *일별 예보*를 가져옵니다 (최대 16일).
    "내일 날씨", "이번 주 날씨", "주말 날씨", "다음주 날씨" 같은 질문에 사용하세요.

    Args:
        city: 도시명 (예: "서울", "부산")
        days: 가져올 일수 (기본 7일, 최대 16)
    """
    days = min(max(days, 1), 16)
    loc = await _geocode(city)
    if not loc:
        return {"error": f"'{city}' 도시를 찾을 수 없습니다."}

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            WEATHER_URL,
            params={
                "latitude": loc["lat"],
                "longitude": loc["lon"],
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "forecast_days": days,
                "timezone": "Asia/Seoul",
            },
        )
        daily = resp.json().get("daily", {})

    dates = daily.get("time", [])
    codes = daily.get("weather_code", [])
    t_max = daily.get("temperature_2m_max", [])
    t_min = daily.get("temperature_2m_min", [])
    rain = daily.get("precipitation_probability_max", [])

    forecast = []
    for i in range(len(dates)):
        forecast.append({
            "date": dates[i],
            "condition": WEATHER_CODES.get(codes[i], "알 수 없음"),
            "temp_max": t_max[i],
            "temp_min": t_min[i],
            "rain_probability": rain[i],
        })

    return {
        "city": loc["name"],
        "forecast": forecast,
    }


@mcp.tool()
async def get_hourly_forecast(city: str, hours: int = 12) -> dict:
    """
    한국 도시의 *시간별 예보*를 가져옵니다 (최대 48시간).
    "오후 날씨", "저녁에 비 와?", "몇 시쯤 비 그쳐?" 같은 질문에 사용하세요.

    Args:
        city: 도시명 (예: "서울", "부산")
        hours: 가져올 시간 수 (기본 12시간, 최대 48)
    """
    hours = min(max(hours, 1), 48)
    loc = await _geocode(city)
    if not loc:
        return {"error": f"'{city}' 도시를 찾을 수 없습니다."}

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            WEATHER_URL,
            params={
                "latitude": loc["lat"],
                "longitude": loc["lon"],
                "hourly": "temperature_2m,weather_code,precipitation_probability",
                "forecast_hours": hours,
                "timezone": "Asia/Seoul",
            },
        )
        hourly = resp.json().get("hourly", {})

    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])
    codes = hourly.get("weather_code", [])
    rain = hourly.get("precipitation_probability", [])

    forecast = []
    for i in range(len(times)):
        forecast.append({
            "time": times[i],
            "temperature": temps[i],
            "condition": WEATHER_CODES.get(codes[i], "알 수 없음"),
            "rain_probability": rain[i],
        })

    return {
        "city": loc["name"],
        "forecast": forecast,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
