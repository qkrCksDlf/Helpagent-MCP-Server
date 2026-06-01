"""
네이버 검색 MCP 서버
- 블로그, 뉴스, 웹 검색 지원
"""
import os
import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("search")

NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")


def _clean_html(text: str) -> str:
    """네이버 결과의 <b>, </b> 같은 HTML 태그 제거"""
    import re
    text = re.sub(r'<[^>]+>', '', text)
    text = text.replace('&quot;', '"').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    return text


async def _naver_search(category: str, query: str, display: int = 5) -> dict:
    """네이버 검색 API 공통 함수"""
    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        return {"error": "네이버 API 키가 설정되어 있지 않습니다."}

    url = f"https://openapi.naver.com/v1/search/{category}.json"
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
    }
    params = {"query": query, "display": display, "sort": "sim"}

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(url, headers=headers, params=params)
        if resp.status_code != 200:
            return {"error": f"네이버 API 오류: {resp.status_code}"}
        data = resp.json()

    items = []
    for item in data.get("items", []):
        items.append({
            "title": _clean_html(item.get("title", "")),
            "description": _clean_html(item.get("description", "")),
            "link": item.get("link", ""),
        })

    return {"query": query, "results": items}


@mcp.tool()
async def search_web(query: str) -> dict:
    """
    네이버에서 일반 웹문서를 검색합니다.
    일반적인 정보, 정의, 지식 검색에 사용하세요.

    Args:
        query: 검색어 (예: "코로나 백신 정보", "전세 보증보험")
    """
    return await _naver_search("webkr", query)


@mcp.tool()
async def search_news(query: str) -> dict:
    """
    네이버 뉴스를 검색합니다.
    최신 사건, 뉴스, 시사 관련 질문에 사용하세요.

    Args:
        query: 검색어 (예: "오늘 뉴스", "대선 결과")
    """
    return await _naver_search("news", query)


@mcp.tool()
async def search_blog(query: str) -> dict:
    """
    네이버 블로그를 검색합니다.
    리뷰, 후기, 맛집, 여행, 생활 정보에 사용하세요.

    Args:
        query: 검색어 (예: "강남역 맛집", "제주도 여행 후기")
    """
    return await _naver_search("blog", query)


@mcp.tool()
async def search_shop(query: str) -> dict:
    """
    네이버 쇼핑에서 상품을 검색합니다.
    제품 가격 비교, 쇼핑 정보에 사용하세요.

    Args:
        query: 검색어 (예: "노트북 추천", "겨울 패딩")
    """
    return await _naver_search("shop", query)


if __name__ == "__main__":
    mcp.run(transport="stdio")