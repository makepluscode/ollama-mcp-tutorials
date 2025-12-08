"""
============================================================================
MCP Client: Test & Inspector
============================================================================

이 스크립트는 mcp_server.py를 하위 프로세스로 실행하고,
MCP 프로토콜(Stdio)을 통해 통신하는 클라이언트 예제입니다.

서버가 정상적으로 도구(Tool)를 노출하는지 확인하고,
실제 검색(search_knowledge_base)을 수행하여 결과를 출력합니다.

실행 방법:
uv run python mcp_client.py "검색어"
예: uv run python mcp_client.py "QoS"
"""

import argparse
import asyncio
import logging
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("mcp_client")


async def run_client(query: str):
    logger.info("🔌 MCP 서버 연결 중...")

    # 서버 실행 파라미터 설정
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "mcp_server.py"],
        env=None,  # 필요한 경우 환경 변수 추가
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # 1. 초기화
                await session.initialize()
                logger.info("✅ 세션 초기화 완료")

                # 2. 도구 목록 조회
                tools = await session.list_tools()
                logger.info(f"🛠️ 발견된 도구: {[tool.name for tool in tools.tools]}")

                # 3. 도구 실행 (검색)
                logger.info(f"🔍 검색 실행: '{query}'")
                result = await session.call_tool(
                    name="search_knowledge_base", arguments={"query": query}
                )

                # 4. 결과 출력
                print("\n" + "=" * 60)
                if result.isError:
                    print(f"❌ 오류 발생: {result.content}")
                else:
                    # TextContent 객체 리스트 처리
                    for content in result.content:
                        if content.type == "text":
                            print(content.text)
                print("=" * 60 + "\n")

    except Exception as e:
        logger.error(f"❌ 클라이언트 실행 중 오류: {e}")


def main():
    parser = argparse.ArgumentParser(description="MCP RAG 검색 클라이언트 테스트")
    parser.add_argument("query", type=str, help="검색할 질문이나 키워드")
    args = parser.parse_args()

    # 비동기 실행
    asyncio.run(run_client(args.query))


if __name__ == "__main__":
    main()
