from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo, StructuredQuery
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.tools import Tool

# 벡터스토어 설정
vector_store = Chroma(
    persist_directory="data/Raw_DB/vector_store",
    collection_name="romance",
    embedding_function=HuggingFaceEmbeddings(model_name="BAAI/bge-m3"),
)

metadata_field_info = [
    AttributeInfo(name="title", description="작품의 제목", type="string"),
    AttributeInfo(
        name="type", description="작품의 타입 (웹툰 또는 웹소설)", type="string"
    ),
    AttributeInfo(
        name="platform",
        description="작품의 연재처 (카카오페이지, 네이버 웹툰, 카카오웹툰, 네이버 시리즈)",
        type="string",
    ),
    AttributeInfo(
        name="genre",
        description="작품의 장르 (로맨스, BL, 로판, 판타지, 현판 등)",
        type="string",
    ),
    AttributeInfo(
        name="status",
        description="작품의 연재 상태 (연재, 완결, 휴재)",
        type="string",
    ),
    AttributeInfo(
        name="update_days",
        description="작품의 연재 요일 (월, 화, 수 등 또는 해당 없음)",
        type="string",
    ),
    AttributeInfo(
        name="age_rating",
        description="작품의 연령 제한 (전체 이용가, 12세 이용가 등)",
        type="string",
    ),
    AttributeInfo(
        name="url",
        description="작품의 링크",
        type="string",
    ),
    AttributeInfo(
        name="thumbnail",
        description="작품의 썸네일",
        type="string",
    ),
    AttributeInfo(
        name="score",
        description="작품의 인기도 점수)",
        type="float",
    ),
    AttributeInfo(
        name="author",
        description="작품의 작가",
        type="string",
    ),
    AttributeInfo(
        name="original",
        description="작품의 원작 제목",
        type="string",
    ),
    AttributeInfo(
        name="episode",
        description="작품의 총 에피소드",
        type="int",
    ),
]


def get_genre_selfquery_tool(genre, tool_name):
    """
    특정 장르에 대한 SelfQueryRetriever 기반의 Tool을 생성
    :param vector_store: Chroma VectorStore
    :param genre: 필터링할 장르 (예: '로맨스', '판타지', 'BL')
    :param tool_name: Tool의 이름
    :return: LangChain Tool
    """
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    retriever = SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vector_store,
        document_contents="웹소설 및 웹툰 데이터",
        metadata_field_info=metadata_field_info,
        search_kwargs={"k": 10},  # 상위 5개 검색
    )

    def search_and_format(query):
        """
        검색 수행 후 메타데이터를 포함한 결과 반환
        """
        results = retriever.invoke(query)
        return format_metadata_results(results)

    return Tool(
        name=f"{tool_name}_retriever_tool",
        func=search_and_format,
        description=f"Use this tool to search {tool_name} genre webtoons and webnovels. Add 'genre: {genre}' to your search query.",
    )


def format_metadata_results(results):
    """
    검색된 결과의 메타데이터를 LLM이 이해할 수 있도록 정리하는 함수
    :param results: 검색 결과 리스트
    :return: LLM 프롬프트에 들어갈 문자열
    """
    if not results:
        return "추천할 작품이 없습니다."

    output_text = "🔍 **추천 가능한 작품 정보:**\n\n"

    for idx, result in enumerate(results, start=1):
        metadata = result.metadata  # ✅ 메타데이터 추출
        output_text += f"**{idx}. {metadata.get('title', '제목 없음')}**\n"
        output_text += f"- **작가:** {metadata.get('author', '알 수 없음')}\n"
        output_text += f"- **타입:** {metadata.get('type', '알 수 없음')}\n"
        output_text += f"- **플랫폼:** {metadata.get('platform', '알 수 없음')}\n"
        output_text += f"- **장르:** {metadata.get('genre', '알 수 없음')}\n"
        output_text += f"- **연재 상태:** {metadata.get('status', '알 수 없음')}\n"
        output_text += f"- **연재 요일:** {metadata.get('update_days', '알 수 없음')}\n"
        output_text += f"- **원작 제목:** {metadata.get('original', '알 수 없음')}\n"
        output_text += f"- **총 에피소드:** {metadata.get('episode', 'N/A')}\n"
        output_text += f"- **연령 제한:** {metadata.get('age_rating', '알 수 없음')}\n"
        output_text += f"- **평점:** {metadata.get('score', 'N/A')}\n"

        thumbnail = metadata.get("thumbnail", "")
        if thumbnail:
            output_text += f"![썸네일]({thumbnail})\n"

        url = metadata.get("url", "#")
        output_text += f"[🔗 작품 보러가기]({url})\n\n"

    return output_text
