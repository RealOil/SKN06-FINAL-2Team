from django.db import connection
from .models import RecommendedWork
import re


def extract_titles(response_text):
    """
    AI 응답에서 추천된 작품 제목을 추출하는 함수.
    - response_text: AI 챗봇의 응답 (Markdown 형식 포함 가능)
    - return: 추천된 작품 제목 리스트
    """
    # 정규 표현식을 사용하여 "1. 제목" 형식의 작품명 추출
    pattern = r"\d+\.\s*\*\*([^*]+)\*\*"  # "1. **제목**" 형식의 작품명 추출
    matches = re.findall(pattern, response_text)

    # 제목을 정리하여 반환 (앞뒤 공백 제거)
    titles = [title.strip() for title in matches]
    return titles


def save_recommended_works(user, recommended_titles, model_name):
    """
    추천된 작품을 추천 기록 테이블에 저장하는 함수
    :param user: 추천을 받은 사용자
    :param recommended_titles: 추천된 작품 제목 리스트
    :param model_name: 추천 모델명 (romance, fantasy 등)
    """
    if not recommended_titles:
        return

    with connection.cursor() as cursor:
        # 추천된 작품 ID 가져오기
        cursor.execute(
            """
            SELECT id FROM preset_preference_contents WHERE title IN %s
        """,
            [tuple(recommended_titles)],
        )

        results = cursor.fetchall()

        # 추천 데이터 저장
        for content_id in results:
            RecommendedWork.objects.create(
                user=user, content_id=content_id[0], recommended_model=model_name
            )
