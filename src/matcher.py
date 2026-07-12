# 1~2단계: 문장 → 약 이름 추출 (사전 스캔 + 자모 유사도 오타 보정)
import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz, process

from dictionary import CATEGORIES, DrugDictionary, jamo

CONFIRM_THRESHOLD = 85.0
SUGGEST_THRESHOLD = 75.0


@dataclass
class Match:
    surface: str            # 문장에 나타난 표기
    status: str             # confirmed | category | suggest | unknown
    name: str | None = None      # 확정/제안된 사전 이름
    candidates: list = field(default_factory=list)  # category·suggest일 때 후보
    score: float = 100.0


class DrugMatcher:
    def __init__(self, dictionary: DrugDictionary | None = None, ner=None):
        self.dic = dictionary or DrugDictionary()
        self.ner = ner  # nlu.DrugNER — 없으면 사전+유사도만으로 동작

    def extract(self, text: str) -> list[Match]:
        matches: list[Match] = []
        covered: list[tuple[int, int]] = []

        # 1단계: 사전 최장일치 (가장 신뢰도 높음)
        for start, end, name in self.dic.scan(text):
            if self._is_stopword(name):  # 일반 단어와 같은 제품명 오탐 방지 (예: '추천')
                continue
            covered.append((start, end))
            if name in CATEGORIES:
                matches.append(Match(surface=name, status="category", name=name,
                                     candidates=CATEGORIES[name]))
            else:
                matches.append(Match(surface=name, status="confirmed", name=name))

        # 2단계: 직접 학습한 NER 모델 — 사전이 못 잡은 구간 보완
        if self.ner is not None and self.ner.available:
            for start, end, surface in self.ner.spans(text):
                # 서브워드 경계에서 잘린 스팬을 어절 경계까지 확장
                start, end, surface = self._expand_to_word(text, start, end)
                if any(s < end and start < e for s, e in covered):
                    continue
                token = self.dic.strip_josa(surface)
                if len(token) < 2 or self._is_stopword(token):
                    continue
                covered.append((start, end))
                if token in self.dic.name_to_seqs:
                    matches.append(Match(surface=surface, status="confirmed", name=token))
                elif token in CATEGORIES:
                    matches.append(Match(surface=surface, status="category", name=token,
                                         candidates=CATEGORIES[token]))
                else:
                    fuzzy = self._fuzzy(token)
                    if fuzzy is not None:
                        matches.append(fuzzy)
                    else:
                        matches.append(Match(surface=surface, status="unknown", score=0.0))

        # 3단계: 남은 한글 어절 → 자모 유사도 보정
        for m in re.finditer(r"[가-힣a-zA-Z0-9]+", text):
            if any(s < m.end() and m.start() < e for s, e in covered):
                continue
            token = self.dic.strip_josa(m.group())
            if len(token) < 2 or self._is_stopword(token):
                continue
            fuzzy = self._fuzzy(token)
            if fuzzy is not None:
                matches.append(fuzzy)

        matches.sort(key=lambda x: text.find(x.surface))
        return matches

    def _fuzzy(self, token: str) -> Match | None:
        # 접두 일치 우선: "판콜" → 판콜에스/판콜아이 (되묻기 후보)
        prefixed = sorted({n for n in self.dic.name_to_seqs
                           if n.startswith(token) and len(n) <= len(token) + 6})
        if len(prefixed) == 1:
            return Match(surface=token, status="confirmed", name=prefixed[0], score=99.0)
        if prefixed:
            return Match(surface=token, status="suggest", name=prefixed[0],
                         candidates=prefixed[:5], score=99.0)

        # 2글자 토큰은 오탐이 잦아 임계 상향 (임신→이뮤신주 80점 차단, 탁쎈→탁센 83.3점 허용)
        cutoff = 82.0 if len(token) <= 2 else SUGGEST_THRESHOLD
        tj = jamo(token)
        found = process.extract(tj, self.dic.jamo_index, scorer=fuzz.ratio,
                                limit=5, score_cutoff=cutoff)
        # 영문·숫자는 정확히 일치해야 함 (비타민C ≠ 비타민E, 이지엔6 ≠ 이지엔8)
        token_alnum = sorted(re.findall(r"[A-Za-z0-9]", token))
        found = [(j, s, n) for j, s, n in found
                 if sorted(re.findall(r"[A-Za-z0-9]", n)) == token_alnum]
        if not found:
            return None
        best_score = found[0][1]
        # 동점권(2점 이내)에서는 첫 글자가 같은 후보 우선 (지르택 → 쥬르택이 아닌 지르텍)
        top = [f for f in found if f[1] >= best_score - 2]
        top.sort(key=lambda f: (f[2][0] != token[0], -f[1]))
        _, score, name = top[0]
        # 첫 글자가 다르면 자동 확정하지 않고 되묻기 (루테인→로테인 오확정 방지)
        if score >= CONFIRM_THRESHOLD and name[0] == token[0]:
            return Match(surface=token, status="confirmed", name=name, score=score)
        return Match(surface=token, status="suggest", name=name,
                     candidates=[name], score=score)

    @staticmethod
    def _expand_to_word(text: str, start: int, end: int) -> tuple[int, int, str]:
        for m in re.finditer(r"[가-힣a-zA-Z0-9]+", text):
            if m.start() <= start < m.end():
                end = max(end, m.end())
                start = m.start()
                break
        return start, end, text[start:end]

    _VERB_ENDINGS = ("는데", "은데", "니까", "면서", "고서", "어서", "아서",
                     "세요", "어요", "아요", "데요", "습니다", "ㅂ니다", "는지", "먹고", "했어")

    @classmethod
    def _is_stopword(cls, token: str) -> bool:
        stop = {"같이", "함께", "동시에", "먹어도", "먹으면", "복용", "복용해도", "돼",
                "되나", "괜찮아", "괜찮나", "괜찮을까", "있어", "알려줘", "그리고", "지금",
                "임신", "임신중", "임산부", "수유", "수유중", "아이", "애기", "어린이",
                "노인", "어르신", "엄마", "아빠", "할머니", "할아버지", "아침", "점심",
                "저녁", "하루", "부작용", "추천", "영양제", "성분"}
        if token in stop:
            return True
        # 동사/서술 어절은 약 이름 후보에서 제외 (예: "먹는데" → 유사도 오탐 방지)
        return any(token.endswith(e) for e in cls._VERB_ENDINGS)
