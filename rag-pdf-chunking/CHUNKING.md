# PDF 문서 청킹(Chunking) 기법 비교 분석

RAG(Retrieval-Augmented Generation) 파이프라인에서 문서 청킹은 검색 성능과 LLM의 답변 품질에 직접적인 영향을 미치는 매우 중요한 전처리 단계입니다. 이 문서는 LangChain에서 제공하는 네 가지 주요 청킹 전략을 동일한 PDF 문서(`DDSI-RTPS-v2.5.pdf`)에 적용하고, 그 결과를 분석 및 비교합니다.

## 청킹 전략 상세 분석

각 전략은 `main.py`의 `ChunkerFactory` 클래스에서 구현되었으며, 결과는 각각의 `chunks-[전략명].txt` 파일에서 확인할 수 있습니다.

### 1. RecursiveCharacterTextSplitter (`recursive`)

#### 설명
가장 보편적으로 추천되는 전략입니다. 지정된 `chunk_size`를 최대한 맞추기 위해, 의미적으로 중요하다고 여겨지는 구분자(separator) 리스트(`["\n\n", "\n", " ", ""]`)를 재귀적으로(Recursive) 사용하여 텍스트를 분할합니다. 문단, 줄바꿈, 공백 순으로 분할을 시도하여 문맥이 최대한 유지되도록 노력합니다.

#### LangChain API
```python
# From ChunkerFactory in main.py
return RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)
```

#### 결과 분석 (`chunks-recursive.txt`)
- **총 청크 수**: 41개
- **평균 청크 크기**: 581.17자
- **분할 소요 시간**: 0.0014초
- **분석**: 설정된 청크 크기(700자)에 비교적 근접하면서도, 문단과 문장을 존중하려는 시도를 보입니다. 41개의 비교적 잘 분배된 청크를 생성했으며, 실행 속도가 매우 빠릅니다. 일반적인 텍스트 문서에 가장 먼저 시도해 볼 만한 균형 잡힌 전략입니다.

### 2. CharacterTextSplitter (`character`)

#### 설명
가장 단순한 전략입니다. 사용자가 지정한 단일 구분자(`\n\n` - 문단)를 기준으로 텍스트를 분할합니다. 문단이 `chunk_size`보다 크면 해당 문단을 기준으로만 나누므로, 청크의 크기가 매우 불균일할 수 있습니다.

#### LangChain API
```python
# From ChunkerFactory in main.py
return CharacterTextSplitter(
    separator="\n\n",
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    is_separator_regex=False,
)
```

#### 결과 분석 (`chunks-character.txt`)
- **총 청크 수**: 20개
- **평균 청크 크기**: 1162.15자
- **분할 소요 시간**: 0.0002초
- **분석**: 네 가지 전략 중 가장 적은 수의 청크(20개)를 생성했으며, 평균 크기가 설정값(700자)을 훨씬 초과합니다. 이는 문단(`\n\n`) 단위로만 분할하기 때문입니다. 긴 문단이 많은 문서의 경우, 문맥은 가장 잘 보존되지만 청크가 너무 커서 RAG의 검색 효율성을 떨어뜨릴 수 있습니다. 실행 속도는 가장 빠릅니다.

### 3. TokenTextSplitter (`token`)

#### 설명
LLM(언어 모델)의 관점에서 텍스트를 분할합니다. `tiktoken` 라이브러리를 사용하여 글자 수가 아닌 토큰 수를 기준으로 `chunk_size`를 맞춥니다. 이는 LLM의 컨텍스트 윈도우(Context Window)를 보다 정확하게 관리하고 비용을 예측하는 데 유리합니다.

#### LangChain API
```python
# From ChunkerFactory in main.py
encoding = tiktoken.get_encoding("cl100k_base")
return TokenTextSplitter(
    encoding_name=encoding.name,
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
)
```

#### 결과 분석 (`chunks-token.txt`)
- **총 청크 수**: 36개
- **평균 청크 크기**: 698.92자
- **분할 소요 시간**: 1.4428초
- **분석**: 생성된 청크의 평균 크기가 설정값(700자)에 가장 근접합니다. 이는 토큰 수를 기준으로 비교적 정확하게 분할했음을 의미합니다. 하지만 토큰 수를 계산하는 과정에서 다른 문자 기반 분할보다 훨씬 느린 실행 속도를 보입니다. 모델의 토큰 제한에 맞춰 정밀하게 청킹해야 할 때 유용합니다.

### 4. SemanticChunker (`semantic`)

#### 설명
가장 진보된 전략입니다. 임베딩 모델(`bge-m3`)을 사용하여 문장 간의 의미적 유사도를 계산합니다. 문장들의 관계를 그래프로 구성하고, 의미적으로 큰 단절이 발생하는 지점(Breakpoint)을 찾아 분할합니다. 이로 인해 생성된 청크들은 문맥적 완결성이 매우 높습니다. `chunk_size`를 직접 사용하지 않고, 임베딩 벡터 간의 거리 임계값(기본값: 95번째 백분위수)을 기준으로 분할합니다.

#### LangChain API
```python
# From ChunkerFactory in main.py
return SemanticChunker(embeddings=embeddings)
```

#### 결과 분석 (`chunks-semantic.txt`)
- **총 청크 수**: 42개
- **평균 청크 크기**: 541.17자
- **분할 소요 시간**: 10.7684초
- **분석**: 청크의 평균 크기는 가장 작지만, 각 청크가 의미적으로 완결된 단위일 가능성이 높습니다. 예를 들어, `chunks-semantic.txt`의 5번째 청크는 단 129자로, 규범적 참조 목록 중 일부만 포함하지만 의미적으로는 하나의 단위를 이룹니다. RAG 검색 시 더 정확하고 관련성 높은 정보를 찾는 데 유리할 수 있습니다. 하지만 임베딩 계산으로 인해 분할 소요 시간이 네 가지 전략 중 압도적으로 가장 깁니다.

--- 

## 종합 비교 분석

| 항목 | Recursive | Character | Token | Semantic |
|---|---|---|---|---|
| **설명** | 계층적 구분자 사용 (일반적) | 단일 구분자 사용 (단순) | 모델 토큰 기준 분할 | 문맥 의미 기반 분할 (고급) |
| **총 청크 수** | 41개 | 20개 | 36개 | 42개 |
| **평균 청크 크기** | 581자 | 1162자 | 699자 | 541자 |
| **분할 소요 시간** | **~0.001초** (매우 빠름) | **~0.0002초** (가장 빠름) | ~1.44초 (느림) | **~10.77초** (매우 느림) |
| **품질 유지** | 좋음 | 문맥은 유지되나, 크기가 너무 큼 | 좋음 (토큰 기준) | **매우 좋음** (의미 중심) |
| **주요 장점** | 속도와 품질의 균형 | 가장 빠르고 단순함 | 모델 컨텍스트에 정확함 | 의미적 일관성이 가장 높음 |
| **주요 단점** | 의미적 단절 가능성 있음 | 청크 크기가 매우 불균일함 | 분할 속도가 느림 | **속도가 매우 느리고**, 임베딩 모델 필요 |
| **추천 사용 사례** | 대부분의 일반적인 텍스트 | 구조가 명확한 짧은 문서 | LLM 비용/성능 최적화 필요 시 | RAG 검색 정확도가 매우 중요할 때 |
