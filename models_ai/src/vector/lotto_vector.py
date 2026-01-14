"""
Vector 기반 로또 번호 생성 모델
- 특성 벡터 추출
- K-Means 클러스터링으로 당첨 패턴 학습
- KNN으로 유사 패턴 기반 번호 생성
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from typing import List
import random


class LottoVectorModel:
    """벡터 임베딩 기반 로또 번호 생성 모델"""
    
    def __init__(self, ball_count: int = 6, ball_range: tuple = (1, 45),
                 n_clusters: int = 5, n_neighbors: int = 10):
        self.ball_count = ball_count
        self.min_num, self.max_num = ball_range
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.scaler = StandardScaler()
        self.kmeans = None
        self.knn = None
        self.embeddings = None
        self.draws = None
        
        # Dynamic normalization constants
        self._mid_point = (self.min_num + self.max_num) // 2
        self._max_sum = self.max_num * self.ball_count
        self._max_gap = self.max_num - 1
        self._decade_count = (self.max_num - self.min_num) // 10 + 1
        
    def extract_features(self, numbers: List[int]) -> np.ndarray:
        """로또 번호 조합에서 특성 벡터 추출 (동적 ball_count/ball_range 지원)"""
        numbers = sorted(numbers)
        n = len(numbers)  # ball_count
        
        total = sum(numbers)
        mean = np.mean(numbers)
        std = np.std(numbers)
        
        odd_count = sum(1 for num in numbers if num % 2 == 1)
        low_count = sum(1 for num in numbers if num <= self._mid_point)
        
        # Dynamic gap calculation
        gaps = [numbers[i+1] - numbers[i] for i in range(n - 1)]
        max_gap = max(gaps) if gaps else 0
        min_gap = min(gaps) if gaps else 0
        avg_gap = np.mean(gaps) if gaps else 0
        
        # AC Value (dynamic)
        differences = set()
        for i in range(n):
            for j in range(i + 1, n):
                differences.add(abs(numbers[j] - numbers[i]))
        ac_value = len(differences) - (n - 1)
        
        last_digits = [num % 10 for num in numbers]
        unique_last_digits = len(set(last_digits))
        
        # Dynamic decade counts
        decade_counts = [0] * self._decade_count
        for num in numbers:
            idx = (num - self.min_num) // 10
            if 0 <= idx < self._decade_count:
                decade_counts[idx] += 1
        decade_std = np.std(decade_counts)
        
        consecutive = sum(1 for i in range(n - 1) if numbers[i+1] - numbers[i] == 1)
        
        # Normalized features (dynamic denominators)
        return np.array([
            total / self._max_sum,                    # sum normalized
            mean / self.max_num,                      # mean normalized
            std / (self.max_num / 3),                 # std normalized
            odd_count / n,                            # odd ratio
            low_count / n,                            # low ratio
            max_gap / self._max_gap,                  # max gap normalized
            min_gap / 10,                             # min gap normalized
            avg_gap / (self._max_gap / n),            # avg gap normalized
            ac_value / 10,                            # AC value normalized
            unique_last_digits / n,                   # unique last digits ratio
            decade_std,                               # decade distribution std
            consecutive / (n - 1) if n > 1 else 0,    # consecutive ratio
            numbers[0] / self.max_num,                # first number normalized
            numbers[-1] / self.max_num,               # last number normalized
            (numbers[-1] - numbers[0]) / self._max_gap,  # spread normalized
        ])
    
    def fit(self, draws: List[List[int]]):
        """과거 당첨 데이터로 모델 학습"""
        self.draws = draws
        embeddings = np.array([self.extract_features(d) for d in draws])
        self.embeddings = self.scaler.fit_transform(embeddings)
        
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.kmeans.fit(self.embeddings)
        
        self.knn = NearestNeighbors(n_neighbors=self.n_neighbors, metric='cosine')
        self.knn.fit(self.embeddings)
        
        print(f"✓ Vector 모델 학습 완료: {len(draws)}개 패턴, {self.n_clusters}개 클러스터 "
              f"(balls: {self.ball_count}, range: {self.min_num}-{self.max_num})")
        
    def generate(self, count: int = 5, temperature: float = 1.0) -> List[dict]:
        """벡터 유사도 기반 번호 생성 (동적 ball_count/ball_range 지원)"""
        if self.embeddings is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        recent_embeddings = self.embeddings[-10:]
        target_vector = np.mean(recent_embeddings, axis=0)
        
        results = []
        attempts = 0
        max_attempts = count * 50
        
        # Dynamic AC threshold based on ball count
        ac_threshold = 7 if self.ball_count == 6 else 5
        
        while len(results) < count and attempts < max_attempts:
            attempts += 1
            # Dynamic range and count
            numbers = sorted(random.sample(range(self.min_num, self.max_num + 1), self.ball_count))
            
            features = self.extract_features(numbers)
            ac_value = int(features[8] * 10)
            if ac_value < ac_threshold:
                continue
            
            embedding = self.scaler.transform([features])[0]
            cluster = self.kmeans.predict([embedding])[0]
            
            distances, indices = self.knn.kneighbors([embedding])
            avg_similarity = 1 - np.mean(distances[0])
            
            min_sim = 0.6 - (temperature * 0.1)
            if min_sim <= avg_similarity <= 0.95:
                if numbers not in [r['numbers'] for r in results]:
                    similar_draws = [self.draws[i] for i in indices[0][:3]]
                    results.append({
                        'numbers': numbers,
                        'cluster': int(cluster),
                        'similarity': round(avg_similarity, 3),
                        'ac_value': ac_value,
                        'similar_patterns': similar_draws
                    })
        
        return results
    
    def save(self, path: str):
        """모델 저장"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler, 'kmeans': self.kmeans,
                'knn': self.knn, 'embeddings': self.embeddings,
                'draws': self.draws, 'n_clusters': self.n_clusters,
                'n_neighbors': self.n_neighbors,
                'ball_count': self.ball_count,
                'ball_range': (self.min_num, self.max_num)
            }, f)
        print(f"✓ 모델 저장: {path} (balls: {self.ball_count}, range: {self.min_num}-{self.max_num})")
    
    @classmethod
    def load(cls, path: str) -> 'LottoVectorModel':
        """모델 로드 (기존 모델 호환성 유지)"""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        # Backward compatibility: 기존 모델에 ball_count/ball_range가 없을 수 있음
        ball_count = data.get('ball_count', 6)
        ball_range = data.get('ball_range', (1, 45))
        
        model = cls(
            ball_count=ball_count,
            ball_range=ball_range,
            n_clusters=data['n_clusters'],
            n_neighbors=data['n_neighbors']
        )
        model.scaler = data['scaler']
        model.kmeans = data['kmeans']
        model.knn = data['knn']
        model.embeddings = data['embeddings']
        model.draws = data['draws']
        return model


def create_model(config: dict = None) -> LottoVectorModel:
    """모델 생성 팩토리 (멀티 로또 지원)"""
    config = config or {}
    
    # ball_range can be list or tuple
    ball_range = config.get('ball_range', (1, 45))
    if isinstance(ball_range, list):
        ball_range = tuple(ball_range)
    
    return LottoVectorModel(
        ball_count=config.get('ball_count', 6),
        ball_range=ball_range,
        n_clusters=config.get('n_clusters', 5),
        n_neighbors=config.get('n_neighbors', 10)
    )


if __name__ == "__main__":
    sample_draws = [
        [1, 7, 15, 23, 38, 42], [3, 12, 19, 28, 35, 44],
        [5, 14, 22, 31, 39, 45], [2, 9, 17, 25, 33, 41],
    ] * 25
    
    model = create_model()
    model.fit(sample_draws)
    
    print("\n🔮 생성된 번호:")
    for r in model.generate(count=3):
        print(f"  {r['numbers']} | 유사도: {r['similarity']:.1%} | AC: {r['ac_value']}")
