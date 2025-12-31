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
    
    def __init__(self, n_clusters: int = 5, n_neighbors: int = 10):
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.scaler = StandardScaler()
        self.kmeans = None
        self.knn = None
        self.embeddings = None
        self.draws = None
        
    def extract_features(self, numbers: List[int]) -> np.ndarray:
        """로또 번호 조합에서 특성 벡터 추출"""
        numbers = sorted(numbers)
        
        total = sum(numbers)
        mean = np.mean(numbers)
        std = np.std(numbers)
        
        odd_count = sum(1 for n in numbers if n % 2 == 1)
        low_count = sum(1 for n in numbers if n <= 22)
        
        gaps = [numbers[i+1] - numbers[i] for i in range(5)]
        max_gap = max(gaps)
        min_gap = min(gaps)
        avg_gap = np.mean(gaps)
        
        # AC값
        differences = set()
        for i in range(len(numbers)):
            for j in range(i + 1, len(numbers)):
                differences.add(abs(numbers[j] - numbers[i]))
        ac_value = len(differences) - 5
        
        last_digits = [n % 10 for n in numbers]
        unique_last_digits = len(set(last_digits))
        
        decade_counts = [0] * 5
        for n in numbers:
            decade_counts[(n - 1) // 10] += 1
        decade_std = np.std(decade_counts)
        
        consecutive = sum(1 for i in range(5) if numbers[i+1] - numbers[i] == 1)
        
        return np.array([
            total / 270, mean / 45, std / 15,
            odd_count / 6, low_count / 6,
            max_gap / 40, min_gap / 10, avg_gap / 10,
            ac_value / 10, unique_last_digits / 6,
            decade_std, consecutive / 5,
            numbers[0] / 45, numbers[5] / 45,
            (numbers[5] - numbers[0]) / 44,
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
        
        print(f"✓ Vector 모델 학습 완료: {len(draws)}개 패턴, {self.n_clusters}개 클러스터")
        
    def generate(self, count: int = 5, temperature: float = 1.0) -> List[dict]:
        """벡터 유사도 기반 번호 생성"""
        if self.embeddings is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        recent_embeddings = self.embeddings[-10:]
        target_vector = np.mean(recent_embeddings, axis=0)
        
        results = []
        attempts = 0
        max_attempts = count * 50
        
        while len(results) < count and attempts < max_attempts:
            attempts += 1
            numbers = sorted(random.sample(range(1, 46), 6))
            
            features = self.extract_features(numbers)
            ac_value = int(features[8] * 10)
            if ac_value < 7:
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
                'n_neighbors': self.n_neighbors
            }, f)
        print(f"✓ 모델 저장: {path}")
    
    @classmethod
    def load(cls, path: str) -> 'LottoVectorModel':
        """모델 로드"""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls(n_clusters=data['n_clusters'], n_neighbors=data['n_neighbors'])
        model.scaler = data['scaler']
        model.kmeans = data['kmeans']
        model.knn = data['knn']
        model.embeddings = data['embeddings']
        model.draws = data['draws']
        return model


def create_model(config: dict = None) -> LottoVectorModel:
    """모델 생성 팩토리"""
    config = config or {}
    return LottoVectorModel(
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
