import numpy as np
from typing import List, Dict

class LottoEnsembleModel:
    """
    앙상블 모델 (Ensemble Model)
    
    Transformer, LSTM, Physics 등 여러 모델의 예측 결과를 결합하여
    최종 추천 번호를 생성하는 가중 평균(Weighted Average) 기반 모델입니다.
    """
    
    def __init__(self):
        # 모델별 기본 가중치 설정
        self.weights = {
            'transformer': 0.45, # 패턴 인식 (가장 높은 신뢰도)
            'lstm': 0.35,        # 시계열 추세
            'physics': 0.20      # 물리적 편향 보정
        }

    def predict(self, model_outputs: Dict[str, List[int]], top_k: int = 6) -> List[int]:
        """
        여러 모델의 예측 결과를 종합하여 최종 번호를 선정합니다.
        
        Args:
            model_outputs: 각 모델의 예측 번호 리스트 
                           예: {'transformer': [1, 5, ...], 'lstm': [2, 5, ...]}
            top_k: 최종적으로 반환할 번호 개수
            
        Returns:
            List[int]: 최종 추천 번호 리스트
        """
        
        # 1. 빈도 및 가중치 점수 계산
        number_scores = {}
        
        for model_name, numbers in model_outputs.items():
            weight = self.weights.get(model_name, 0.1)
            
            # 상위 번호일수록 높은 점수 부여 (순위 가중치)
            for rank, number in enumerate(numbers):
                # rank 0(1위) -> score 1.0, rank N -> score 낮아짐
                rank_score = 1.0 / (rank + 1)
                
                # 최종 점수 = 모델 가중치 * 순위 점수
                score = weight * rank_score
                
                if number in number_scores:
                    number_scores[number] += score
                else:
                    number_scores[number] = score
                    
        # 2. 점수 기준 정렬
        sorted_numbers = sorted(number_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 3. Top K 선정
        recommended_numbers = [num for num, score in sorted_numbers[:top_k]]
        
        # 4. 정렬하여 반환 (로또 번호는 오름차순이 일반적)
        return sorted(recommended_numbers)

    def set_weights(self, new_weights: Dict[str, float]):
        """모델 가중치를 동적으로 조정합니다."""
        self.weights.update(new_weights)
        print(f"⚖️ Ensemble weights updated: {self.weights}")

if __name__ == "__main__":
    # 테스트 코드
    ensemble = LottoEnsembleModel()
    
    # 가상의 모델 예측 결과
    mock_outputs = {
        'transformer': [10, 23, 45, 2, 8, 15],
        'lstm': [23, 8, 12, 45, 30, 1],
        'physics': [2, 10, 33, 23, 44, 11]
    }
    
    result = ensemble.predict(mock_outputs)
    print(f"Inputs: {mock_outputs}")
    print(f"Ensemble Result: {result}")
