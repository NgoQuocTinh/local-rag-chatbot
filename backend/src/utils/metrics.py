"""
Metrics tracking for RAG system

Học:  Measuring system performance
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from config.setting import get_settings

@dataclass
class QueryMetrics:
    """Metrics for a single query"""
    timestamp: str
    query:  str
    response_time: float          # Total time (seconds)
    retrieval_time:  float         # Time to retrieve docs
    generation_time: float        # Time to generate response
    num_retrieved_docs: int
    avg_similarity_score: float   # Average similarity of retrieved docs
    success:  bool
    error: Optional[str] = None

class MetricsTracker:
    """
    Track and persist metrics
    
    Học: Singleton pattern for metrics
    """
    _instance:  Optional['MetricsTracker'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.settings = get_settings()
        self.metrics_file = Path(self.settings.paths.metrics_file)
        self.metrics:  list[QueryMetrics] = []
        self.query_count = 0
        
        # Load existing metrics
        if self.metrics_file.exists():
            self._load_metrics()
        
        self._initialized = True
    
    def _load_metrics(self):
        """Load metrics from file"""
        try:
            with open(self.metrics_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.metrics = [
                    QueryMetrics(**m) for m in data.get('queries', [])
                ]
                self.query_count = data.get('query_count', 0)
        except Exception: 
            pass
    
    def _save_metrics(self):
        """Save metrics to file"""
        try:
            self.metrics_file.parent.mkdir(exist_ok=True)
            
            data = {
                'query_count': self.query_count,
                'queries': [asdict(m) for m in self.metrics],
                'summary': self. get_summary()
            }
            
            with open(self. metrics_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to save metrics: {e}")
    
    def track_query(
        self,
        query: str,
        response_time: float,
        retrieval_time:  float,
        generation_time:  float,
        num_retrieved_docs: int,
        avg_similarity_score: float,
        success: bool = True,
        error: Optional[str] = None
    ):
        """Track a query"""
        metric = QueryMetrics(
            timestamp=datetime.now().isoformat(),
            query=query,
            response_time=response_time,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            num_retrieved_docs=num_retrieved_docs,
            avg_similarity_score=avg_similarity_score,
            success=success,
            error=error
        )
        
        self.metrics.append(metric)
        self.query_count += 1
        
        # Auto-save periodically
        save_interval = self.settings.metrics.save_interval
        if self.query_count % save_interval == 0:
            self._save_metrics()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        if not self.metrics:
            return {}
        
        successful = [m for m in self.metrics if m.success]
        
        if not successful:
            return {'total_queries': len(self.metrics), 'successful': 0}
        
        return {
            'total_queries': len(self.metrics),
            'successful': len(successful),
            'failed': len(self.metrics) - len(successful),
            'avg_response_time': sum(m.response_time for m in successful) / len(successful),
            'avg_retrieval_time': sum(m.retrieval_time for m in successful) / len(successful),
            'avg_generation_time': sum(m.generation_time for m in successful) / len(successful),
            'avg_similarity_score': sum(m.avg_similarity_score for m in successful) / len(successful),
        }
    
    def save(self):
        """Force save metrics"""
        self._save_metrics()

# Singleton instance
metrics_tracker = MetricsTracker()