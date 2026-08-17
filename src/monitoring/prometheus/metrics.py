from prometheus_client import Counter, Histogram

prediction_latency = Histogram(
    'prediction_latency_seconds',
    'Time taken for model prediction',
    ['model_version', 'client_type']
)

confidence_score_metric = Histogram(
    'confidence_score',
    'Model confidence_score per prediction',
    ['model_version'],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

client_requests = Counter(
    'client_requests_total',
    'Total request by client_type',
    ['client_type']
)
