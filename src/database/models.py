from sqlalchemy import (
    Column,
    DateTime,
    Float,
    String,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class ModelPredictionLog(Base):
    __tablename__ = "model_prediction_logs"

    request_id = Column(String, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    client_type = Column(String, nullable=False)
    model_version = Column(String, nullable=False)
    input_features = Column(JSONB, nullable=False)
    predicted_output = Column(String, nullable=False)
    confidence_score = Column(Float, nullable=False)
    latency = Column(Float, nullable=False)
    truth_label = Column(String, nullable=True)
