# database.py
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# DATABASE_URL can be swapped to PostgreSQL just by changing this string
DATABASE_URL = "sqlite:///./threat_detector.db"
# For PostgreSQL: "postgresql://user:password@localhost/threat_detector"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class ScanLog(Base):
    """
    Every scan result gets stored here.
    This is our audit trail — critical for security compliance.
    """
    __tablename__ = "scan_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    scan_type = Column(String(20))       # "email", "url", "network"
    input_data = Column(Text)            # The raw input (email body / URL)
    prediction = Column(String(20))      # "phishing", "legitimate", "malicious", "safe"
    risk_level = Column(String(10))      # "CRITICAL", "HIGH", "MEDIUM", "LOW"
    confidence = Column(Float)           # 0.0 - 100.0
    malicious_score = Column(Float)      # Specific threat score
    ip_address = Column(String(50))      # Who submitted it (for audit)
    timestamp = Column(DateTime, default=datetime.utcnow)
    extra_data = Column(Text)            # JSON string for additional features


class ThreatSummary(Base):
    """Aggregated daily stats for the dashboard charts"""
    __tablename__ = "threat_summaries"
    
    id = Column(Integer, primary_key=True, index=True)
    date = Column(String(10))            # "2024-01-15"
    total_scans = Column(Integer, default=0)
    threats_detected = Column(Integer, default=0)
    critical_count = Column(Integer, default=0)
    high_count = Column(Integer, default=0)


def get_db():
    """
    Dependency injection pattern for FastAPI.
    Creates a DB session per request, closes it when done.
    This is important — you don't want DB connections leaking.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Creates all tables if they don't exist"""
    Base.metadata.create_all(bind=engine)