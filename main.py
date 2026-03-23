import os
import json
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel

from Data_Base import get_db, init_db, ScanLog, ThreatSummary
from models.Email_Classifier import EmailPhishingClassifier
from models.Url_Classifier import URLThreatClassifier

# ── App setup ─────────────────────────────────────────────────

app = FastAPI(
    title="AI Threat Detection API",
    description="Detects phishing emails, malicious URLs, and suspicious network activity",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Load ML models ────────────────────────────────────────────

email_classifier = EmailPhishingClassifier().load("models/email_model.pkl")
url_classifier = URLThreatClassifier().load("models/url_model.pkl")

# ── Schemas ───────────────────────────────────────────────────

class EmailScanRequest(BaseModel):
    subject: str
    body: str
    sender: Optional[str] = None

class URLScanRequest(BaseModel):
    url: str

class NetworkScanRequest(BaseModel):
    ip_address: str
    domain: Optional[str] = None
    request_count: Optional[int] = None
    user_agent: Optional[str] = None

# ── Startup ───────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    init_db()
    print("✅ Database initialized. Models loaded.")

# ── Routes ────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": True}


@app.post("/scan/email")
async def scan_email(
    request: EmailScanRequest,
    req: Request,
    db: Session = Depends(get_db)
):
    full_text = f"Subject: {request.subject}\n\nBody: {request.body}"
    result = email_classifier.predict(full_text)

    log = ScanLog(
        scan_type="email",
        input_data=full_text[:500],
        prediction=result["prediction"],
        risk_level=result["risk_level"],
        confidence=result["confidence"],
        malicious_score=result["phishing_score"],
        ip_address=req.client.host,
    )
    db.add(log)
    db.commit()
    db.refresh(log)

    return {"scan_id": log.id, "timestamp": log.timestamp, **result}


@app.post("/scan/url")
async def scan_url(
    request: URLScanRequest,
    req: Request,
    db: Session = Depends(get_db)
):
    result = url_classifier.predict(request.url)

    log = ScanLog(
        scan_type="url",
        input_data=request.url,
        prediction=result["prediction"],
        risk_level=result["risk_level"],
        confidence=result["confidence"],
        malicious_score=result["malicious_score"],
        ip_address=req.client.host,
        extra_data=json.dumps(result.get("features", {}))
    )
    db.add(log)
    db.commit()
    db.refresh(log)

    return {"scan_id": log.id, "timestamp": log.timestamp, **result}


@app.get("/logs")
async def get_logs(
    limit: int = 50,
    scan_type: Optional[str] = None,
    risk_level: Optional[str] = None,
    db: Session = Depends(get_db)
):
    query = db.query(ScanLog)

    if scan_type:
        query = query.filter(ScanLog.scan_type == scan_type)
    if risk_level:
        query = query.filter(ScanLog.risk_level == risk_level)

    logs = query.order_by(ScanLog.timestamp.desc()).limit(limit).all()

    return [
        {
            "id": log.id,
            "scan_type": log.scan_type,
            "prediction": log.prediction,
            "risk_level": log.risk_level,
            "confidence": log.confidence,
            "timestamp": log.timestamp,
            "input_preview": (log.input_data[:80] + "...") if log.input_data and len(log.input_data) > 80 else log.input_data
        }
        for log in logs
    ]


@app.get("/stats")
async def get_stats(db: Session = Depends(get_db)):
    total = db.query(ScanLog).count()
    threats = db.query(ScanLog).filter(
        ScanLog.prediction.in_(["phishing", "malicious"])
    ).count()

    critical = db.query(ScanLog).filter(ScanLog.risk_level == "CRITICAL").count()
    high = db.query(ScanLog).filter(ScanLog.risk_level == "HIGH").count()

    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    recent = db.query(ScanLog).filter(ScanLog.timestamp >= seven_days_ago).all()

    daily = {}
    for log in recent:
        day = log.timestamp.strftime("%Y-%m-%d") if log.timestamp else "unknown"
        if day not in daily:
            daily[day] = {"total": 0, "threats": 0}
        daily[day]["total"] += 1
        if log.prediction in ["phishing", "malicious"]:
            daily[day]["threats"] += 1

    return {
        "total_scans": total,
        "threats_detected": threats,
        "threat_rate": round((threats / total * 100) if total > 0 else 0, 1),
        "critical_count": critical,
        "high_count": high,
        "daily_breakdown": daily
    }
