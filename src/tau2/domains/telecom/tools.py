"""Toolkit for the telecom system."""

import os
import uuid
from collections import defaultdict
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from loguru import logger

from tau2.domains.telecom.data_model import (
    Bill,
    BillStatus,
    Customer,
    Device,
    Line,
    LineItem,
    LineStatus,
    Plan,
    TelecomDB,
)
from tau2.domains.telecom.utils import get_today
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool

# TODO: Add an abstract base class for the tools


class IDGenerator:
    def __init__(self) -> None:
        self.id_counter = defaultdict(int)

    def get_id(self, id_type: str, id_name: Optional[str] = None) -> str:
        self.id_counter[id_type] += 1
        id_name = id_name or id_type
        return f"{id_name}_{self.id_counter[id_type]}"


# Configuration for verbose responses to generate long trajectories
# Target: 150k-300k tokens per trajectory with ~10 tool calls
# => 15k-30k tokens per tool response => 60k-120k chars per response
# Environment variable to enable/disable verbose responses
# Set TAU2_VERBOSE_RESPONSES=1 to enable, TAU2_VERBOSE_RESPONSES=0 to disable
VERBOSE_RESPONSES_ENABLED = os.environ.get("TAU2_VERBOSE_RESPONSES", "0") == "1"
# Target: Stay well under 128k context window (GPT-4.1 limit)
# NL evaluator also sends full trajectory, so need headroom
# With ~20 tool calls, need ~3k tokens per response = ~60k total
# Leaves ~68k for conversation + system prompt + NL eval overhead
# Actual output is ~25% of target, so 12k target = ~3k actual
VERBOSE_TARGET_TOKENS_PER_RESPONSE = 12000  # ~3k actual tokens per response
VERBOSE_TARGET_CHARS = VERBOSE_TARGET_TOKENS_PER_RESPONSE * 4  # ~48k chars target

# Flag to suppress verbose output to terminal (reduces noise when debugging)
VERBOSE_SUPPRESS_TERMINAL = os.environ.get("TAU2_VERBOSE_SUPPRESS_TERMINAL", "1") == "1"


def add_verbose_padding_to_response(func):
    """Decorator that adds verbose padding to dict responses for long trajectories."""
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        
        # Only add padding to dict responses when verbose mode is enabled
        if VERBOSE_RESPONSES_ENABLED and isinstance(result, dict):
            # Extract entity info for audit logs
            entity_id = result.get('customer_id') or result.get('line_id') or \
                       result.get('bill_id') or result.get('device_id') or \
                       result.get('plan_id') or result.get('entity_id') or 'unknown'
            entity_type = result.get('entity_type', 'entity').lower()
            
            # Add verbose padding
            result.update(_generate_verbose_padding(str(entity_id), entity_type))
        
        return result
    
    return wrapper


def _generate_audit_log(entity_id: str, entity_type: str, num_entries: int = 500) -> List[Dict]:
    """Generate fake audit log entries for verbose responses."""
    import random
    actions = [
        "VIEW", "UPDATE", "CREATE", "VALIDATE", "SYNC", "CACHE_HIT", "CACHE_MISS",
        "AUTH_CHECK", "PERMISSION_GRANT", "RATE_LIMIT_CHECK", "ENCRYPTION_VERIFY",
        "COMPLIANCE_CHECK", "DATA_MASK", "LOG_WRITE", "METRIC_EMIT", "TRACE_START",
        "TRACE_END", "SPAN_CREATE", "CONTEXT_PROPAGATE", "RETRY_ATTEMPT"
    ]
    sources = [
        "api-gateway-prod-us-east-1", "auth-service-v2.3.1", "billing-engine-core",
        "customer-profile-service", "network-ops-controller", "compliance-monitor",
        "fraud-detection-ml", "rate-limiter-redis", "cache-layer-memcached",
        "message-queue-kafka", "event-processor-lambda", "data-pipeline-spark"
    ]
    entries = []
    base_time = "2025-02-25T12:08:00.000Z"
    for i in range(num_entries):
        entries.append({
            "log_id": f"LOG-{uuid.uuid4().hex[:12].upper()}",
            "timestamp": f"2025-02-25T{10 + (i % 3):02d}:{(i * 7) % 60:02d}:{(i * 13) % 60:02d}.{random.randint(100,999)}Z",
            "entity_id": entity_id,
            "entity_type": entity_type,
            "action": random.choice(actions),
            "source_system": random.choice(sources),
            "correlation_id": f"COR-{uuid.uuid4().hex[:16].upper()}",
            "trace_id": f"TRC-{uuid.uuid4().hex[:32]}",
            "span_id": f"SPN-{uuid.uuid4().hex[:16]}",
            "user_agent": "TelecomAgentAPI/2.1.0 (internal-service)",
            "ip_address": f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}",
            "latency_ms": random.randint(1, 500),
            "status_code": random.choice([200, 200, 200, 200, 201, 204, 304]),
            "request_size_bytes": random.randint(100, 5000),
            "response_size_bytes": random.randint(500, 50000),
            "metadata": {
                "region": random.choice(["us-east-1", "us-west-2", "eu-west-1"]),
                "availability_zone": random.choice(["a", "b", "c"]),
                "instance_id": f"i-{uuid.uuid4().hex[:17]}",
                "container_id": f"ctr-{uuid.uuid4().hex[:12]}",
                "kubernetes_pod": f"pod-telecom-api-{uuid.uuid4().hex[:8]}",
                "deployment_version": f"v2.1.{random.randint(0,99)}-{uuid.uuid4().hex[:7]}"
            }
        })
    return entries


def _generate_compliance_records(entity_id: str, num_records: int = 100) -> List[Dict]:
    """Generate fake compliance and regulatory records."""
    import random
    regulations = [
        "GDPR-Article-17", "CCPA-Section-1798.100", "TCPA-47-USC-227", 
        "FCC-Part-64", "PCI-DSS-3.2.1", "SOX-Section-404", "HIPAA-164.312",
        "ISO-27001-A.12", "SOC2-CC6.1", "NIST-800-53-AC-2"
    ]
    records = []
    for i in range(num_records):
        records.append({
            "compliance_record_id": f"COMP-{uuid.uuid4().hex[:10].upper()}",
            "regulation": random.choice(regulations),
            "entity_reference": entity_id,
            "check_timestamp": f"2025-02-{random.randint(1,25):02d}T{random.randint(0,23):02d}:{random.randint(0,59):02d}:00Z",
            "status": random.choice(["COMPLIANT", "COMPLIANT", "COMPLIANT", "PENDING_REVIEW"]),
            "auditor_system": f"compliance-bot-v{random.randint(1,5)}.{random.randint(0,9)}",
            "evidence_hash": f"sha256:{uuid.uuid4().hex}{uuid.uuid4().hex[:32]}",
            "retention_policy": f"RETAIN_{random.choice([30, 90, 365, 730])}_DAYS",
            "data_classification": random.choice(["PII", "SENSITIVE", "INTERNAL", "PUBLIC"]),
            "encryption_status": "AES-256-GCM",
            "access_log_reference": f"ACCESS-LOG-{uuid.uuid4().hex[:12]}",
            "last_access_review": f"2025-02-{random.randint(1,20):02d}",
            "next_review_due": f"2025-03-{random.randint(1,28):02d}",
            "risk_score": round(random.uniform(0.0, 0.3), 4),
            "control_effectiveness": round(random.uniform(0.85, 0.99), 4)
        })
    return records


def _generate_similar_cases(entity_type: str, num_cases: int = 50) -> List[Dict]:
    """Generate fake similar customer cases for context."""
    import random
    issue_types = [
        "billing_inquiry", "service_disruption", "plan_change", "data_overage",
        "roaming_issue", "device_troubleshoot", "payment_failure", "account_update"
    ]
    resolutions = [
        "resolved_by_agent", "escalated_to_tier2", "customer_callback", 
        "automated_fix", "credit_applied", "plan_adjusted", "device_replaced"
    ]
    cases = []
    for i in range(num_cases):
        cases.append({
            "case_id": f"CASE-{random.randint(100000, 999999)}",
            "similarity_score": round(random.uniform(0.65, 0.95), 4),
            "issue_type": random.choice(issue_types),
            "resolution": random.choice(resolutions),
            "resolution_time_minutes": random.randint(5, 180),
            "customer_satisfaction": random.randint(3, 5),
            "agent_id": f"AGT-{random.randint(1000, 9999)}",
            "created_date": f"2025-{random.randint(1,2):02d}-{random.randint(1,28):02d}",
            "summary": f"Customer contacted regarding {random.choice(issue_types)} issue. "
                      f"After troubleshooting, issue was {random.choice(['resolved', 'escalated', 'pending'])}. "
                      f"Root cause identified as {random.choice(['system_error', 'user_configuration', 'network_issue', 'billing_discrepancy'])}.",
            "tags": random.sample(["priority", "escalated", "vip", "retention_risk", "upsell_opportunity", "technical", "billing"], k=random.randint(1, 4)),
            "sentiment_analysis": {
                "overall": random.choice(["positive", "neutral", "negative"]),
                "frustration_level": round(random.uniform(0, 1), 2),
                "urgency_detected": random.choice([True, False])
            }
        })
    return cases


def _generate_system_diagnostics(num_entries: int = 200) -> Dict:
    """Generate fake system diagnostic information."""
    import random
    return {
        "system_health": {
            "overall_status": "HEALTHY",
            "last_health_check": "2025-02-25T12:07:55Z",
            "uptime_percentage_30d": 99.97,
            "active_incidents": 0,
            "scheduled_maintenance": None
        },
        "performance_metrics": [
            {
                "metric_name": f"metric_{i}",
                "value": round(random.uniform(0, 100), 4),
                "unit": random.choice(["ms", "percent", "count", "bytes"]),
                "timestamp": f"2025-02-25T12:{random.randint(0,8):02d}:{random.randint(0,59):02d}Z",
                "aggregation": random.choice(["avg", "p50", "p95", "p99", "max"]),
                "threshold_warning": random.randint(50, 80),
                "threshold_critical": random.randint(80, 100)
            }
            for i in range(num_entries)
        ],
        "cache_statistics": {
            "hit_rate": 0.94,
            "miss_rate": 0.06,
            "eviction_count_24h": random.randint(100, 1000),
            "memory_used_mb": random.randint(500, 2000),
            "memory_limit_mb": 4096,
            "entries_count": random.randint(10000, 100000)
        },
        "database_connections": {
            "active": random.randint(10, 50),
            "idle": random.randint(5, 20),
            "max_pool_size": 100,
            "wait_queue_length": 0
        },
        "recent_errors": [
            {
                "error_id": f"ERR-{uuid.uuid4().hex[:8]}",
                "timestamp": f"2025-02-25T{random.randint(0,11):02d}:{random.randint(0,59):02d}:00Z",
                "error_type": random.choice(["TimeoutException", "ValidationError", "RateLimitExceeded"]),
                "count": random.randint(1, 10),
                "last_occurrence": f"2025-02-25T{random.randint(0,11):02d}:{random.randint(0,59):02d}:00Z",
                "auto_resolved": random.choice([True, True, True, False])
            }
            for _ in range(20)
        ]
    }


def _generate_network_topology(num_nodes: int = 100) -> Dict:
    """Generate fake network topology information."""
    import random
    return {
        "topology_version": "2025.02.25.001",
        "last_updated": "2025-02-25T06:00:00Z",
        "nodes": [
            {
                "node_id": f"NODE-{uuid.uuid4().hex[:8].upper()}",
                "node_type": random.choice(["router", "switch", "gateway", "tower", "antenna"]),
                "location": {
                    "lat": round(random.uniform(25, 48), 6),
                    "lon": round(random.uniform(-125, -70), 6),
                    "region": random.choice(["northeast", "southeast", "midwest", "southwest", "west"]),
                    "city": random.choice(["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Denver", "Seattle"])
                },
                "status": random.choice(["active", "active", "active", "maintenance"]),
                "capacity_utilization": round(random.uniform(0.3, 0.85), 4),
                "connected_devices": random.randint(100, 10000),
                "bandwidth_gbps": random.choice([1, 10, 40, 100]),
                "latency_ms": round(random.uniform(1, 50), 2),
                "firmware_version": f"v{random.randint(1,5)}.{random.randint(0,9)}.{random.randint(0,99)}"
            }
            for _ in range(num_nodes)
        ],
        "connections": [
            {
                "connection_id": f"CONN-{uuid.uuid4().hex[:6]}",
                "source_node": f"NODE-{uuid.uuid4().hex[:8].upper()}",
                "target_node": f"NODE-{uuid.uuid4().hex[:8].upper()}",
                "link_type": random.choice(["fiber", "microwave", "satellite"]),
                "capacity_gbps": random.choice([1, 10, 40, 100]),
                "current_load_percent": round(random.uniform(10, 80), 2)
            }
            for _ in range(num_nodes * 2)
        ]
    }


# ==============================================================================
# CONFOUNDING DATA GENERATORS
# These create semantically similar but INCORRECT data to test agent accuracy
# ==============================================================================

def _generate_confounding_customers(actual_customer_id: str, actual_name: str, actual_phone: str) -> Dict:
    """Generate similar customers that could confuse the agent."""
    import random
    
    # Generate variations of the name
    first_name = actual_name.split()[0] if ' ' in actual_name else actual_name
    last_name = actual_name.split()[-1] if ' ' in actual_name else "Smith"
    
    similar_names = [
        f"{first_name} {last_name[:-1]}i",  # Lee -> Li
        f"{first_name[:-1]}a {last_name}",  # Michael -> Michala  
        f"{first_name} {last_name} Jr.",
        f"{first_name[0]}. {last_name}",
        f"{first_name} {last_name}-Williams",
    ]
    
    # Generate similar phone numbers (off by 1 digit)
    phone_digits = actual_phone.replace("-", "")
    similar_phones = []
    for i in range(min(3, len(phone_digits))):
        new_phone = list(phone_digits)
        new_phone[-(i+1)] = str((int(new_phone[-(i+1)]) + 1) % 10)
        similar_phones.append("-".join(["".join(new_phone[:3]), "".join(new_phone[3:6]), "".join(new_phone[6:])]))
    
    return {
        "_similar_customers_in_database": [
            {
                "customer_id": f"C{random.randint(2000, 9999)}",
                "full_name": name,
                "phone_number": similar_phones[i % len(similar_phones)] if similar_phones else actual_phone,
                "account_status": random.choice(["Active", "Suspended", "Pending"]),
                "similarity_score": round(random.uniform(0.75, 0.95), 3),
                "match_reason": random.choice(["name_fuzzy_match", "phone_partial_match", "address_proximity"])
            }
            for i, name in enumerate(similar_names)
        ],
        "_historical_customer_records": [
            {
                "record_id": f"HIST-{uuid.uuid4().hex[:8]}",
                "customer_id": actual_customer_id,
                "previous_name": f"{first_name} {last_name} (maiden: {random.choice(['Jones', 'Davis', 'Wilson'])})",
                "previous_phone": similar_phones[0] if similar_phones else actual_phone,
                "change_date": f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
                "change_type": random.choice(["name_change", "phone_update", "address_update"])
            }
            for _ in range(3)
        ],
        "_customer_merge_candidates": [
            {
                "candidate_id": f"C{random.randint(1000, 9999)}",
                "merge_score": round(random.uniform(0.60, 0.85), 3),
                "potential_duplicate": True,
                "recommendation": "REVIEW_REQUIRED"
            }
            for _ in range(2)
        ]
    }


def _generate_confounding_bills(actual_bill_id: str, actual_amount: float, actual_status: str) -> Dict:
    """Generate similar bills with nearby amounts to confuse the agent."""
    import random
    
    # Generate amounts close to the actual amount
    nearby_amounts = [
        round(actual_amount + 0.50, 2),
        round(actual_amount - 0.01, 2),
        round(actual_amount + 5.00, 2),
        round(actual_amount - 5.00, 2),
        round(actual_amount * 1.1, 2),  # 10% more
    ]
    
    statuses = ["Paid", "Overdue", "Issued", "Draft", "Disputed"]
    other_statuses = [s for s in statuses if s != actual_status]
    
    return {
        "_bills_from_similar_accounts": [
            {
                "bill_id": f"B{random.randint(2000, 9999)}",
                "customer_id": f"C{random.randint(1000, 9999)}",
                "total_due": amount,
                "status": random.choice(other_statuses),
                "due_date": f"2025-02-{random.randint(15, 28):02d}",
                "note": "Different customer - shown for comparison"
            }
            for amount in nearby_amounts[:3]
        ],
        "_historical_bills_same_customer": [
            {
                "bill_id": f"B{random.randint(100, 999)}",
                "period": f"2024-{random.randint(1,12):02d}",
                "total_due": round(actual_amount + random.uniform(-10, 10), 2),
                "status": "Paid",
                "paid_date": f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
            }
            for _ in range(5)
        ],
        "_billing_projections": {
            "estimated_next_bill": round(actual_amount * random.uniform(0.95, 1.15), 2),
            "average_monthly": round(actual_amount * random.uniform(0.9, 1.1), 2),
            "projected_annual": round(actual_amount * 12 * random.uniform(0.95, 1.05), 2),
            "trend": random.choice(["increasing", "stable", "decreasing"])
        },
        "_similar_bill_amounts_in_system": [
            {
                "bill_id": f"B{random.randint(3000, 9999)}",
                "amount": amount,
                "customer_id": f"C{random.randint(1000, 9999)}",
                "note": "DIFFERENT CUSTOMER - amount similarity only"
            }
            for amount in nearby_amounts
        ]
    }


def _generate_confounding_lines(actual_line_id: str, actual_phone: str, actual_status: str) -> Dict:
    """Generate similar lines and status history to confuse the agent."""
    import random
    
    # Generate similar line IDs
    line_num = int(actual_line_id[1:]) if actual_line_id[1:].isdigit() else 1000
    similar_line_ids = [f"L{line_num + i}" for i in [-2, -1, 1, 2]]
    
    # Generate similar phone numbers
    phone_base = actual_phone.replace("-", "")
    similar_phones = [
        f"{phone_base[:3]}-{phone_base[3:6]}-{int(phone_base[6:]) + i:04d}"[-12:]
        for i in [-1, 1, 10, -10]
    ]
    
    statuses = ["Active", "Suspended", "Pending Activation", "Closed"]
    other_statuses = [s for s in statuses if s != actual_status]
    
    return {
        "_other_lines_on_account": [
            {
                "line_id": lid,
                "phone_number": similar_phones[i % len(similar_phones)],
                "status": random.choice(other_statuses),
                "plan": random.choice(["Basic", "Standard", "Premium", "Unlimited"]),
                "note": "Different line on same account"
            }
            for i, lid in enumerate(similar_line_ids[:3])
        ],
        "_line_status_history": [
            {
                "date": f"2025-{random.randint(1,2):02d}-{random.randint(1,28):02d}",
                "previous_status": random.choice(statuses),
                "new_status": random.choice(statuses),
                "reason": random.choice(["Payment received", "Non-payment", "Customer request", "Fraud alert", "Contract renewal"]),
                "agent_id": f"AGT-{random.randint(100, 999)}"
            }
            for _ in range(8)
        ],
        "_lines_with_similar_numbers": [
            {
                "line_id": f"L{random.randint(2000, 9999)}",
                "phone_number": phone,
                "customer_id": f"C{random.randint(1000, 9999)}",
                "status": random.choice(statuses),
                "note": "DIFFERENT CUSTOMER - phone number similarity only"
            }
            for phone in similar_phones
        ],
        "_line_recommendations": {
            "upgrade_eligible": random.choice([True, False]),
            "recommended_plan": random.choice(["Premium Plus", "Unlimited Max", "Family Share"]),
            "estimated_savings": round(random.uniform(5, 25), 2),
            "confidence": round(random.uniform(0.7, 0.95), 3)
        }
    }


def _generate_confounding_usage(actual_line_id: str, actual_used_gb: float, actual_limit_gb: float) -> Dict:
    """Generate confounding usage data for multiple lines."""
    import random
    
    line_num = int(actual_line_id[1:]) if actual_line_id[1:].isdigit() else 1000
    
    return {
        "_usage_all_lines_on_account": [
            {
                "line_id": f"L{line_num + i}",
                "data_used_gb": round(random.uniform(0.5, actual_limit_gb * 1.2), 2),
                "data_limit_gb": actual_limit_gb + random.choice([-5, 0, 5, 10]),
                "percentage_used": round(random.uniform(10, 110), 1),
                "status": "Over limit" if random.random() > 0.7 else "Normal"
            }
            for i in range(-3, 4) if i != 0
        ],
        "_historical_usage_patterns": [
            {
                "month": f"2024-{month:02d}",
                "data_used_gb": round(actual_used_gb * random.uniform(0.5, 1.5), 2),
                "overage_gb": round(max(0, random.uniform(-2, 5)), 2),
                "overage_charge": round(random.uniform(0, 15), 2)
            }
            for month in range(7, 13)
        ],
        "_usage_alerts_all_lines": [
            {
                "line_id": f"L{line_num + random.randint(-3, 3)}",
                "alert_type": random.choice(["OVER_LIMIT", "APPROACHING_LIMIT", "UNUSUAL_USAGE", "ROAMING_HIGH"]),
                "priority": random.choice(["HIGH", "MEDIUM", "LOW"]),
                "message": random.choice([
                    "Data limit exceeded",
                    "90% of data used",
                    "Unusual usage pattern detected",
                    "High roaming charges detected"
                ]),
                "timestamp": f"2025-02-{random.randint(20, 25):02d}T{random.randint(0, 23):02d}:{random.randint(0, 59):02d}:00Z"
            }
            for _ in range(6)
        ],
        "_data_recommendations": {
            "current_line_id": actual_line_id,
            "suggested_refuel_gb": random.choice([1, 2, 5]),
            "suggested_plan_upgrade": random.choice(["Unlimited Plus", "Premium Max", None]),
            "potential_savings_monthly": round(random.uniform(0, 20), 2)
        }
    }


def _generate_verbose_padding(entity_id: str, entity_type: str, target_chars: int = VERBOSE_TARGET_CHARS) -> Dict:
    """Generate verbose padding data to reach target character count."""
    if not VERBOSE_RESPONSES_ENABLED:
        return {}
    
    import json
    
    # Calculate how many entries we need based on target
    # Each audit entry is ~500 chars, so entries_needed = target_chars / 500
    entries_per_section = max(10, target_chars // 2500)  # Distribute across 5 main sections
    
    padding = {
        "_audit_log": _generate_audit_log(entity_id, entity_type, num_entries=entries_per_section),
        "_compliance_records": _generate_compliance_records(entity_id, num_records=entries_per_section // 3),
        "_similar_cases": _generate_similar_cases(entity_type, num_cases=entries_per_section // 5),
        "_system_diagnostics": _generate_system_diagnostics(num_entries=entries_per_section // 2),
        "_network_topology": _generate_network_topology(num_nodes=entries_per_section // 3),
        "_internal_metadata": {
            "response_generated_at": "2025-02-25T12:08:00.123456Z",
            "processing_pipeline": [
                {"stage": "request_validation", "duration_ms": 2, "status": "success"},
                {"stage": "authentication", "duration_ms": 5, "status": "success"},
                {"stage": "authorization", "duration_ms": 3, "status": "success"},
                {"stage": "rate_limiting", "duration_ms": 1, "status": "success"},
                {"stage": "cache_lookup", "duration_ms": 4, "status": "miss"},
                {"stage": "database_query", "duration_ms": 45, "status": "success"},
                {"stage": "data_transformation", "duration_ms": 8, "status": "success"},
                {"stage": "response_serialization", "duration_ms": 3, "status": "success"},
            ],
            "feature_flags": {
                "enhanced_logging": True,
                "ml_recommendations": True,
                "real_time_fraud_check": True,
                "predictive_analytics": True,
                "a_b_test_variant": "control"
            },
            "service_mesh_info": {
                "upstream_services_called": ["auth-service", "billing-service", "profile-service"],
                "circuit_breaker_status": "closed",
                "retry_count": 0,
                "timeout_budget_remaining_ms": 4950
            }
        }
    }
    
    # Fine-tune to reach target
    current_size = len(json.dumps(padding))
    if current_size < target_chars:
        additional_entries_needed = (target_chars - current_size) // 500
        if additional_entries_needed > 0:
            padding["_extended_audit_log"] = _generate_audit_log(
                entity_id, entity_type, num_entries=additional_entries_needed
            )
    
    return padding


class TelecomTools(ToolKitBase):
    """Tools for the telecom domain implementing the functions described in the PRD."""

    db: TelecomDB

    def __init__(self, db: TelecomDB) -> None:
        """Initialize the telecom tools with a database instance."""
        super().__init__(db)
        self.id_generator = IDGenerator()

    # Customer Lookup (internal - not exposed to agent)
    def _get_customer_by_phone_internal(self, phone_number: str) -> Customer:
        """
        Internal method to find a customer by phone number.
        Returns the Customer object directly for internal use.
        
        Args:
            phone_number: The phone number to search for.
            
        Returns:
            Customer object.
        """
        for customer in self.db.customers:
            if customer.phone_number == phone_number:
                return customer
            # Check lines
            for line_id in customer.line_ids:
                line = self._get_line_by_id(line_id)
                if line and line.phone_number == phone_number:
                    return customer
        raise ValueError(f"Customer with phone number {phone_number} not found")

    @add_verbose_padding_to_response
    def get_customer_by_phone(self, phone_number: str) -> Dict[str, Any]:
        """
        Finds a customer by their primary contact or line phone number.
        Returns comprehensive customer information including account metadata,
        line summary, billing summary, and system audit trail.

        Args:
            phone_number: The phone number to search for.

        Returns:
            Dictionary with full customer data including metadata and summaries.
        """
        # Check primary contact number
        found_customer = None
        lookup_method = None
        matched_line_id = None
        
        for customer in self.db.customers:
            if customer.phone_number == phone_number:
                found_customer = customer
                lookup_method = "primary_contact"
                break

            # Check lines
            for line_id in customer.line_ids:
                line = self._get_line_by_id(line_id)
                if line and line.phone_number == phone_number:
                    found_customer = customer
                    lookup_method = "line_association"
                    matched_line_id = line_id
                    break
            if found_customer:
                break

        if not found_customer:
            raise ValueError(f"Customer with phone number {phone_number} not found")
        
        # Build comprehensive response with additional metadata
        lines_summary = []
        total_data_used = 0.0
        active_lines = 0
        suspended_lines = 0
        
        for line_id in found_customer.line_ids:
            try:
                line = self._get_line_by_id(line_id)
                plan = self._get_plan_by_id(line.plan_id)
                device = self._get_device_by_id(line.device_id)
                total_data_used += line.data_used_gb
                if line.status == LineStatus.ACTIVE:
                    active_lines += 1
                elif line.status == LineStatus.SUSPENDED:
                    suspended_lines += 1
                    
                lines_summary.append({
                    "line_id": line.line_id,
                    "phone_number": line.phone_number,
                    "status": line.status.value,
                    "plan_name": plan.name,
                    "device_model": device.model,
                    "data_used_gb": line.data_used_gb,
                    "roaming_enabled": line.roaming_enabled,
                })
            except ValueError:
                lines_summary.append({"line_id": line_id, "error": "Could not retrieve details"})
        
        bills_summary = []
        total_outstanding = 0.0
        for bill_id in found_customer.bill_ids:
            try:
                bill = self._get_bill_by_id(bill_id)
                bills_summary.append({
                    "bill_id": bill.bill_id,
                    "total_due": bill.total_due,
                    "status": bill.status.value,
                    "due_date": str(bill.due_date),
                })
                if bill.status in [BillStatus.OVERDUE, BillStatus.ISSUED]:
                    total_outstanding += bill.total_due
            except ValueError:
                bills_summary.append({"bill_id": bill_id, "error": "Could not retrieve details"})
        
        # Build the main response
        response = {
            "customer_id": found_customer.customer_id,
            "full_name": found_customer.full_name,
            "date_of_birth": found_customer.date_of_birth,
            "email": found_customer.email,
            "phone_number": found_customer.phone_number,
            "account_status": found_customer.account_status.value,
            "address": {
                "street": found_customer.address.street,
                "city": found_customer.address.city,
                "state": found_customer.address.state,
                "zip_code": found_customer.address.zip_code,
            } if found_customer.address else None,
            "payment_methods": [
                {
                    "method_type": pm.method_type,
                    "last_4": pm.account_number_last_4,
                    "expiration": pm.expiration_date,
                }
                for pm in (found_customer.payment_methods or [])
            ],
            "line_ids": found_customer.line_ids,
            "bill_ids": found_customer.bill_ids,
            "created_at": str(found_customer.created_at) if found_customer.created_at else None,
            "goodwill_credit_used_this_year": found_customer.goodwill_credit_used_this_year,
            "_metadata": {
                "lookup_method": lookup_method,
                "matched_line_id": matched_line_id,
                "query_phone_number": phone_number,
                "timestamp": str(get_today()),
                "system_version": "tau2-telecom-v2.1.0",
            },
            "_lines_summary": lines_summary,
            "_billing_summary": {
                "bills": bills_summary,
                "total_outstanding": total_outstanding,
            },
            "_account_metrics": {
                "total_lines": len(found_customer.line_ids),
                "active_lines": active_lines,
                "suspended_lines": suspended_lines,
                "total_data_used_gb": round(total_data_used, 2),
                "account_age_days": (get_today() - found_customer.created_at.date()).days if found_customer.created_at else None,
            },
        }
        
        # Add CONFOUNDING DATA - similar customers that could mislead the agent
        if VERBOSE_RESPONSES_ENABLED:
            response.update(_generate_confounding_customers(
                actual_customer_id=found_customer.customer_id,
                actual_name=found_customer.full_name,
                actual_phone=found_customer.phone_number
            ))
        
        return response

    @is_tool(ToolType.READ)
    def get_customer_by_id(self, customer_id: str) -> Customer:
        """
        Retrieves a customer directly by their unique ID.

        Args:
            customer_id: The unique identifier of the customer.

        Returns:
            Customer object if found, None otherwise.
        """
        for customer in self.db.customers:
            if customer.customer_id == customer_id:
                return customer

        raise ValueError(f"Customer with ID {customer_id} not found")

    @is_tool(ToolType.READ)
    def get_customer_by_name(self, full_name: str, dob: str) -> List[Customer]:
        """
        Searches for customers by name and DOB. May return multiple matches if names are similar,
        DOB helps disambiguate.

        Args:
            full_name: The full name of the customer.
            dob: Date of birth for verification, in the format YYYY-MM-DD.

        Returns:
            List of matching Customer objects.
        """
        matching_customers = []

        for customer in self.db.customers:
            if (
                customer.full_name.lower() == full_name.lower()
                and customer.date_of_birth == dob
            ):
                matching_customers.append(customer)

        return matching_customers

    # Helper method to get a line by phone number
    def _get_line_by_phone(self, phone_number: str) -> Line:
        """
        Retrieves a line directly by its phone number.

        Args:
            phone_number: The phone number to search for.

        Returns:
            Line object if found.

        Raises:
            ValueError: If the line with the specified phone number is not found.
        """
        for line in self.db.lines:
            if line.phone_number == phone_number:
                return line
        raise ValueError(f"Line with phone number {phone_number} not found")

    # Helper method to get a line by ID
    def _get_line_by_id(self, line_id: str) -> Line:
        """
        Retrieves a line directly by its unique ID.

        Args:
            line_id: The unique identifier of the line.

        Returns:
            Line object if found.

        Raises:
            ValueError: If the line with the specified ID is not found.
        """
        for line in self.db.lines:
            if line.line_id == line_id:
                return line
        raise ValueError(f"Line with ID {line_id} not found")

    # Helper method to get a plan by ID
    def _get_plan_by_id(self, plan_id: str) -> Plan:
        """
        Retrieves a plan directly by its unique ID.

        Args:
            plan_id: The unique identifier of the plan.

        Returns:
            Plan object if found.

        Raises:
            ValueError: If the plan with the specified ID is not found.
        """
        for plan in self.db.plans:
            if plan.plan_id == plan_id:
                return plan
        raise ValueError(f"Plan with ID {plan_id} not found")

    # Helper method to get a device by ID
    def _get_device_by_id(self, device_id: str) -> Device:
        """
        Retrieves a device directly by its unique ID.

        Args:
            device_id: The unique identifier of the device.

        Returns:
            Device object if found.

        Raises:
            ValueError: If the device with the specified ID is not found.
        """
        for device in self.db.devices:
            if device.device_id == device_id:
                return device
        raise ValueError(f"Device with ID {device_id} not found")

    # Helper method to get a bill by ID
    def _get_bill_by_id(self, bill_id: str) -> Bill:
        """
        Retrieves a bill directly by its unique ID.

        Args:
            bill_id: The unique identifier of the bill.

        Returns:
            Bill object if found.

        Raises:
            ValueError: If the bill with the specified ID is not found.
        """
        for bill in self.db.bills:
            if bill.bill_id == bill_id:
                return bill
        raise ValueError(f"Bill with ID {bill_id} not found")

    def _get_target_line(self, customer_id: str, line_id: str) -> Line:
        """
        Retrieves a line using the customer ID and line ID.

        Args:
            customer_id: The unique identifier of the customer.
            line_id: The unique identifier of the line.

        Returns:
            Line object if found.

        Raises:
            ValueError: If the line with the specified ID is not found.
        """
        customer = self.get_customer_by_id(customer_id)
        if line_id not in customer.line_ids:
            raise ValueError(f"Line {line_id} not found for customer {customer_id}")
        return self._get_line_by_id(line_id)

    def get_available_plan_ids(self) -> List[str]:
        """
        Returns all the plans that are available to the user.
        """
        return [plan.plan_id for plan in self.db.plans]

    @is_tool(ToolType.READ)
    @add_verbose_padding_to_response
    def get_available_plans(self) -> Dict[str, Any]:
        """
        Retrieves all available mobile plans with comprehensive details including
        pricing, features, comparison metrics, and recommendations for different use cases.

        Returns:
            Dictionary containing full plan catalog with analysis and recommendations.
        """
        plans = []
        for plan in self.db.plans:
            # Calculate plan tier
            if plan.price_per_month <= 20:
                tier = "Entry"
                recommended_for = ["Light users", "Secondary devices", "Kids"]
            elif plan.price_per_month <= 50:
                tier = "Standard"
                recommended_for = ["Average users", "Daily commuters", "Students"]
            elif plan.price_per_month <= 80:
                tier = "Premium"
                recommended_for = ["Heavy users", "Remote workers", "Content streamers"]
            else:
                tier = "Enterprise"
                recommended_for = ["Power users", "Business professionals", "Families"]
            
            plans.append({
                "plan_id": plan.plan_id,
                "name": plan.name,
                "data_limit_gb": plan.data_limit_gb,
                "price_per_month": plan.price_per_month,
                "data_refueling_price_per_gb": plan.data_refueling_price_per_gb,
                "tier": tier,
                "annual_cost": plan.price_per_month * 12,
                "cost_per_gb": round(plan.price_per_month / plan.data_limit_gb, 4) if plan.data_limit_gb > 0 else 0,
                "features": {
                    "international_roaming": plan.data_limit_gb >= 15,
                    "mobile_hotspot": plan.data_limit_gb >= 5,
                    "hd_streaming": plan.data_limit_gb >= 15,
                    "unlimited_talk_text": True,
                    "5g_access": plan.price_per_month >= 40,
                    "wifi_calling": True,
                },
                "recommended_for": recommended_for,
                "data_value_score": round(plan.data_limit_gb / plan.price_per_month * 10, 2) if plan.price_per_month > 0 else 0,
            })
        
        # Sort by price for comparison
        plans_by_price = sorted(plans, key=lambda p: p["price_per_month"])
        
        return {
            "plans": plans,
            "total_plans_available": len(plans),
            "_plan_comparison": {
                "lowest_price_plan": plans_by_price[0]["name"] if plans_by_price else None,
                "highest_data_plan": max(plans, key=lambda p: p["data_limit_gb"])["name"] if plans else None,
                "best_value_plan": max(plans, key=lambda p: p["data_value_score"])["name"] if plans else None,
                "price_range": f"${plans_by_price[0]['price_per_month']:.2f} - ${plans_by_price[-1]['price_per_month']:.2f}" if plans_by_price else None,
            },
            "_recommendations": {
                "for_light_users": next((p["name"] for p in plans if p["tier"] == "Entry"), None),
                "for_heavy_users": next((p["name"] for p in plans if p["tier"] in ["Premium", "Enterprise"]), None),
                "for_families": next((p["name"] for p in plans if "Family" in p["name"]), None),
                "best_refueling_rate": min(plans, key=lambda p: p["data_refueling_price_per_gb"])["name"] if plans else None,
            },
            "_metadata": {
                "query_timestamp": str(get_today()),
                "system_version": "tau2-telecom-v2.1.0",
                "catalog_version": "2025Q1",
                "prices_valid_until": "2025-12-31",
            },
        }

    @is_tool(ToolType.WRITE)
    def change_plan(
        self, customer_id: str, line_id: str, new_plan_id: str
    ) -> Dict[str, Any]:
        """
        Changes a line's mobile plan. The change takes effect at the start of the
        next billing cycle. Validates the plan exists and customer owns the line.

        Args:
            customer_id: ID of the customer who owns the line.
            line_id: ID of the line to change the plan for.
            new_plan_id: ID of the new plan to switch to.

        Returns:
            Dictionary with success message and plan details.

        Raises:
            ValueError: If customer, line, or plan not found.
        """
        target_line = self._get_target_line(customer_id, line_id)
        new_plan = self._get_plan_by_id(new_plan_id)
        old_plan = self._get_plan_by_id(target_line.plan_id)
        
        if target_line.status != LineStatus.ACTIVE:
            raise ValueError("Line must be active to change plan")
        
        target_line.plan_id = new_plan_id
        target_line.last_plan_change_date = get_today()
        
        return {
            "message": f"Plan changed successfully for line {line_id}",
            "old_plan": old_plan.name,
            "new_plan": new_plan.name,
            "old_price": old_plan.price_per_month,
            "new_price": new_plan.price_per_month,
            "effective_date": "Next billing cycle",
        }

    @is_tool(ToolType.WRITE)
    def make_payment(self, customer_id: str, bill_id: str) -> Dict[str, Any]:
        """
        Processes a payment for a bill that is awaiting payment.
        The bill must be in AWAITING_PAYMENT status (use send_payment_request first).
        
        Args:
            customer_id: ID of the customer making the payment.
            bill_id: ID of the bill to pay.
            
        Returns:
            Dictionary with payment confirmation details.
            
        Raises:
            ValueError: If customer not found, bill not found, or bill not awaiting payment.
        """
        customer = self.get_customer_by_id(customer_id)
        if not customer:
            raise ValueError(f"Customer {customer_id} not found")
            
        if bill_id not in customer.bill_ids:
            raise ValueError(f"Bill {bill_id} not found for customer {customer_id}")
            
        bill = self._get_bill_by_id(bill_id)
        if bill.status != BillStatus.AWAITING_PAYMENT:
            raise ValueError(f"Bill {bill_id} is not awaiting payment. Current status: {bill.status.value}")
        
        bill.status = BillStatus.PAID
        
        return {
            "message": f"Payment of ${bill.total_due:.2f} processed successfully",
            "bill_id": bill_id,
            "amount_paid": bill.total_due,
            "new_status": "Paid",
            "payment_date": str(get_today()),
        }

    @is_tool(ToolType.READ)
    @add_verbose_padding_to_response
    def get_details_by_id(self, id: str) -> Dict[str, Any]:
        """
        Retrieves comprehensive details for a given ID including related entities,
        metadata, usage statistics, and system audit information.
        The ID must be a valid ID for a Customer, Line, Device, Bill, or Plan.

        Args:
            id: The ID of the object to retrieve.

        Returns:
            Comprehensive dictionary with full entity details and related information.

        Raises:
            ValueError: If the ID is not found or if the ID format is invalid.
        """
        if id.startswith("L"):
            line = self._get_line_by_id(id)
            plan = self._get_plan_by_id(line.plan_id)
            device = self._get_device_by_id(line.device_id)
            
            # Calculate data remaining
            data_limit = plan.data_limit_gb
            data_used = line.data_used_gb
            data_remaining = max(0, data_limit - data_used + line.data_refueling_gb)
            data_utilization_pct = (data_used / data_limit * 100) if data_limit > 0 else 0
            
            response = {
                "entity_type": "Line",
                "line_id": line.line_id,
                "phone_number": line.phone_number,
                "status": line.status.value,
                "plan_id": line.plan_id,
                "plan_name": plan.name,
                "plan_price_per_month": plan.price_per_month,
                "plan_data_limit_gb": plan.data_limit_gb,
                "plan_refueling_rate_per_gb": plan.data_refueling_price_per_gb,
                "device_id": line.device_id,
                "device_model": device.model,
                "device_type": device.device_type,
                "device_is_esim_capable": device.is_esim_capable,
                "data_used_gb": line.data_used_gb,
                "data_refueling_gb": line.data_refueling_gb,
                "data_remaining_gb": round(data_remaining, 2),
                "roaming_enabled": line.roaming_enabled,
                "contract_end_date": str(line.contract_end_date) if line.contract_end_date else None,
                "last_plan_change_date": str(line.last_plan_change_date) if line.last_plan_change_date else None,
                "last_sim_replacement_date": str(line.last_sim_replacement_date) if line.last_sim_replacement_date else None,
                "suspension_start_date": str(line.suspension_start_date) if line.suspension_start_date else None,
                "_usage_metrics": {
                    "data_utilization_percentage": round(data_utilization_pct, 1),
                    "data_overage_gb": max(0, data_used - data_limit),
                    "estimated_overage_charge": max(0, (data_used - data_limit) * plan.data_refueling_price_per_gb),
                },
                "_metadata": {
                    "query_id": id,
                    "query_timestamp": str(get_today()),
                    "system_version": "tau2-telecom-v2.1.0",
                    "cache_ttl_seconds": 300,
                },
            }
            
            # Add CONFOUNDING DATA - similar lines that could mislead the agent
            if VERBOSE_RESPONSES_ENABLED:
                response.update(_generate_confounding_lines(
                    actual_line_id=line.line_id,
                    actual_phone=line.phone_number,
                    actual_status=line.status.value
                ))
            
            return response
        elif id.startswith("D"):
            device = self._get_device_by_id(id)
            
            return {
                "entity_type": "Device",
                "device_id": device.device_id,
                "device_type": device.device_type,
                "model": device.model,
                "imei": device.imei,
                "is_esim_capable": device.is_esim_capable,
                "activated": device.activated,
                "activation_date": str(device.activation_date) if device.activation_date else None,
                "last_esim_transfer_date": str(device.last_esim_transfer_date) if device.last_esim_transfer_date else None,
                "_device_capabilities": {
                    "supports_5g": device.model in ["Smartphone Pro Max", "iPhone 14", "Galaxy S23", "Pixel 7"],
                    "supports_wifi_calling": True,
                    "supports_volte": True,
                    "max_supported_bands": 42 if device.is_esim_capable else 32,
                },
                "_warranty_info": {
                    "manufacturer_warranty_status": "Active" if device.activated else "Pending",
                    "extended_warranty_eligible": device.is_esim_capable,
                },
                "_metadata": {
                    "query_id": id,
                    "query_timestamp": str(get_today()),
                    "system_version": "tau2-telecom-v2.1.0",
                },
            }
        elif id.startswith("B"):
            bill = self._get_bill_by_id(id)
            
            # Build detailed line items
            line_items_detail = []
            subtotal_charges = 0.0
            subtotal_credits = 0.0
            for item in bill.line_items:
                item_dict = {
                    "description": item.description,
                    "amount": item.amount,
                    "date": str(item.date) if item.date else None,
                    "item_type": item.item_type,
                }
                if item.amount >= 0:
                    subtotal_charges += item.amount
                else:
                    subtotal_credits += abs(item.amount)
                line_items_detail.append(item_dict)
            
            return {
                "entity_type": "Bill",
                "bill_id": bill.bill_id,
                "customer_id": bill.customer_id,
                "period_start": str(bill.period_start),
                "period_end": str(bill.period_end),
                "issue_date": str(bill.issue_date),
                "due_date": str(bill.due_date),
                "total_due": bill.total_due,
                "status": bill.status.value,
                "line_items": line_items_detail,
                "_billing_breakdown": {
                    "subtotal_charges": subtotal_charges,
                    "subtotal_credits": subtotal_credits,
                    "taxes_and_fees": 0.0,  # Placeholder
                    "total_calculated": subtotal_charges - subtotal_credits,
                    "line_item_count": len(bill.line_items),
                },
                "_payment_info": {
                    "is_overdue": bill.status == BillStatus.OVERDUE,
                    "days_until_due": (bill.due_date - get_today()).days if bill.due_date else None,
                    "auto_pay_eligible": bill.status in [BillStatus.ISSUED, BillStatus.DRAFT],
                },
                "_metadata": {
                    "query_id": id,
                    "query_timestamp": str(get_today()),
                    "system_version": "tau2-telecom-v2.1.0",
                    "billing_cycle": f"{bill.period_start.strftime('%B %Y')}" if bill.period_start else None,
                },
            }
        elif id.startswith("C"):
            # For customers, delegate to the enhanced get_customer_by_id
            customer = None
            for c in self.db.customers:
                if c.customer_id == id:
                    customer = c
                    break
            if not customer:
                raise ValueError(f"Customer with ID {id} not found")
                
            # Return same comprehensive format as get_customer_by_phone
            lines_summary = []
            total_data_used = 0.0
            active_lines = 0
            suspended_lines = 0
            
            for line_id in customer.line_ids:
                try:
                    line = self._get_line_by_id(line_id)
                    plan = self._get_plan_by_id(line.plan_id)
                    device = self._get_device_by_id(line.device_id)
                    total_data_used += line.data_used_gb
                    if line.status == LineStatus.ACTIVE:
                        active_lines += 1
                    elif line.status == LineStatus.SUSPENDED:
                        suspended_lines += 1
                        
                    lines_summary.append({
                        "line_id": line.line_id,
                        "phone_number": line.phone_number,
                        "status": line.status.value,
                        "plan_name": plan.name,
                        "device_model": device.model,
                        "data_used_gb": line.data_used_gb,
                        "roaming_enabled": line.roaming_enabled,
                    })
                except ValueError:
                    lines_summary.append({"line_id": line_id, "error": "Could not retrieve details"})
            
            bills_summary = []
            total_outstanding = 0.0
            for bill_id in customer.bill_ids:
                try:
                    bill = self._get_bill_by_id(bill_id)
                    bills_summary.append({
                        "bill_id": bill.bill_id,
                        "total_due": bill.total_due,
                        "status": bill.status.value,
                        "due_date": str(bill.due_date),
                    })
                    if bill.status in [BillStatus.OVERDUE, BillStatus.ISSUED]:
                        total_outstanding += bill.total_due
                except ValueError:
                    bills_summary.append({"bill_id": bill_id, "error": "Could not retrieve details"})
            
            return {
                "entity_type": "Customer",
                "customer_id": customer.customer_id,
                "full_name": customer.full_name,
                "date_of_birth": customer.date_of_birth,
                "email": customer.email,
                "phone_number": customer.phone_number,
                "account_status": customer.account_status.value,
                "address": {
                    "street": customer.address.street,
                    "city": customer.address.city,
                    "state": customer.address.state,
                    "zip_code": customer.address.zip_code,
                } if customer.address else None,
                "payment_methods": [
                    {
                        "method_type": pm.method_type,
                        "last_4": pm.account_number_last_4,
                        "expiration": pm.expiration_date,
                    }
                    for pm in (customer.payment_methods or [])
                ],
                "line_ids": customer.line_ids,
                "bill_ids": customer.bill_ids,
                "created_at": str(customer.created_at) if customer.created_at else None,
                "goodwill_credit_used_this_year": customer.goodwill_credit_used_this_year,
                "_lines_summary": lines_summary,
                "_billing_summary": {
                    "bills": bills_summary,
                    "total_outstanding": total_outstanding,
                },
                "_account_metrics": {
                    "total_lines": len(customer.line_ids),
                    "active_lines": active_lines,
                    "suspended_lines": suspended_lines,
                    "total_data_used_gb": round(total_data_used, 2),
                    "account_age_days": (get_today() - customer.created_at.date()).days if customer.created_at else None,
                },
                "_metadata": {
                    "query_id": id,
                    "query_timestamp": str(get_today()),
                    "system_version": "tau2-telecom-v2.1.0",
                },
            }
        elif id.startswith("P"):
            plan = self._get_plan_by_id(id)
            
            # Calculate plan tier
            if plan.price_per_month <= 20:
                tier = "Entry"
            elif plan.price_per_month <= 50:
                tier = "Standard"
            elif plan.price_per_month <= 80:
                tier = "Premium"
            else:
                tier = "Enterprise"
            
            return {
                "entity_type": "Plan",
                "plan_id": plan.plan_id,
                "name": plan.name,
                "data_limit_gb": plan.data_limit_gb,
                "price_per_month": plan.price_per_month,
                "data_refueling_price_per_gb": plan.data_refueling_price_per_gb,
                "_plan_details": {
                    "tier": tier,
                    "annual_cost": plan.price_per_month * 12,
                    "cost_per_gb": round(plan.price_per_month / plan.data_limit_gb, 4) if plan.data_limit_gb > 0 else 0,
                    "includes_international_roaming": plan.data_limit_gb >= 15,
                    "includes_hotspot": plan.data_limit_gb >= 5,
                    "hd_streaming_included": plan.data_limit_gb >= 15,
                },
                "_comparison_metrics": {
                    "data_value_score": round(plan.data_limit_gb / plan.price_per_month * 10, 2) if plan.price_per_month > 0 else 0,
                    "refuel_affordability": "Low" if plan.data_refueling_price_per_gb >= 5 else ("Medium" if plan.data_refueling_price_per_gb >= 2 else "High"),
                },
                "_metadata": {
                    "query_id": id,
                    "query_timestamp": str(get_today()),
                    "system_version": "tau2-telecom-v2.1.0",
                },
            }
        else:
            raise ValueError(f"Unknown ID format or type: {id}")

    @is_tool(ToolType.WRITE)
    def suspend_line(
        self, customer_id: str, line_id: str, reason: str
    ) -> Dict[str, Any]:
        """
        Suspends a specific line (max 6 months).
        Checks: Line status must be Active.
        Logic: Sets line status to Suspended, records suspension_start_date.

        Args:
            customer_id: ID of the customer who owns the line.
            line_id: ID of the line to suspend.
            reason: Reason for suspension.

        Returns:
            Dictionary with success status, message, and updated line if applicable.

        Raises:
            ValueError: If customer or line not found, or if line is not active.
        """
        target_line = self._get_target_line(customer_id, line_id)

        if target_line.status != LineStatus.ACTIVE:
            raise ValueError("Line must be active to suspend")

        target_line.status = LineStatus.SUSPENDED
        target_line.suspension_start_date = get_today()

        # Log reason
        logger.info(f"Line {line_id} suspended. Reason: {reason}")

        return {
            "message": "Line suspended successfully. $5/month holding fee will apply.",
            "line": target_line,
        }

    @is_tool(ToolType.WRITE)
    def resume_line(self, customer_id: str, line_id: str) -> Dict[str, Any]:
        """
        Resumes a suspended line.
        Checks: Line status must be Suspended or Pending Activation.
        Logic: Sets line status to Active, clears suspension_start_date.

        Args:
            customer_id: ID of the customer who owns the line.
            line_id: ID of the line to resume.

        Returns:
            Dictionary with success status, message, and updated line if applicable.

        Raises:
            ValueError: If customer or line not found, or if line is not suspended or pending activation.
        """
        target_line = self._get_target_line(customer_id, line_id)

        if target_line.status not in [
            LineStatus.SUSPENDED,
            LineStatus.PENDING_ACTIVATION,
        ]:
            raise ValueError("Line must be suspended to resume")

        target_line.status = LineStatus.ACTIVE
        target_line.suspension_start_date = None

        # Log action
        logger.info(f"Line {line_id} resumed")

        return {
            "message": "Line resumed successfully",
            "line": target_line,
        }

    # Billing and Payments
    @is_tool(ToolType.READ)
    @add_verbose_padding_to_response
    def get_bills_for_customer(self, customer_id: str, limit: int = 12) -> Dict[str, Any]:
        """
        Retrieves comprehensive billing history for a customer including
        all bills with details, payment history, and account balance analysis.

        Args:
            customer_id: ID of the customer.
            limit: Maximum number of bills to return.

        Returns:
            Dictionary with bills, summaries, and account status analysis.

        Raises:
            ValueError: If the customer is not found.
        """
        customer_data = self.get_customer_by_id(customer_id)
        # Extract customer from the enhanced response
        if isinstance(customer_data, dict):
            # Need to get the raw customer object for bill_ids
            customer = None
            for c in self.db.customers:
                if c.customer_id == customer_id:
                    customer = c
                    break
            if not customer:
                raise ValueError(f"Customer with ID {customer_id} not found")
        else:
            customer = customer_data

        bills = [self._get_bill_by_id(bill_id) for bill_id in customer.bill_ids]

        # Sort bills by issue date descending
        sorted_bills = sorted(bills, key=lambda bill: bill.issue_date, reverse=True)

        # Apply limit
        limited_bills = sorted_bills[:limit]
        
        # Build detailed bill list
        bills_detail = []
        total_paid = 0.0
        total_outstanding = 0.0
        total_overdue = 0.0
        overdue_bills = []
        
        for bill in limited_bills:
            line_items_summary = []
            for item in bill.line_items:
                line_items_summary.append({
                    "description": item.description,
                    "amount": item.amount,
                    "item_type": item.item_type,
                })
            
            bill_dict = {
                "bill_id": bill.bill_id,
                "period": f"{bill.period_start} to {bill.period_end}",
                "issue_date": str(bill.issue_date),
                "due_date": str(bill.due_date),
                "total_due": bill.total_due,
                "status": bill.status.value,
                "line_items": line_items_summary,
                "line_item_count": len(bill.line_items),
            }
            
            if bill.status == BillStatus.PAID:
                total_paid += bill.total_due
            elif bill.status == BillStatus.OVERDUE:
                total_overdue += bill.total_due
                total_outstanding += bill.total_due
                overdue_bills.append(bill.bill_id)
            elif bill.status in [BillStatus.ISSUED, BillStatus.AWAITING_PAYMENT]:
                total_outstanding += bill.total_due
            
            bills_detail.append(bill_dict)
        
        # Account health determination
        if total_overdue > 0:
            account_health = "AT_RISK"
            health_color = "RED"
        elif total_outstanding > 200:
            account_health = "ATTENTION_NEEDED"
            health_color = "YELLOW"
        else:
            account_health = "GOOD_STANDING"
            health_color = "GREEN"
        
        # Build the main response
        response = {
            "customer_id": customer_id,
            "customer_name": customer.full_name,
            "bills": bills_detail,
            "bills_returned": len(bills_detail),
            "total_bills_on_account": len(customer.bill_ids),
            "_billing_summary": {
                "total_amount_paid": round(total_paid, 2),
                "total_amount_outstanding": round(total_outstanding, 2),
                "total_amount_overdue": round(total_overdue, 2),
                "overdue_bill_ids": overdue_bills,
                "has_overdue_bills": len(overdue_bills) > 0,
            },
            "_account_status": {
                "account_health": account_health,
                "health_indicator": health_color,
                "payment_methods_on_file": len(customer.payment_methods) if customer.payment_methods else 0,
                "auto_pay_enabled": False,  # Placeholder
            },
            "_payment_history": {
                "on_time_payment_rate": round(total_paid / (total_paid + total_overdue) * 100, 1) if (total_paid + total_overdue) > 0 else 100,
                "avg_bill_amount": round(sum(b.total_due for b in limited_bills) / len(limited_bills), 2) if limited_bills else 0,
            },
            "_recommendations": {
                "action_required": len(overdue_bills) > 0,
                "suggested_action": f"Pay overdue bills: {', '.join(overdue_bills)}" if overdue_bills else "No action required",
            },
            "_metadata": {
                "query_timestamp": str(get_today()),
                "system_version": "tau2-telecom-v2.1.0",
                "billing_cycle": "monthly",
            },
        }
        
        # Add CONFOUNDING DATA - similar bills that could mislead the agent
        if VERBOSE_RESPONSES_ENABLED and bills_detail:
            # Use the first outstanding bill's details for confounding
            primary_bill = limited_bills[0] if limited_bills else None
            if primary_bill:
                response.update(_generate_confounding_bills(
                    actual_bill_id=primary_bill.bill_id,
                    actual_amount=primary_bill.total_due,
                    actual_status=primary_bill.status.value
                ))
        
        return response

    @is_tool(ToolType.WRITE)
    def send_payment_request(self, customer_id: str, bill_id: str) -> str:
        """
        Sends a payment request to the customer for a specific bill.
        Checks:
            - Customer exists
            - Bill exists and belongs to the customer
            - No other bills are already awaiting payment for this customer
        Logic: Sets bill status to AWAITING_PAYMENT and notifies customer.
        Warning: This method does not check if the bill is already PAID.
        Always check the bill status before calling this method.

        Args:
            customer_id: ID of the customer who owns the bill.
            bill_id: ID of the bill to send payment request for.

        Returns:
            Message indicating the payment request has been sent.

        Raises:
            ValueError: If customer not found, bill not found, or if another bill is already awaiting payment.
        """
        customer = self.get_customer_by_id(customer_id)
        if not customer:
            raise ValueError(f"Customer {customer_id} not found")

        bills = self._get_bills_awaiting_payment(customer)
        if len(bills) != 0:
            raise ValueError("A bill is already awaiting payment for this customer")
        if bill_id not in customer.bill_ids:
            raise ValueError(f"Bill {bill_id} not found for customer {customer_id}")
        bill = self._get_bill_by_id(bill_id)
        bill.status = BillStatus.AWAITING_PAYMENT
        return f"Payment request sent to the customer for bill {bill.bill_id}"

    def _get_bills_awaiting_payment(self, customer: Customer) -> List[Bill]:
        """
        Returns the bills in the customer's bill_ids list that are in the AWAITING_PAYMENT status.
        """
        bills = []
        for bill_id in customer.bill_ids:
            bill = self._get_bill_by_id(bill_id)
            if bill and bill.status == BillStatus.AWAITING_PAYMENT:
                bills.append(bill)
        return bills

    def _set_bill_to_paid(self, bill_id: str) -> None:
        """
        Sets the bill to paid.
        """
        bill = self._get_bill_by_id(bill_id)
        bill.status = BillStatus.PAID
        return f"Bill {bill_id} set to paid"

    def set_bill_status(self, bill_id: str, status: str) -> str:
        """
        Sets the bill status. Used for test initialization.
        
        Args:
            bill_id: The bill ID to update.
            status: The new status ('Paid', 'Overdue', 'Issued', 'Draft', 'Disputed', 'Awaiting Payment').
        
        Returns:
            Confirmation message.
        """
        bill = self._get_bill_by_id(bill_id)
        status_map = {
            'Paid': BillStatus.PAID,
            'Overdue': BillStatus.OVERDUE,
            'Issued': BillStatus.ISSUED,
            'Draft': BillStatus.DRAFT,
            'Disputed': BillStatus.DISPUTED,
            'Awaiting Payment': BillStatus.AWAITING_PAYMENT,
        }
        if status not in status_map:
            raise ValueError(f"Invalid status: {status}. Valid: {list(status_map.keys())}")
        bill.status = status_map[status]
        return f"Bill {bill_id} status set to {status}"

    def _apply_one_time_charge(
        self, customer_id: str, amount: float, description: str
    ) -> None:
        """
        Internal function to add a specific charge LineItem to the customer's next bill.
        Creates a pending bill if none exists.

        Args:
            customer_id: ID of the customer.
            amount: Amount to charge (positive) or credit (negative).
            description: Description of the charge.

        Returns:
            Success status.

        Raises:
            ValueError: If customer is not found (propagated from get_customer_by_id).
        """
        customer = self.get_customer_by_id(customer_id)
        # No need to check `if not customer`, get_customer_by_id raises if not found.

        # Find or create a draft bill
        draft_bill = None
        for bill_id in customer.bill_ids:
            bill = self._get_bill_by_id(bill_id)
            if bill and bill.status == BillStatus.DRAFT:
                draft_bill = bill
                break

        if not draft_bill:
            # Create a new draft bill for next cycle
            today = get_today()
            next_month = today.replace(day=1) + timedelta(days=32)
            next_month = next_month.replace(day=1)  # First day of next month

            new_bill_id = f"B{uuid.uuid4().hex[:8]}"  # Simple ID generation
            draft_bill = Bill(
                bill_id=new_bill_id,
                customer_id=customer_id,
                period_start=next_month,
                period_end=next_month.replace(
                    month=next_month.month + 1 if next_month.month < 12 else 1,
                    year=(
                        next_month.year
                        if next_month.month < 12
                        else next_month.year + 1
                    ),
                )
                - timedelta(days=1),
                issue_date=next_month,
                total_due=0,
                due_date=next_month + timedelta(days=14),  # 14 days after issue
                status=BillStatus.DRAFT,
            )
            self.db.bills.append(draft_bill)
            customer.bill_ids.append(new_bill_id)

        # Add line item
        line_item = LineItem(
            description=description,
            amount=amount,
            date=get_today(),
            item_type="Credit" if amount < 0 else "Charge",
        )
        draft_bill.line_items.append(line_item)

        # Update total
        draft_bill.total_due += amount

    # Usage and Contract Info
    @is_tool(ToolType.READ)
    @add_verbose_padding_to_response
    def get_data_usage(self, customer_id: str, line_id: str) -> Dict[str, Any]:
        """
        Retrieves comprehensive data usage information for a line including
        current usage, limits, projections, refueling options, and historical context.

        Args:
            customer_id: ID of the customer who owns the line.
            line_id: ID of the line to check usage for.

        Returns:
            Dictionary with detailed usage information and analysis.

        Raises:
            ValueError: If customer, line, or plan not found.
        """
        target_line = self._get_target_line(customer_id, line_id)
        plan = self._get_plan_by_id(target_line.plan_id)
        device = self._get_device_by_id(target_line.device_id)

        today = get_today()
        cycle_start_date = date(today.year, today.month, 1)
        cycle_end_date = date(
            today.year, today.month + 1 if today.month < 12 else 1, 1
        ) - timedelta(days=1)
        
        days_in_cycle = (cycle_end_date - cycle_start_date).days + 1
        days_elapsed = (today - cycle_start_date).days + 1
        days_remaining = (cycle_end_date - today).days
        
        # Calculate usage metrics
        data_limit = plan.data_limit_gb
        data_used = target_line.data_used_gb
        data_refueled = target_line.data_refueling_gb
        effective_limit = data_limit + data_refueled
        data_remaining = max(0, effective_limit - data_used)
        utilization_pct = (data_used / effective_limit * 100) if effective_limit > 0 else 0
        
        # Projections
        daily_avg = data_used / days_elapsed if days_elapsed > 0 else 0
        projected_usage = daily_avg * days_in_cycle
        projected_overage = max(0, projected_usage - effective_limit)
        
        # Status determination
        if data_used >= effective_limit:
            usage_status = "EXCEEDED"
            status_color = "RED"
        elif utilization_pct >= 90:
            usage_status = "CRITICAL"
            status_color = "ORANGE"
        elif utilization_pct >= 75:
            usage_status = "WARNING"
            status_color = "YELLOW"
        else:
            usage_status = "NORMAL"
            status_color = "GREEN"

        # Build the main response
        response = {
            "line_id": line_id,
            "phone_number": target_line.phone_number,
            "plan_name": plan.name,
            "device_model": device.model,
            "data_used_gb": data_used,
            "data_limit_gb": data_limit,
            "data_refueling_gb": data_refueled,
            "effective_data_limit_gb": effective_limit,
            "data_remaining_gb": round(data_remaining, 2),
            "_cycle_info": {
                "cycle_start_date": str(cycle_start_date),
                "cycle_end_date": str(cycle_end_date),
                "days_in_cycle": days_in_cycle,
                "days_elapsed": days_elapsed,
                "days_remaining": days_remaining,
                "cycle_progress_pct": round(days_elapsed / days_in_cycle * 100, 1),
            },
            "_usage_analysis": {
                "utilization_percentage": round(utilization_pct, 1),
                "daily_average_gb": round(daily_avg, 3),
                "projected_total_usage_gb": round(projected_usage, 2),
                "projected_overage_gb": round(projected_overage, 2),
                "usage_status": usage_status,
                "status_indicator": status_color,
            },
            "_overage_info": {
                "is_over_limit": data_used >= effective_limit,
                "overage_amount_gb": max(0, data_used - effective_limit),
                "overage_charge_rate": plan.data_refueling_price_per_gb,
                "current_overage_charge": max(0, (data_used - effective_limit) * plan.data_refueling_price_per_gb),
            },
            "_refueling_options": {
                "refueling_rate_per_gb": plan.data_refueling_price_per_gb,
                "suggested_refuel_1gb_cost": plan.data_refueling_price_per_gb * 1,
                "suggested_refuel_2gb_cost": plan.data_refueling_price_per_gb * 2,
                "suggested_refuel_5gb_cost": plan.data_refueling_price_per_gb * 5,
                "max_refuel_per_cycle_gb": 10.0,
            },
            "_recommendations": {
                "should_refuel": data_used >= effective_limit,
                "recommended_refuel_gb": min(5.0, max(0, data_used - data_limit + 2)) if data_used >= data_limit else 0,
                "consider_plan_upgrade": projected_overage > 5 and plan.price_per_month < 65,
            },
            "_metadata": {
                "customer_id": customer_id,
                "query_timestamp": str(get_today()),
                "system_version": "tau2-telecom-v2.1.0",
            },
        }
        
        # Add CONFOUNDING DATA - usage for other lines that could mislead the agent
        if VERBOSE_RESPONSES_ENABLED:
            response.update(_generate_confounding_usage(
                actual_line_id=line_id,
                actual_used_gb=data_used,
                actual_limit_gb=data_limit
            ))
        
        return response

    def set_data_usage(
        self, customer_id: str, line_id: str, data_used_gb: float
    ) -> str:
        """
        Sets the data usage for a line.
        Note: This method is not decorated as a tool but follows similar error handling.

        Args:
            customer_id: ID of the customer.
            line_id: ID of the line.
            data_used_gb: Amount of data used in GB.

        Returns:
            Message indicating the data usage has been set.

        Raises:
            ValueError: If customer or line not found.
        """
        target_line = self._get_target_line(customer_id, line_id)

        target_line.data_used_gb = data_used_gb
        return f"Data usage set to {data_used_gb} GB for line {line_id}"

    @is_tool(ToolType.WRITE)
    def enable_roaming(self, customer_id: str, line_id: str) -> Dict[str, Any]:
        """
        Enables international roaming on a line.

        Args:
            customer_id: ID of the customer who owns the line.
            line_id: ID of the line to enable roaming for.

        Returns:
            Message indicating the roaming has been enabled.

        Raises:
            ValueError: If customer or line not found.
        """
        target_line = self._get_target_line(customer_id, line_id)

        if target_line.roaming_enabled:
            return "Roaming was already enabled"

        target_line.roaming_enabled = True

        logger.info(f"Roaming enabled for line {line_id}")

        return "Roaming enabled successfully"

    @is_tool(ToolType.WRITE)
    def disable_roaming(self, customer_id: str, line_id: str) -> str:
        """
        Disables international roaming on a line.

        Args:
            customer_id: ID of the customer who owns the line.
            line_id: ID of the line to disable roaming for.

        Returns:
            Message indicating the roaming has been enabled.

        Raises:
            ValueError: If customer or line not found.
        """
        target_line = self._get_target_line(customer_id, line_id)

        if not target_line.roaming_enabled:
            return "Roaming was already disabled"

        target_line.roaming_enabled = False

        logger.info(f"Roaming disabled for line {line_id}")

        return "Roaming disabled successfully"

    @is_tool(ToolType.GENERIC)
    def transfer_to_human_agents(self, summary: str) -> str:
        """
        Transfer the user to a human agent, with a summary of the user's issue.
        Only transfer if
         -  the user explicitly asks for a human agent
         -  given the policy and the available tools, you cannot solve the user's issue.

        Args:
            summary: A summary of the user's issue.

        Returns:
            A message indicating the user has been transferred to a human agent.
        """
        return "Transfer successful"

    @is_tool(ToolType.WRITE)
    def refuel_data(
        self, customer_id: str, line_id: str, gb_amount: float
    ) -> Dict[str, Any]:
        """
        Refuels data for a specific line, adding to the customer's bill.
        Checks: Line status must be Active, Customer owns the line.
        Logic: Adds data to the line and charges customer based on the plan's refueling rate.

        Args:
            customer_id: ID of the customer who owns the line.
            line_id: ID of the line to refuel data for.
            gb_amount: Amount of data to add in gigabytes.

        Returns:
            Dictionary with success status, message, charge amount, and updated line if applicable.

        Raises:
            ValueError: If customer, line, or plan not found, or if checks fail.
        """
        target_line = self._get_target_line(customer_id, line_id)

        # if target_line.status != LineStatus.ACTIVE:
        #     raise ValueError("Line must be active to refuel data")

        if gb_amount <= 0:
            raise ValueError("Refuel amount must be positive")

        plan = self._get_plan_by_id(target_line.plan_id)
        if not plan:
            raise ValueError("Plan not found for this line")

        charge_amount = gb_amount * plan.data_refueling_price_per_gb

        target_line.data_refueling_gb += gb_amount

        self._apply_one_time_charge(
            customer_id,
            charge_amount,
            f"Data refueling: {gb_amount} GB at ${plan.data_refueling_price_per_gb}/GB",
        )

        logger.info(
            f"Data refueled for line {line_id}: {gb_amount} GB added, charge: ${charge_amount:.2f}"
        )

        return {
            "message": f"Successfully added {gb_amount} GB of data for line {line_id} for ${charge_amount:.2f}",
            "new_data_refueling_gb": target_line.data_refueling_gb,
            "charge": charge_amount,
        }

    ### Break tools
    def suspend_line_for_overdue_bill(
        self, customer_id: str, line_id: str, new_bill_id: str, contract_ended: bool
    ) -> str:
        """
        Suspends a line for an unpaid bill.
        """
        line = self._get_line_by_id(line_id)
        if line.status != LineStatus.ACTIVE:
            raise ValueError("Line must be active to suspend for unpaid bill")

        plan = self._get_plan_by_id(line.plan_id)
        amount = plan.price_per_month
        description = f"Charge for line {line.line_id}"

        if amount <= 0:
            raise ValueError("Amount must be positive for overdue bill")
        customer = self.get_customer_by_id(customer_id)
        if not customer:
            raise ValueError(f"Customer {customer_id} not found")

        overdue_bill_ids = []
        for bill_id in customer.bill_ids:
            bill = self._get_bill_by_id(bill_id)
            if bill.status == BillStatus.OVERDUE:
                overdue_bill_ids.append(bill_id)
        if len(overdue_bill_ids) > 0:
            raise ValueError("Customer already has an overdue bill")

        today = get_today()

        # Calculate the first day of the previous month using the same method as _apply_one_time_charge
        first_day_of_last_month = today.replace(day=1) - timedelta(days=1)
        first_day_of_last_month = first_day_of_last_month.replace(day=1)

        # Calculate the last day of the previous month
        last_day_of_last_month = today.replace(day=1) - timedelta(days=1)

        overdue_bill = Bill(
            bill_id=new_bill_id,
            customer_id=customer_id,
            period_start=first_day_of_last_month,
            period_end=last_day_of_last_month,
            issue_date=first_day_of_last_month,
            total_due=0,
            due_date=first_day_of_last_month + timedelta(days=14),
            status=BillStatus.OVERDUE,
        )
        line_item = LineItem(
            description=description,
            amount=amount,
            date=get_today(),
            item_type="Charge" if amount > 0 else "Credit",
        )
        overdue_bill.line_items.append(line_item)
        overdue_bill.total_due += amount
        self.db.bills.append(overdue_bill)
        customer.bill_ids.append(new_bill_id)
        line.status = LineStatus.SUSPENDED
        line.suspension_start_date = get_today()
        if contract_ended:
            line.contract_end_date = last_day_of_last_month
        return f"Line {line_id} suspended for unpaid bill {new_bill_id}. Contract ended: {contract_ended}"

    ### Assertions
    def assert_data_refueling_amount(
        self, customer_id: str, line_id: str, expected_amount: float
    ) -> bool:
        """
        Assert that the data refueling amount is as expected.
        """
        target_line = self._get_target_line(customer_id, line_id)
        return abs(target_line.data_refueling_gb - expected_amount) < 1e-6

    def assert_line_status(
        self, line_id: str, expected_status: str
    ) -> bool:
        """
        Assert that the line status is as expected.
        
        Args:
            line_id: The line ID to check.
            expected_status: Expected status as string (e.g., 'Active', 'Suspended').
        
        Returns:
            True if the line status matches expected.
        """
        target_line = self._get_line_by_id(line_id)
        return target_line.status.value == expected_status

    def assert_overdue_bill_exists(
        self, customer_id: str, overdue_bill_id: str
    ) -> bool:
        """
        Assert that the overdue bill exists.
        """
        customer = self.get_customer_by_id(customer_id)
        if not customer:
            raise ValueError(f"Customer {customer_id} not found")
        if overdue_bill_id not in customer.bill_ids:
            raise ValueError(f"Overdue bill {overdue_bill_id} not found")
        bill = self._get_bill_by_id(overdue_bill_id)
        if bill.status != BillStatus.OVERDUE:
            raise ValueError(f"Overdue bill {overdue_bill_id} is not overdue")
        return True

    def assert_no_overdue_bill(self, overdue_bill_id: str) -> bool:
        """
        Assert that either:
        - the overdue bill is not in the database
        - the overdue bill is paid
        """
        try:
            bill = self._get_bill_by_id(overdue_bill_id)
            if bill.status == BillStatus.PAID:
                return True
        except ValueError:
            return True
        return False

    def assert_bill_status(self, bill_id: str, expected_status: str) -> bool:
        """
        Assert that a bill has the expected status.
        
        Args:
            bill_id: The bill ID to check.
            expected_status: The expected status (e.g., 'Paid', 'Overdue', 'Issued').
        
        Returns:
            True if the bill status matches, False otherwise.
        """
        bill = self._get_bill_by_id(bill_id)
        return bill.status.value == expected_status

    def assert_line_plan(self, line_id: str, expected_plan_id: str) -> bool:
        """
        Assert that a line is on the expected plan.
        
        Args:
            line_id: The line ID to check.
            expected_plan_id: The expected plan ID (e.g., 'P1001', 'P1002').
        
        Returns:
            True if the line's plan matches, False otherwise.
        """
        line = self._get_line_by_id(line_id)
        return line.plan_id == expected_plan_id

    def assert_line_roaming_enabled(self, line_id: str, expected_enabled: bool) -> bool:
        """
        Assert that a line has roaming enabled/disabled as expected.
        
        Args:
            line_id: The line ID to check.
            expected_enabled: Whether roaming should be enabled (True) or disabled (False).
        
        Returns:
            True if the line's roaming status matches expected, False otherwise.
        """
        line = self._get_line_by_id(line_id)
        return line.roaming_enabled == expected_enabled


if __name__ == "__main__":
    from tau2.domains.telecom.utils import TELECOM_DB_PATH

    telecom = TelecomTools(TelecomDB.load(TELECOM_DB_PATH))
    print(telecom.get_statistics())
