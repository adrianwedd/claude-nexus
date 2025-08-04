#!/usr/bin/env python3
"""
Enterprise Monitoring and Alerting System

Comprehensive monitoring solution for multi-tenant claude-nexus agent
ecosystem with real-time metrics, SLA tracking, alerting, and
performance analytics. Provides enterprise-grade observability
with tenant-specific dashboards and compliance reporting.

Features:
- Real-time performance monitoring and metrics collection
- SLA tracking and violation detection with automated alerting
- Tenant-specific dashboards and reporting
- Health scoring and predictive failure detection
- Integration with popular monitoring tools (Prometheus, Grafana)
- Distributed tracing and error tracking
- Custom alerting rules and escalation policies
- Performance analytics and optimization recommendations

Metrics Collected:
- Request latency and throughput
- Error rates and failure patterns
- Resource utilization (CPU, memory, network)
- Agent performance and availability
- Security events and threat detection
- Compliance and audit metrics

Author: Fortress Guardian
Version: 1.0.0
Compliance: SOC 2 Type II, Enterprise SLA
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
from collections import defaultdict, deque
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import socket
import psutil
import platform

# Import security and orchestration systems
from enterprise_security_architecture import (
    AuditEvent, AuditEventType, SecurityLevel
)
from multi_tenant_orchestration import MultiTenantOrchestrator
from enterprise_api_gateway import EnterpriseAPIGateway

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of metrics collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class AlertStatus(Enum):
    """Alert status states."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"

class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DOWN = "down"

@dataclass
class Metric:
    """Individual metric data point."""
    name: str
    value: float
    metric_type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tenant_id: Optional[str] = None
    
    def to_prometheus_format(self) -> str:
        """Convert metric to Prometheus format."""
        labels_str = ",".join([f'{k}="{v}"' for k, v in self.labels.items()])
        if labels_str:
            return f"{self.name}{{{labels_str}}} {self.value} {int(self.timestamp.timestamp() * 1000)}"
        else:
            return f"{self.name} {self.value} {int(self.timestamp.timestamp() * 1000)}"

@dataclass
class SLADefinition:
    """Service Level Agreement definition."""
    sla_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    tenant_id: str = ""
    
    # SLA targets
    availability_percent: float = 99.9  # 99.9% uptime
    response_time_ms: float = 2000      # < 2 seconds response time
    error_rate_percent: float = 0.1     # < 0.1% error rate
    
    # Measurement window
    measurement_window_hours: int = 24
    
    # Violation thresholds
    breach_threshold_minutes: int = 5   # How long before SLA breach
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True

@dataclass 
class AlertRule:
    """Alert rule definition."""
    rule_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    tenant_id: str = ""
    
    # Rule conditions
    metric_name: str = ""
    condition: str = ">"  # >, <, >=, <=, ==, !=
    threshold: float = 0.0
    duration_minutes: int = 5  # How long condition must be true
    
    # Alert configuration
    severity: AlertSeverity = AlertSeverity.MEDIUM
    notification_channels: Set[str] = field(default_factory=set)
    suppress_duration_minutes: int = 60  # Suppress repeating alerts
    
    # Escalation
    escalation_rules: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True
    last_triggered: Optional[datetime] = None

@dataclass
class Alert:
    """Active alert instance."""
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    rule_id: str = ""
    tenant_id: str = ""
    
    # Alert details
    title: str = ""
    description: str = ""
    severity: AlertSeverity = AlertSeverity.MEDIUM
    status: AlertStatus = AlertStatus.ACTIVE
    
    # Values
    current_value: float = 0.0
    threshold_value: float = 0.0
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    # Assignment
    assigned_to: Optional[str] = None
    acknowledged_by: Optional[str] = None
    resolved_by: Optional[str] = None
    
    # Metadata
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)

@dataclass
class HealthCheck:
    """Health check definition and status."""
    check_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    tenant_id: Optional[str] = None
    
    # Check configuration
    check_type: str = "http"  # http, tcp, custom
    target: str = ""  # URL, host:port, etc.
    timeout_seconds: int = 10
    interval_seconds: int = 30
    
    # Health criteria
    expected_status_code: int = 200
    expected_response_contains: Optional[str] = None
    max_response_time_ms: int = 5000
    
    # Current status
    status: HealthStatus = HealthStatus.HEALTHY
    last_check: Optional[datetime] = None
    last_success: Optional[datetime] = None
    consecutive_failures: int = 0
    
    # History
    check_history: List[Tuple[datetime, bool, float]] = field(default_factory=list)  # (timestamp, success, response_time)
    
    def calculate_uptime_percent(self, hours: int = 24) -> float:
        """Calculate uptime percentage over specified hours."""
        if not self.check_history:
            return 100.0
        
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        recent_checks = [(ts, success, rt) for ts, success, rt in self.check_history if ts > cutoff]
        
        if not recent_checks:
            return 100.0
        
        successful_checks = sum(1 for _, success, _ in recent_checks if success)
        return (successful_checks / len(recent_checks)) * 100.0

class MetricsCollector:
    """Collects and stores metrics from various sources."""
    
    def __init__(self, retention_hours: int = 168):  # 7 days default
        self.metrics: Dict[str, List[Metric]] = defaultdict(list)
        self.retention_hours = retention_hours
        self._lock = threading.Lock()
        
        # Start cleanup task
        self._start_cleanup_task()
    
    def record_metric(self, metric: Metric):
        """Record a new metric."""
        with self._lock:
            self.metrics[metric.name].append(metric)
    
    def record_counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None,
                     tenant_id: str = None):
        """Record counter metric."""
        metric = Metric(
            name=name,
            value=value,
            metric_type=MetricType.COUNTER,
            labels=labels or {},
            tenant_id=tenant_id
        )
        self.record_metric(metric)
    
    def record_gauge(self, name: str, value: float, labels: Dict[str, str] = None,
                    tenant_id: str = None):
        """Record gauge metric."""
        metric = Metric(
            name=name,
            value=value,
            metric_type=MetricType.GAUGE,
            labels=labels or {},
            tenant_id=tenant_id
        )
        self.record_metric(metric)
    
    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None,
                        tenant_id: str = None):
        """Record histogram metric."""
        metric = Metric(
            name=name,
            value=value,
            metric_type=MetricType.HISTOGRAM,
            labels=labels or {},
            tenant_id=tenant_id
        )
        self.record_metric(metric)
    
    def get_metrics(self, name: str, hours: int = 1, tenant_id: str = None) -> List[Metric]:
        """Get metrics for specified time range."""
        with self._lock:
            if name not in self.metrics:
                return []
            
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            filtered_metrics = []
            
            for metric in self.metrics[name]:
                if metric.timestamp < cutoff:
                    continue
                
                if tenant_id and metric.tenant_id != tenant_id:
                    continue
                
                filtered_metrics.append(metric)
            
            return filtered_metrics
    
    def get_latest_value(self, name: str, tenant_id: str = None) -> Optional[float]:
        """Get latest metric value."""
        metrics = self.get_metrics(name, hours=1, tenant_id=tenant_id)
        if metrics:
            return max(metrics, key=lambda m: m.timestamp).value
        return None
    
    def calculate_rate(self, name: str, minutes: int = 5, tenant_id: str = None) -> float:
        """Calculate rate of change for counter metrics."""
        metrics = self.get_metrics(name, hours=1, tenant_id=tenant_id)
        
        # Filter to specified time window
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        recent_metrics = [m for m in metrics if m.timestamp > cutoff]
        
        if len(recent_metrics) < 2:
            return 0.0
        
        # Sort by timestamp
        recent_metrics.sort(key=lambda m: m.timestamp)
        
        # Calculate rate
        first_metric = recent_metrics[0]
        last_metric = recent_metrics[-1]
        
        time_diff = (last_metric.timestamp - first_metric.timestamp).total_seconds()
        value_diff = last_metric.value - first_metric.value
        
        if time_diff > 0:
            return value_diff / time_diff  # Per second rate
        
        return 0.0
    
    def calculate_percentile(self, name: str, percentile: float, hours: int = 1,
                           tenant_id: str = None) -> Optional[float]:
        """Calculate percentile for histogram metrics."""
        metrics = self.get_metrics(name, hours=hours, tenant_id=tenant_id)
        
        if not metrics:
            return None
        
        values = [m.value for m in metrics]
        values.sort()
        
        if not values:
            return None
        
        index = (percentile / 100.0) * (len(values) - 1)
        
        if index == int(index):
            return values[int(index)]
        else:
            lower = values[int(index)]
            upper = values[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def _start_cleanup_task(self):
        """Start background task to clean up old metrics."""
        def cleanup():
            while True:
                try:
                    cutoff = datetime.utcnow() - timedelta(hours=self.retention_hours)
                    
                    with self._lock:
                        for metric_name in list(self.metrics.keys()):
                            self.metrics[metric_name] = [
                                m for m in self.metrics[metric_name]
                                if m.timestamp > cutoff
                            ]
                            
                            # Remove empty metric lists
                            if not self.metrics[metric_name]:
                                del self.metrics[metric_name]
                    
                    time.sleep(3600)  # Clean up every hour
                    
                except Exception as e:
                    logger.error(f"Error in metrics cleanup: {e}")
                    time.sleep(300)  # Retry in 5 minutes
        
        cleanup_thread = threading.Thread(target=cleanup, daemon=True)
        cleanup_thread.start()

class AlertManager:
    """Manages alerts and notifications."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_channels: Dict[str, Dict[str, Any]] = {}
        
        # State tracking
        self.rule_states: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Start alert evaluation
        self._start_alert_evaluation()
    
    def add_alert_rule(self, rule: AlertRule):
        """Add alert rule."""
        self.alert_rules[rule.rule_id] = rule
        logger.info(f"Added alert rule {rule.name} for tenant {rule.tenant_id}")
    
    def add_notification_channel(self, channel_id: str, channel_type: str, config: Dict[str, Any]):
        """Add notification channel (email, webhook, etc.)."""
        self.notification_channels[channel_id] = {
            'type': channel_type,
            'config': config
        }
        logger.info(f"Added notification channel {channel_id} of type {channel_type}")
    
    def create_alert(self, rule: AlertRule, current_value: float) -> Alert:
        """Create new alert from rule."""
        alert = Alert(
            rule_id=rule.rule_id,
            tenant_id=rule.tenant_id,
            title=f"{rule.name} - {rule.metric_name} {rule.condition} {rule.threshold}",
            description=rule.description,
            severity=rule.severity,
            current_value=current_value,
            threshold_value=rule.threshold
        )
        
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)
        
        # Send notifications
        self._send_alert_notifications(alert, rule)
        
        logger.warning(f"Created alert {alert.title} for tenant {alert.tenant_id}")
        return alert
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.utcnow()
            alert.acknowledged_by = acknowledged_by
            
            logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
            return True
        
        return False
    
    def resolve_alert(self, alert_id: str, resolved_by: str, resolution_note: str = None) -> bool:
        """Resolve an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.utcnow()
            alert.resolved_by = resolved_by
            
            if resolution_note:
                alert.annotations['resolution_note'] = resolution_note
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            
            logger.info(f"Alert {alert_id} resolved by {resolved_by}")
            return True
        
        return False
    
    def _start_alert_evaluation(self):
        """Start background task to evaluate alert rules."""
        def evaluate_alerts():
            while True:
                try:
                    for rule in self.alert_rules.values():
                        if not rule.is_active:
                            continue
                        
                        self._evaluate_rule(rule)
                    
                    time.sleep(60)  # Evaluate every minute
                    
                except Exception as e:
                    logger.error(f"Error in alert evaluation: {e}")
                    time.sleep(60)
        
        eval_thread = threading.Thread(target=evaluate_alerts, daemon=True)
        eval_thread.start()
    
    def _evaluate_rule(self, rule: AlertRule):
        """Evaluate individual alert rule."""
        
        # Get current metric value
        current_value = self.metrics_collector.get_latest_value(
            rule.metric_name, tenant_id=rule.tenant_id
        )
        
        if current_value is None:
            return
        
        # Check condition
        condition_met = self._check_condition(current_value, rule.condition, rule.threshold)
        
        rule_state = self.rule_states[rule.rule_id]
        
        if condition_met:
            # Track how long condition has been true
            if 'condition_start' not in rule_state:
                rule_state['condition_start'] = datetime.utcnow()
            
            # Check if duration threshold met
            duration = datetime.utcnow() - rule_state['condition_start']
            if duration.total_seconds() >= rule.duration_minutes * 60:
                
                # Check if alert already exists for this rule
                existing_alert = None
                for alert in self.active_alerts.values():
                    if alert.rule_id == rule.rule_id and alert.status == AlertStatus.ACTIVE:
                        existing_alert = alert
                        break
                
                if not existing_alert:
                    # Create new alert
                    self.create_alert(rule, current_value)
                    rule.last_triggered = datetime.utcnow()
        else:
            # Condition no longer met
            if 'condition_start' in rule_state:
                del rule_state['condition_start']
            
            # Auto-resolve alerts if condition clears
            for alert_id, alert in list(self.active_alerts.items()):
                if alert.rule_id == rule.rule_id and alert.status == AlertStatus.ACTIVE:
                    self.resolve_alert(alert_id, "system", "Condition no longer met")
    
    def _check_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Check if alert condition is met."""
        if condition == ">":
            return value > threshold
        elif condition == "<":
            return value < threshold
        elif condition == ">=":
            return value >= threshold
        elif condition == "<=":
            return value <= threshold
        elif condition == "==":
            return value == threshold
        elif condition == "!=":
            return value != threshold
        else:
            return False
    
    def _send_alert_notifications(self, alert: Alert, rule: AlertRule):
        """Send notifications for alert."""
        
        for channel_id in rule.notification_channels:
            if channel_id in self.notification_channels:
                channel = self.notification_channels[channel_id]
                
                try:
                    if channel['type'] == 'email':
                        self._send_email_notification(alert, channel['config'])
                    elif channel['type'] == 'webhook':
                        self._send_webhook_notification(alert, channel['config'])
                    elif channel['type'] == 'slack':
                        self._send_slack_notification(alert, channel['config'])
                except Exception as e:
                    logger.error(f"Failed to send notification via {channel_id}: {e}")
    
    def _send_email_notification(self, alert: Alert, config: Dict[str, Any]):
        """Send email notification."""
        
        msg = MIMEMultipart()
        msg['From'] = config['from_email']
        msg['To'] = ", ".join(config['to_emails'])
        msg['Subject'] = f"[{alert.severity.name}] {alert.title}"
        
        body = f"""
        Alert: {alert.title}
        Severity: {alert.severity.name}
        Tenant: {alert.tenant_id}
        
        Description: {alert.description}
        
        Current Value: {alert.current_value}
        Threshold: {alert.threshold_value}
        
        Created: {alert.created_at.isoformat()}
        Alert ID: {alert.alert_id}
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(config['smtp_host'], config['smtp_port'])
        if config.get('smtp_tls'):
            server.starttls()
        if config.get('smtp_username'):
            server.login(config['smtp_username'], config['smtp_password'])
        
        server.send_message(msg)
        server.quit()
    
    def _send_webhook_notification(self, alert: Alert, config: Dict[str, Any]):
        """Send webhook notification."""
        
        payload = {
            'alert_id': alert.alert_id,
            'title': alert.title,
            'description': alert.description,
            'severity': alert.severity.name,
            'tenant_id': alert.tenant_id,
            'current_value': alert.current_value,
            'threshold_value': alert.threshold_value,
            'created_at': alert.created_at.isoformat(),
            'status': alert.status.value
        }
        
        headers = {'Content-Type': 'application/json'}
        if 'auth_token' in config:
            headers['Authorization'] = f"Bearer {config['auth_token']}"
        
        response = requests.post(
            config['webhook_url'],
            json=payload,
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
    
    def _send_slack_notification(self, alert: Alert, config: Dict[str, Any]):
        """Send Slack notification."""
        
        color_map = {
            AlertSeverity.LOW: "good",
            AlertSeverity.MEDIUM: "warning", 
            AlertSeverity.HIGH: "danger",
            AlertSeverity.CRITICAL: "danger"
        }
        
        payload = {
            "attachments": [{
                "color": color_map.get(alert.severity, "warning"),
                "title": alert.title,
                "text": alert.description,
                "fields": [
                    {"title": "Severity", "value": alert.severity.name, "short": True},
                    {"title": "Tenant", "value": alert.tenant_id, "short": True},
                    {"title": "Current Value", "value": str(alert.current_value), "short": True},
                    {"title": "Threshold", "value": str(alert.threshold_value), "short": True}
                ],
                "footer": "Claude Nexus Monitoring",
                "ts": int(alert.created_at.timestamp())
            }]
        }
        
        response = requests.post(
            config['webhook_url'],
            json=payload,
            timeout=10
        )
        response.raise_for_status()
    
    def get_alert_summary(self, tenant_id: str = None) -> Dict[str, Any]:
        """Get alert summary statistics."""
        
        active_alerts = list(self.active_alerts.values())
        if tenant_id:
            active_alerts = [a for a in active_alerts if a.tenant_id == tenant_id]
        
        # Count by severity
        severity_counts = defaultdict(int)
        for alert in active_alerts:
            severity_counts[alert.severity.name] += 1
        
        # Recent alert history (last 24 hours)
        cutoff = datetime.utcnow() - timedelta(hours=24)
        recent_alerts = [a for a in self.alert_history if a.created_at > cutoff]
        if tenant_id:
            recent_alerts = [a for a in recent_alerts if a.tenant_id == tenant_id]
        
        return {
            'active_alerts': len(active_alerts),
            'severity_breakdown': dict(severity_counts),
            'recent_alerts_24h': len(recent_alerts),
            'total_rules': len([r for r in self.alert_rules.values() 
                              if not tenant_id or r.tenant_id == tenant_id])
        }

class HealthMonitor:
    """Monitors system and service health."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.health_checks: Dict[str, HealthCheck] = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Start health checking
        self._start_health_monitoring()
    
    def add_health_check(self, health_check: HealthCheck):
        """Add health check."""
        self.health_checks[health_check.check_id] = health_check
        logger.info(f"Added health check {health_check.name}")
    
    def _start_health_monitoring(self):
        """Start background health monitoring."""
        def monitor_health():
            while True:
                try:
                    # Submit health checks to thread pool
                    futures = []
                    for check in self.health_checks.values():
                        if self._should_run_check(check):
                            future = self.executor.submit(self._perform_health_check, check)
                            futures.append(future)
                    
                    # Wait for all checks to complete
                    for future in futures:
                        try:
                            future.result(timeout=30)
                        except Exception as e:
                            logger.error(f"Health check failed: {e}")
                    
                    time.sleep(10)  # Check interval
                    
                except Exception as e:
                    logger.error(f"Error in health monitoring: {e}")
                    time.sleep(30)
        
        monitor_thread = threading.Thread(target=monitor_health, daemon=True)
        monitor_thread.start()
    
    def _should_run_check(self, check: HealthCheck) -> bool:
        """Check if health check should be run."""
        if not check.last_check:
            return True
        
        elapsed = datetime.utcnow() - check.last_check
        return elapsed.total_seconds() >= check.interval_seconds
    
    def _perform_health_check(self, check: HealthCheck):
        """Perform individual health check."""
        start_time = time.time()
        success = False
        response_time_ms = 0
        
        try:
            if check.check_type == "http":
                success, response_time_ms = self._http_health_check(check)
            elif check.check_type == "tcp":
                success, response_time_ms = self._tcp_health_check(check)
            elif check.check_type == "custom":
                success, response_time_ms = self._custom_health_check(check)
            
        except Exception as e:
            logger.error(f"Health check {check.name} failed: {e}")
            success = False
            response_time_ms = (time.time() - start_time) * 1000
        
        # Update check status
        check.last_check = datetime.utcnow()
        
        if success:
            check.last_success = datetime.utcnow()
            check.consecutive_failures = 0
            
            if check.status in [HealthStatus.CRITICAL, HealthStatus.DOWN]:
                check.status = HealthStatus.HEALTHY
        else:
            check.consecutive_failures += 1
            
            # Update status based on failure count
            if check.consecutive_failures >= 3:
                check.status = HealthStatus.DOWN
            elif check.consecutive_failures >= 2:
                check.status = HealthStatus.CRITICAL
            else:
                check.status = HealthStatus.WARNING
        
        # Record check history
        check.check_history.append((datetime.utcnow(), success, response_time_ms))
        
        # Limit history size
        if len(check.check_history) > 1000:
            check.check_history = check.check_history[-1000:]
        
        # Record metrics
        labels = {
            'check_name': check.name,
            'check_type': check.check_type,
            'status': 'success' if success else 'failure'
        }
        
        if check.tenant_id:
            labels['tenant_id'] = check.tenant_id
        
        self.metrics_collector.record_counter(
            'health_check_total', 1.0, labels, check.tenant_id
        )
        
        self.metrics_collector.record_histogram(
            'health_check_duration_ms', response_time_ms, labels, check.tenant_id
        )
        
        self.metrics_collector.record_gauge(
            'health_check_up', 1.0 if success else 0.0, 
            {'check_name': check.name}, check.tenant_id
        )
    
    def _http_health_check(self, check: HealthCheck) -> Tuple[bool, float]:
        """Perform HTTP health check."""
        start_time = time.time()
        
        try:
            response = requests.get(
                check.target,
                timeout=check.timeout_seconds,
                allow_redirects=True
            )
            
            response_time_ms = (time.time() - start_time) * 1000
            
            # Check status code
            if response.status_code != check.expected_status_code:
                return False, response_time_ms
            
            # Check response time
            if response_time_ms > check.max_response_time_ms:
                return False, response_time_ms
            
            # Check response content
            if check.expected_response_contains:
                if check.expected_response_contains not in response.text:
                    return False, response_time_ms
            
            return True, response_time_ms
            
        except Exception:
            response_time_ms = (time.time() - start_time) * 1000
            return False, response_time_ms
    
    def _tcp_health_check(self, check: HealthCheck) -> Tuple[bool, float]:
        """Perform TCP health check."""
        start_time = time.time()
        
        try:
            host, port = check.target.split(':')
            port = int(port)
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(check.timeout_seconds)
            
            result = sock.connect_ex((host, port))
            sock.close()
            
            response_time_ms = (time.time() - start_time) * 1000
            
            return result == 0, response_time_ms
            
        except Exception:
            response_time_ms = (time.time() - start_time) * 1000
            return False, response_time_ms
    
    def _custom_health_check(self, check: HealthCheck) -> Tuple[bool, float]:
        """Perform custom health check (placeholder)."""
        # Implement custom health check logic based on check.target
        # For now, return success
        return True, 100.0
    
    def get_overall_health(self, tenant_id: str = None) -> Dict[str, Any]:
        """Get overall system health status."""
        
        relevant_checks = list(self.health_checks.values())
        if tenant_id:
            relevant_checks = [c for c in relevant_checks if c.tenant_id == tenant_id]
        
        if not relevant_checks:
            return {'status': 'unknown', 'checks': 0}
        
        # Count by status
        status_counts = defaultdict(int)
        for check in relevant_checks:
            status_counts[check.status.value] += 1
        
        # Determine overall status
        if status_counts['down'] > 0:
            overall_status = 'down'
        elif status_counts['critical'] > 0:
            overall_status = 'critical'
        elif status_counts['warning'] > 0:
            overall_status = 'warning'
        else:
            overall_status = 'healthy'
        
        # Calculate uptime
        uptime_values = []
        for check in relevant_checks:
            uptime = check.calculate_uptime_percent(24)
            uptime_values.append(uptime)
        
        avg_uptime = statistics.mean(uptime_values) if uptime_values else 100.0
        
        return {
            'status': overall_status,
            'checks': len(relevant_checks),
            'status_breakdown': dict(status_counts),
            'avg_uptime_24h': round(avg_uptime, 2)
        }

class EnterpriseMonitoringSystem:
    """Main enterprise monitoring system."""
    
    def __init__(self, orchestrator: MultiTenantOrchestrator = None,
                 api_gateway: EnterpriseAPIGateway = None,
                 audit_logger=None):
        
        self.orchestrator = orchestrator
        self.api_gateway = api_gateway
        self.audit_logger = audit_logger
        
        # Core components
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager(self.metrics_collector)
        self.health_monitor = HealthMonitor(self.metrics_collector)
        
        # SLA management
        self.sla_definitions: Dict[str, SLADefinition] = {}
        
        # System metrics collection
        self._start_system_metrics_collection()
        
        # Integration metrics collection
        self._start_integration_metrics_collection()
    
    def add_sla_definition(self, sla: SLADefinition):
        """Add SLA definition."""
        self.sla_definitions[sla.sla_id] = sla
        
        # Create corresponding alert rules
        self._create_sla_alert_rules(sla)
        
        logger.info(f"Added SLA definition {sla.name} for tenant {sla.tenant_id}")
    
    def _create_sla_alert_rules(self, sla: SLADefinition):
        """Create alert rules for SLA monitoring."""
        
        # Response time SLA rule
        response_time_rule = AlertRule(
            name=f"SLA: Response Time - {sla.name}",
            description=f"Response time exceeds SLA target of {sla.response_time_ms}ms",
            tenant_id=sla.tenant_id,
            metric_name="http_request_duration_ms_p95",
            condition=">",
            threshold=sla.response_time_ms,
            duration_minutes=sla.breach_threshold_minutes,
            severity=AlertSeverity.HIGH
        )
        self.alert_manager.add_alert_rule(response_time_rule)
        
        # Error rate SLA rule
        error_rate_rule = AlertRule(
            name=f"SLA: Error Rate - {sla.name}",
            description=f"Error rate exceeds SLA target of {sla.error_rate_percent}%",
            tenant_id=sla.tenant_id,
            metric_name="http_request_error_rate_percent",
            condition=">",
            threshold=sla.error_rate_percent,
            duration_minutes=sla.breach_threshold_minutes,
            severity=AlertSeverity.HIGH
        )
        self.alert_manager.add_alert_rule(error_rate_rule)
        
        # Availability SLA rule
        availability_rule = AlertRule(
            name=f"SLA: Availability - {sla.name}",
            description=f"Availability below SLA target of {sla.availability_percent}%",
            tenant_id=sla.tenant_id,
            metric_name="service_availability_percent",
            condition="<",
            threshold=sla.availability_percent,
            duration_minutes=sla.breach_threshold_minutes,
            severity=AlertSeverity.CRITICAL
        )
        self.alert_manager.add_alert_rule(availability_rule)
    
    def _start_system_metrics_collection(self):
        """Start collecting system-level metrics."""
        def collect_system_metrics():
            while True:
                try:
                    # CPU usage
                    cpu_percent = psutil.cpu_percent(interval=1)
                    self.metrics_collector.record_gauge('system_cpu_usage_percent', cpu_percent)
                    
                    # Memory usage
                    memory = psutil.virtual_memory()
                    self.metrics_collector.record_gauge('system_memory_usage_percent', memory.percent)
                    self.metrics_collector.record_gauge('system_memory_available_bytes', memory.available)
                    
                    # Disk usage
                    disk = psutil.disk_usage('/')
                    self.metrics_collector.record_gauge('system_disk_usage_percent', 
                                                       (disk.used / disk.total) * 100)
                    
                    # Network I/O
                    network = psutil.net_io_counters()
                    self.metrics_collector.record_counter('system_network_bytes_sent', network.bytes_sent)
                    self.metrics_collector.record_counter('system_network_bytes_recv', network.bytes_recv)
                    
                    # Load average (Unix systems)
                    if hasattr(psutil, 'getloadavg'):
                        load_avg = psutil.getloadavg()
                        self.metrics_collector.record_gauge('system_load_average_1m', load_avg[0])
                        self.metrics_collector.record_gauge('system_load_average_5m', load_avg[1])
                        self.metrics_collector.record_gauge('system_load_average_15m', load_avg[2])
                    
                    time.sleep(30)  # Collect every 30 seconds
                    
                except Exception as e:
                    logger.error(f"Error collecting system metrics: {e}")
                    time.sleep(60)
        
        metrics_thread = threading.Thread(target=collect_system_metrics, daemon=True)
        metrics_thread.start()
    
    def _start_integration_metrics_collection(self):
        """Start collecting metrics from integrated systems."""
        def collect_integration_metrics():
            while True:
                try:
                    # Orchestrator metrics
                    if self.orchestrator:
                        system_metrics = self.orchestrator.get_system_metrics()
                        
                        self.metrics_collector.record_gauge(
                            'orchestrator_total_tenants', system_metrics.get('total_tenants', 0)
                        )
                        self.metrics_collector.record_gauge(
                            'orchestrator_total_agents', system_metrics.get('total_agents', 0)
                        )
                        self.metrics_collector.record_gauge(
                            'orchestrator_available_agents', system_metrics.get('available_agents', 0)
                        )
                        self.metrics_collector.record_gauge(
                            'orchestrator_active_consultations', system_metrics.get('active_consultations', 0)
                        )
                        self.metrics_collector.record_gauge(
                            'orchestrator_queue_length', system_metrics.get('queue_length', 0)
                        )
                        self.metrics_collector.record_gauge(
                            'orchestrator_avg_response_time_ms', system_metrics.get('avg_response_time_ms', 0)
                        )
                        
                        # Per-tenant metrics
                        for tenant_id in self.orchestrator.tenant_profiles.keys():
                            tenant_metrics = self.orchestrator.get_tenant_metrics(tenant_id)
                            
                            labels = {'tenant_id': tenant_id}
                            
                            self.metrics_collector.record_gauge(
                                'tenant_total_consultations', 
                                tenant_metrics.get('total_consultations', 0),
                                labels, tenant_id
                            )
                            self.metrics_collector.record_gauge(
                                'tenant_success_rate_percent',
                                tenant_metrics.get('success_rate_percent', 0),
                                labels, tenant_id
                            )
                    
                    # API Gateway metrics
                    if self.api_gateway:
                        gateway_metrics = self.api_gateway._get_gateway_metrics()
                        
                        self.metrics_collector.record_counter(
                            'gateway_total_requests', gateway_metrics.get('total_requests', 0)
                        )
                        self.metrics_collector.record_counter(
                            'gateway_total_errors', gateway_metrics.get('total_errors', 0)
                        )
                        self.metrics_collector.record_gauge(
                            'gateway_error_rate_percent', gateway_metrics.get('error_rate_percent', 0)
                        )
                    
                    time.sleep(60)  # Collect every minute
                    
                except Exception as e:
                    logger.error(f"Error collecting integration metrics: {e}")
                    time.sleep(120)
        
        integration_thread = threading.Thread(target=collect_integration_metrics, daemon=True)
        integration_thread.start()
    
    def get_tenant_dashboard(self, tenant_id: str) -> Dict[str, Any]:
        """Get comprehensive dashboard data for tenant."""
        
        # Get basic metrics
        tenant_metrics = {}
        if self.orchestrator:
            tenant_metrics = self.orchestrator.get_tenant_metrics(tenant_id)
        
        # Get health status
        health_status = self.health_monitor.get_overall_health(tenant_id)
        
        # Get alert summary
        alert_summary = self.alert_manager.get_alert_summary(tenant_id)
        
        # Get SLA compliance
        sla_compliance = self._calculate_sla_compliance(tenant_id)
        
        # Get recent performance metrics
        response_times = self.metrics_collector.get_metrics(
            'http_request_duration_ms', hours=24, tenant_id=tenant_id
        )
        
        avg_response_time = 0.0
        if response_times:
            avg_response_time = statistics.mean([m.value for m in response_times])
        
        return {
            'tenant_id': tenant_id,
            'health_status': health_status,
            'alert_summary': alert_summary,
            'sla_compliance': sla_compliance,
            'performance_metrics': {
                'avg_response_time_24h': round(avg_response_time, 2),
                'total_consultations': tenant_metrics.get('total_consultations', 0),
                'success_rate_percent': tenant_metrics.get('success_rate_percent', 0),
                'active_consultations': tenant_metrics.get('active_consultations', 0)
            },
            'quota_status': tenant_metrics.get('quota_status', {}),
            'last_updated': datetime.utcnow().isoformat()
        }
    
    def _calculate_sla_compliance(self, tenant_id: str) -> Dict[str, Any]:
        """Calculate SLA compliance for tenant."""
        
        tenant_slas = [sla for sla in self.sla_definitions.values() 
                      if sla.tenant_id == tenant_id]
        
        compliance_results = {}
        
        for sla in tenant_slas:
            # Calculate availability
            availability = self.health_monitor.get_overall_health(tenant_id).get('avg_uptime_24h', 100.0)
            
            # Calculate response time compliance
            response_times = self.metrics_collector.get_metrics(
                'http_request_duration_ms', hours=sla.measurement_window_hours, tenant_id=tenant_id
            )
            
            response_time_compliance = 100.0
            if response_times:
                slow_requests = len([m for m in response_times if m.value > sla.response_time_ms])
                response_time_compliance = ((len(response_times) - slow_requests) / len(response_times)) * 100
            
            # Calculate error rate compliance
            error_rate = 0.0
            if self.orchestrator and tenant_id in self.orchestrator.tenant_profiles:
                tenant_profile = self.orchestrator.tenant_profiles[tenant_id]
                total = tenant_profile.total_consultations
                if total > 0:
                    error_rate = (tenant_profile.failed_consultations / total) * 100
            
            error_rate_compliance = 100.0 if error_rate <= sla.error_rate_percent else 0.0
            
            # Overall compliance
            overall_compliance = min(availability, response_time_compliance, error_rate_compliance)
            
            compliance_results[sla.sla_id] = {
                'sla_name': sla.name,
                'overall_compliance_percent': round(overall_compliance, 2),
                'availability_percent': round(availability, 2),
                'response_time_compliance_percent': round(response_time_compliance, 2),
                'error_rate_compliance_percent': round(error_rate_compliance, 2),
                'target_availability': sla.availability_percent,
                'target_response_time_ms': sla.response_time_ms,
                'target_error_rate_percent': sla.error_rate_percent
            }
        
        return compliance_results
    
    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        
        lines = []
        
        for metric_name, metric_list in self.metrics_collector.metrics.items():
            for metric in metric_list[-100:]:  # Last 100 points
                lines.append(metric.to_prometheus_format())
        
        return "\n".join(lines)
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get system-wide overview metrics."""
        
        # System metrics
        cpu_usage = self.metrics_collector.get_latest_value('system_cpu_usage_percent')
        memory_usage = self.metrics_collector.get_latest_value('system_memory_usage_percent')
        
        # Orchestrator metrics
        orchestrator_metrics = {}
        if self.orchestrator:
            orchestrator_metrics = self.orchestrator.get_system_metrics()
        
        # Gateway metrics
        gateway_metrics = {}
        if self.api_gateway:
            gateway_metrics = self.api_gateway._get_gateway_metrics()
        
        # Overall health
        overall_health = self.health_monitor.get_overall_health()
        
        # Alert summary
        alert_summary = self.alert_manager.get_alert_summary()
        
        return {
            'system_health': {
                'status': overall_health.get('status', 'unknown'),
                'cpu_usage_percent': cpu_usage or 0.0,
                'memory_usage_percent': memory_usage or 0.0
            },
            'orchestrator_metrics': orchestrator_metrics,
            'gateway_metrics': gateway_metrics,
            'alert_summary': alert_summary,
            'uptime_hours': self._get_system_uptime_hours(),
            'last_updated': datetime.utcnow().isoformat()
        }
    
    def _get_system_uptime_hours(self) -> float:
        """Get system uptime in hours."""
        try:
            uptime_seconds = time.time() - psutil.boot_time()
            return uptime_seconds / 3600
        except:
            return 0.0

# Example usage
if __name__ == "__main__":
    from multi_tenant_orchestration import MultiTenantOrchestrator
    from enterprise_api_gateway import EnterpriseAPIGateway
    from enterprise_security_architecture import SOC2ComplianceEngine
    
    # Initialize monitoring system
    compliance_engine = SOC2ComplianceEngine()
    orchestrator = MultiTenantOrchestrator(audit_logger=compliance_engine)
    gateway = EnterpriseAPIGateway(audit_logger=compliance_engine)
    
    monitoring = EnterpriseMonitoringSystem(
        orchestrator=orchestrator,
        api_gateway=gateway,
        audit_logger=compliance_engine
    )
    
    # Add SLA definition
    sla = SLADefinition(
        name="Premium SLA",
        tenant_id="test-corp",
        availability_percent=99.9,
        response_time_ms=1000,
        error_rate_percent=0.1
    )
    monitoring.add_sla_definition(sla)
    
    # Add health check
    health_check = HealthCheck(
        name="API Gateway Health",
        check_type="http",
        target="http://localhost:8080/health",
        interval_seconds=30
    )
    monitoring.health_monitor.add_health_check(health_check)
    
    # Add notification channel
    monitoring.alert_manager.add_notification_channel(
        "email_ops",
        "email",
        {
            'smtp_host': 'smtp.example.com',
            'smtp_port': 587,
            'smtp_tls': True,
            'from_email': 'alerts@claude-nexus.com',
            'to_emails': ['ops@example.com']
        }
    )
    
    # Get system overview
    overview = monitoring.get_system_overview()
    print(f"System Overview: {json.dumps(overview, indent=2)}")
    
    # Get tenant dashboard
    dashboard = monitoring.get_tenant_dashboard("test-corp")
    print(f"Tenant Dashboard: {json.dumps(dashboard, indent=2)}")
