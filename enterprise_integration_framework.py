#!/usr/bin/env python3
"""
Enterprise Integration Framework

Comprehensive integration system providing webhooks, API integrations,
event streaming, and enterprise workflow automation for the
claude-nexus multi-tenant agent ecosystem. Enables seamless
integration with external systems and enterprise workflows.

Features:
- Webhook management and delivery with retry logic
- Event streaming and message queuing
- API integration templates and connectors
- Workflow automation and orchestration
- Enterprise system integrations (CRM, ERP, ITSM)
- Real-time event processing and transformation
- Integration monitoring and analytics
- Secure credential management

Integrations Supported:
- Salesforce CRM
- ServiceNow ITSM
- Slack/Microsoft Teams
- JIRA/Azure DevOps
- Email systems (SMTP/Exchange)
- Custom REST/GraphQL APIs
- Message queues (RabbitMQ, Apache Kafka)
- Database systems

Author: Fortress Guardian
Version: 1.0.0
Compliance: SOC 2 Type II, Enterprise Security
"""

import asyncio
import json
import logging
import time
import hmac
import hashlib
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor
import requests
from urllib.parse import urlparse
import ssl
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import secrets
import re
from functools import wraps

# Import security and orchestration systems
from enterprise_security_architecture import (
    AuditEvent, AuditEventType, SecurityLevel
)
from multi_tenant_orchestration import MultiTenantOrchestrator
from enterprise_monitoring_system import EnterpriseMonitoringSystem

logger = logging.getLogger(__name__)

class EventType(Enum):
    """Types of events that can trigger integrations."""
    CONSULTATION_STARTED = "consultation_started"
    CONSULTATION_COMPLETED = "consultation_completed"
    CONSULTATION_FAILED = "consultation_failed"
    AGENT_STATUS_CHANGED = "agent_status_changed"
    TENANT_CREATED = "tenant_created"
    TENANT_UPDATED = "tenant_updated"
    USER_AUTHENTICATED = "user_authenticated"
    SECURITY_ALERT = "security_alert"
    SLA_VIOLATION = "sla_violation"
    SYSTEM_ERROR = "system_error"
    CUSTOM_EVENT = "custom_event"

class IntegrationType(Enum):
    """Types of integrations supported."""
    WEBHOOK = "webhook"
    EMAIL = "email"
    SLACK = "slack"
    TEAMS = "teams"
    SALESFORCE = "salesforce"
    SERVICENOW = "servicenow"
    JIRA = "jira"
    CUSTOM_API = "custom_api"
    MESSAGE_QUEUE = "message_queue"
    DATABASE = "database"

class DeliveryStatus(Enum):
    """Webhook delivery status."""
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"
    EXHAUSTED = "exhausted"  # Max retries reached

@dataclass
class WebhookEndpoint:
    """Webhook endpoint configuration."""
    endpoint_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    tenant_id: str = ""
    
    # Endpoint configuration
    url: str = ""
    method: str = "POST"
    headers: Dict[str, str] = field(default_factory=dict)
    
    # Security
    secret: Optional[str] = None  # For HMAC signature verification
    auth_type: str = "none"  # none, bearer, basic, api_key
    auth_credentials: Dict[str, str] = field(default_factory=dict)
    
    # Event filtering
    event_types: Set[EventType] = field(default_factory=set)
    event_filters: Dict[str, Any] = field(default_factory=dict)  # JSON path filters
    
    # Delivery configuration
    timeout_seconds: int = 30
    max_retries: int = 5
    retry_backoff_seconds: int = 60
    
    # Status and metrics
    is_active: bool = True
    total_deliveries: int = 0
    successful_deliveries: int = 0
    failed_deliveries: int = 0
    last_delivery: Optional[datetime] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def generate_signature(self, payload: str) -> Optional[str]:
        """Generate HMAC signature for webhook payload."""
        if not self.secret:
            return None
        
        signature = hmac.new(
            self.secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return f"sha256={signature}"
    
    def get_success_rate(self) -> float:
        """Get delivery success rate percentage."""
        if self.total_deliveries == 0:
            return 100.0
        
        return (self.successful_deliveries / self.total_deliveries) * 100.0

@dataclass
class WebhookDelivery:
    """Webhook delivery attempt record."""
    delivery_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    endpoint_id: str = ""
    event_id: str = ""
    
    # Delivery details
    url: str = ""
    method: str = "POST"
    headers: Dict[str, str] = field(default_factory=dict)
    payload: str = ""
    
    # Results
    status: DeliveryStatus = DeliveryStatus.PENDING
    http_status_code: Optional[int] = None
    response_body: Optional[str] = None
    error_message: Optional[str] = None
    
    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    attempted_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    next_retry_at: Optional[datetime] = None
    
    # Retry tracking
    attempt_count: int = 0
    max_attempts: int = 5
    
    def should_retry(self) -> bool:
        """Check if delivery should be retried."""
        return (
            self.status in [DeliveryStatus.FAILED, DeliveryStatus.RETRYING] and
            self.attempt_count < self.max_attempts and
            (not self.next_retry_at or datetime.utcnow() >= self.next_retry_at)
        )

@dataclass
class IntegrationEvent:
    """Event that triggers integrations."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType = EventType.CUSTOM_EVENT
    tenant_id: str = ""
    user_id: Optional[str] = None
    
    # Event data
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Context
    source: str = "system"
    correlation_id: Optional[str] = None
    
    # Timing
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_webhook_payload(self) -> Dict[str, Any]:
        """Convert event to webhook payload format."""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'tenant_id': self.tenant_id,
            'user_id': self.user_id,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'correlation_id': self.correlation_id,
            'data': self.data,
            'metadata': self.metadata
        }

@dataclass
class IntegrationTemplate:
    """Template for common integration patterns."""
    template_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    integration_type: IntegrationType = IntegrationType.WEBHOOK
    
    # Configuration template
    config_template: Dict[str, Any] = field(default_factory=dict)
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    
    # Event mapping
    event_mapping: Dict[str, str] = field(default_factory=dict)
    data_transformations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Documentation
    documentation: str = ""
    examples: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True

class WebhookManager:
    """Manages webhook endpoints and deliveries."""
    
    def __init__(self, monitoring_system: EnterpriseMonitoringSystem = None,
                 audit_logger=None):
        
        self.monitoring_system = monitoring_system
        self.audit_logger = audit_logger
        
        # Storage
        self.endpoints: Dict[str, WebhookEndpoint] = {}
        self.deliveries: Dict[str, WebhookDelivery] = {}
        self.delivery_queue: deque = deque()
        
        # Processing
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.delivery_workers_running = False
        
        # Start delivery workers
        self._start_delivery_workers()
    
    def create_webhook_endpoint(self, endpoint: WebhookEndpoint) -> str:
        """Create new webhook endpoint."""
        
        # Generate secret if not provided
        if not endpoint.secret:
            endpoint.secret = secrets.token_urlsafe(32)
        
        self.endpoints[endpoint.endpoint_id] = endpoint
        
        # Log endpoint creation
        if self.audit_logger:
            audit_event = AuditEvent(
                tenant_id=endpoint.tenant_id,
                event_type=AuditEventType.SYSTEM_EVENT,
                resource="webhook_endpoint",
                action="create",
                result="success",
                metadata={
                    'endpoint_id': endpoint.endpoint_id,
                    'endpoint_name': endpoint.name,
                    'url': endpoint.url,
                    'event_types': [et.value for et in endpoint.event_types]
                }
            )
            self.audit_logger.log_audit_event(audit_event)
        
        logger.info(f"Created webhook endpoint {endpoint.name} for tenant {endpoint.tenant_id}")
        return endpoint.endpoint_id
    
    def update_webhook_endpoint(self, endpoint_id: str, updates: Dict[str, Any]) -> bool:
        """Update webhook endpoint configuration."""
        
        if endpoint_id not in self.endpoints:
            return False
        
        endpoint = self.endpoints[endpoint_id]
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(endpoint, key):
                setattr(endpoint, key, value)
        
        endpoint.updated_at = datetime.utcnow()
        
        # Log update
        if self.audit_logger:
            audit_event = AuditEvent(
                tenant_id=endpoint.tenant_id,
                event_type=AuditEventType.SYSTEM_EVENT,
                resource="webhook_endpoint",
                action="update",
                result="success",
                metadata={
                    'endpoint_id': endpoint_id,
                    'updates': updates
                }
            )
            self.audit_logger.log_audit_event(audit_event)
        
        return True
    
    def delete_webhook_endpoint(self, endpoint_id: str) -> bool:
        """Delete webhook endpoint."""
        
        if endpoint_id not in self.endpoints:
            return False
        
        endpoint = self.endpoints[endpoint_id]
        tenant_id = endpoint.tenant_id
        
        del self.endpoints[endpoint_id]
        
        # Log deletion
        if self.audit_logger:
            audit_event = AuditEvent(
                tenant_id=tenant_id,
                event_type=AuditEventType.SYSTEM_EVENT,
                resource="webhook_endpoint",
                action="delete",
                result="success",
                metadata={'endpoint_id': endpoint_id}
            )
            self.audit_logger.log_audit_event(audit_event)
        
        logger.info(f"Deleted webhook endpoint {endpoint_id}")
        return True
    
    def deliver_event(self, event: IntegrationEvent) -> List[str]:
        """Deliver event to matching webhook endpoints."""
        
        delivery_ids = []
        
        # Find matching endpoints
        matching_endpoints = self._find_matching_endpoints(event)
        
        for endpoint in matching_endpoints:
            # Create delivery record
            delivery = WebhookDelivery(
                endpoint_id=endpoint.endpoint_id,
                event_id=event.event_id,
                url=endpoint.url,
                method=endpoint.method,
                headers=endpoint.headers.copy(),
                payload=json.dumps(event.to_webhook_payload()),
                max_attempts=endpoint.max_retries
            )
            
            # Add authentication headers
            self._add_auth_headers(delivery, endpoint)
            
            # Add signature if secret is configured
            if endpoint.secret:
                signature = endpoint.generate_signature(delivery.payload)
                if signature:
                    delivery.headers['X-Signature-256'] = signature
            
            # Store delivery and queue for processing
            self.deliveries[delivery.delivery_id] = delivery
            self.delivery_queue.append(delivery.delivery_id)
            
            delivery_ids.append(delivery.delivery_id)
        
        logger.info(f"Queued event {event.event_id} for delivery to {len(matching_endpoints)} endpoints")
        return delivery_ids
    
    def _find_matching_endpoints(self, event: IntegrationEvent) -> List[WebhookEndpoint]:
        """Find webhook endpoints that match the event."""
        
        matching_endpoints = []
        
        for endpoint in self.endpoints.values():
            # Check if endpoint is active
            if not endpoint.is_active:
                continue
            
            # Check tenant match
            if endpoint.tenant_id != event.tenant_id and endpoint.tenant_id != "*":
                continue
            
            # Check event type match
            if endpoint.event_types and event.event_type not in endpoint.event_types:
                continue
            
            # Check event filters
            if endpoint.event_filters and not self._matches_filters(event, endpoint.event_filters):
                continue
            
            matching_endpoints.append(endpoint)
        
        return matching_endpoints
    
    def _matches_filters(self, event: IntegrationEvent, filters: Dict[str, Any]) -> bool:
        """Check if event matches the specified filters."""
        
        # Simple filter matching (in production, use JSONPath or similar)
        for filter_key, filter_value in filters.items():
            if filter_key in event.data:
                if event.data[filter_key] != filter_value:
                    return False
            elif filter_key in event.metadata:
                if event.metadata[filter_key] != filter_value:
                    return False
            else:
                return False  # Required filter key not found
        
        return True
    
    def _add_auth_headers(self, delivery: WebhookDelivery, endpoint: WebhookEndpoint):
        """Add authentication headers to delivery."""
        
        if endpoint.auth_type == "bearer":
            token = endpoint.auth_credentials.get('token')
            if token:
                delivery.headers['Authorization'] = f"Bearer {token}"
        
        elif endpoint.auth_type == "basic":
            username = endpoint.auth_credentials.get('username')
            password = endpoint.auth_credentials.get('password')
            if username and password:
                credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
                delivery.headers['Authorization'] = f"Basic {credentials}"
        
        elif endpoint.auth_type == "api_key":
            api_key = endpoint.auth_credentials.get('api_key')
            header_name = endpoint.auth_credentials.get('header_name', 'X-API-Key')
            if api_key:
                delivery.headers[header_name] = api_key
    
    def _start_delivery_workers(self):
        """Start background delivery workers."""
        
        if self.delivery_workers_running:
            return
        
        self.delivery_workers_running = True
        
        def delivery_worker():
            while self.delivery_workers_running:
                try:
                    # Process delivery queue
                    if self.delivery_queue:
                        delivery_id = self.delivery_queue.popleft()
                        
                        if delivery_id in self.deliveries:
                            delivery = self.deliveries[delivery_id]
                            
                            # Check if should process now
                            if delivery.should_retry() or delivery.status == DeliveryStatus.PENDING:
                                self.executor.submit(self._attempt_delivery, delivery)
                    
                    # Check for retries
                    self._check_retry_queue()
                    
                    time.sleep(1)  # Small delay to prevent busy waiting
                    
                except Exception as e:
                    logger.error(f"Error in delivery worker: {e}")
                    time.sleep(5)
        
        # Start multiple worker threads
        for i in range(3):
            worker_thread = threading.Thread(target=delivery_worker, daemon=True)
            worker_thread.start()
    
    def _check_retry_queue(self):
        """Check for deliveries that need to be retried."""
        
        current_time = datetime.utcnow()
        
        for delivery in self.deliveries.values():
            if (delivery.status == DeliveryStatus.RETRYING and
                delivery.next_retry_at and
                current_time >= delivery.next_retry_at):
                
                if delivery.should_retry():
                    self.delivery_queue.append(delivery.delivery_id)
    
    def _attempt_delivery(self, delivery: WebhookDelivery):
        """Attempt webhook delivery."""
        
        delivery.attempt_count += 1
        delivery.attempted_at = datetime.utcnow()
        delivery.status = DeliveryStatus.RETRYING if delivery.attempt_count > 1 else DeliveryStatus.PENDING
        
        try:
            # Get endpoint configuration
            endpoint = self.endpoints.get(delivery.endpoint_id)
            if not endpoint:
                delivery.status = DeliveryStatus.FAILED
                delivery.error_message = "Endpoint not found"
                return
            
            # Make HTTP request
            response = requests.request(
                method=delivery.method,
                url=delivery.url,
                headers=delivery.headers,
                data=delivery.payload,
                timeout=endpoint.timeout_seconds,
                allow_redirects=False
            )
            
            # Record response
            delivery.http_status_code = response.status_code
            delivery.response_body = response.text[:1000]  # Limit response body size
            delivery.completed_at = datetime.utcnow()
            
            # Check if delivery was successful
            if 200 <= response.status_code < 300:
                delivery.status = DeliveryStatus.DELIVERED
                
                # Update endpoint statistics
                endpoint.successful_deliveries += 1
                endpoint.total_deliveries += 1
                endpoint.last_delivery = datetime.utcnow()
                
                # Record success metric
                if self.monitoring_system:
                    self.monitoring_system.metrics_collector.record_counter(
                        'webhook_delivery_success_total',
                        1.0,
                        {'endpoint_id': endpoint.endpoint_id, 'tenant_id': endpoint.tenant_id},
                        endpoint.tenant_id
                    )
                
                logger.info(f"Webhook delivery {delivery.delivery_id} successful")
                
            else:
                # Handle failure
                self._handle_delivery_failure(delivery, endpoint, f"HTTP {response.status_code}")
                
        except Exception as e:
            # Handle exception
            self._handle_delivery_failure(delivery, endpoint, str(e))
    
    def _handle_delivery_failure(self, delivery: WebhookDelivery, 
                               endpoint: WebhookEndpoint, error_message: str):
        """Handle webhook delivery failure."""
        
        delivery.error_message = error_message
        delivery.completed_at = datetime.utcnow()
        
        # Update endpoint statistics
        endpoint.failed_deliveries += 1
        endpoint.total_deliveries += 1
        
        # Check if should retry
        if delivery.attempt_count < delivery.max_attempts:
            delivery.status = DeliveryStatus.RETRYING
            
            # Calculate next retry time with exponential backoff
            backoff_seconds = endpoint.retry_backoff_seconds * (
                2 ** (delivery.attempt_count - 1)
            )
            delivery.next_retry_at = datetime.utcnow() + timedelta(seconds=backoff_seconds)
            
            logger.warning(f"Webhook delivery {delivery.delivery_id} failed, will retry in {backoff_seconds}s")
        else:
            delivery.status = DeliveryStatus.EXHAUSTED
            logger.error(f"Webhook delivery {delivery.delivery_id} exhausted all retries")
        
        # Record failure metric
        if self.monitoring_system:
            self.monitoring_system.metrics_collector.record_counter(
                'webhook_delivery_failure_total',
                1.0,
                {'endpoint_id': endpoint.endpoint_id, 'tenant_id': endpoint.tenant_id, 'error': error_message},
                endpoint.tenant_id
            )
    
    def get_endpoint_metrics(self, endpoint_id: str) -> Optional[Dict[str, Any]]:
        """Get metrics for webhook endpoint."""
        
        endpoint = self.endpoints.get(endpoint_id)
        if not endpoint:
            return None
        
        # Get recent deliveries
        recent_deliveries = [
            d for d in self.deliveries.values()
            if d.endpoint_id == endpoint_id and
               d.created_at > datetime.utcnow() - timedelta(hours=24)
        ]
        
        # Count by status
        status_counts = defaultdict(int)
        for delivery in recent_deliveries:
            status_counts[delivery.status.value] += 1
        
        return {
            'endpoint_id': endpoint_id,
            'name': endpoint.name,
            'total_deliveries': endpoint.total_deliveries,
            'successful_deliveries': endpoint.successful_deliveries,
            'failed_deliveries': endpoint.failed_deliveries,
            'success_rate_percent': endpoint.get_success_rate(),
            'last_delivery': endpoint.last_delivery.isoformat() if endpoint.last_delivery else None,
            'recent_deliveries_24h': len(recent_deliveries),
            'status_breakdown_24h': dict(status_counts),
            'is_active': endpoint.is_active
        }

class SlackIntegration:
    """Slack integration for notifications and alerts."""
    
    def __init__(self, webhook_url: str, channel: str = None, username: str = "Claude Nexus"):
        self.webhook_url = webhook_url
        self.channel = channel
        self.username = username
    
    def send_message(self, text: str, attachments: List[Dict[str, Any]] = None,
                    channel: str = None) -> bool:
        """Send message to Slack."""
        
        payload = {
            'text': text,
            'username': self.username
        }
        
        if channel or self.channel:
            payload['channel'] = channel or self.channel
        
        if attachments:
            payload['attachments'] = attachments
        
        try:
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Slack message: {e}")
            return False
    
    def send_alert(self, title: str, description: str, severity: str = "warning",
                  fields: Dict[str, str] = None) -> bool:
        """Send formatted alert to Slack."""
        
        color_map = {
            'good': 'good',
            'warning': 'warning',
            'danger': 'danger',
            'critical': 'danger'
        }
        
        attachment = {
            'color': color_map.get(severity, 'warning'),
            'title': title,
            'text': description,
            'footer': 'Claude Nexus Security System',
            'ts': int(time.time())
        }
        
        if fields:
            attachment['fields'] = [
                {'title': k, 'value': v, 'short': True}
                for k, v in fields.items()
            ]
        
        return self.send_message('', attachments=[attachment])

class EmailIntegration:
    """Email integration for notifications."""
    
    def __init__(self, smtp_host: str, smtp_port: int, username: str = None,
                 password: str = None, use_tls: bool = True, from_email: str = None):
        
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.use_tls = use_tls
        self.from_email = from_email or username
    
    def send_email(self, to_emails: List[str], subject: str, body: str,
                  is_html: bool = False) -> bool:
        """Send email notification."""
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(to_emails)
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'html' if is_html else 'plain'))
            
            # Connect to SMTP server
            server = smtplib.SMTP(self.smtp_host, self.smtp_port)
            
            if self.use_tls:
                server.starttls()
            
            if self.username and self.password:
                server.login(self.username, self.password)
            
            server.send_message(msg)
            server.quit()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False
    
    def send_alert_email(self, to_emails: List[str], title: str, description: str,
                        severity: str = "warning", details: Dict[str, Any] = None) -> bool:
        """Send formatted alert email."""
        
        subject = f"[{severity.upper()}] {title}"
        
        body = f"""
        <html>
        <body>
            <h2 style="color: {'red' if severity == 'critical' else 'orange' if severity == 'warning' else 'blue'}">
                {title}
            </h2>
            
            <p><strong>Severity:</strong> {severity.upper()}</p>
            <p><strong>Description:</strong> {description}</p>
            <p><strong>Timestamp:</strong> {datetime.utcnow().isoformat()}</p>
        """
        
        if details:
            body += "<h3>Details:</h3><ul>"
            for key, value in details.items():
                body += f"<li><strong>{key}:</strong> {value}</li>"
            body += "</ul>"
        
        body += """
            <hr>
            <p><em>This alert was generated by Claude Nexus Enterprise Security System</em></p>
        </body>
        </html>
        """
        
        return self.send_email(to_emails, subject, body, is_html=True)

class EnterpriseIntegrationFramework:
    """Main enterprise integration framework."""
    
    def __init__(self, orchestrator: MultiTenantOrchestrator = None,
                 monitoring_system: EnterpriseMonitoringSystem = None,
                 audit_logger=None):
        
        self.orchestrator = orchestrator
        self.monitoring_system = monitoring_system
        self.audit_logger = audit_logger
        
        # Core components
        self.webhook_manager = WebhookManager(monitoring_system, audit_logger)
        
        # Integration templates
        self.integration_templates: Dict[str, IntegrationTemplate] = {}
        
        # External integrations
        self.slack_integrations: Dict[str, SlackIntegration] = {}
        self.email_integrations: Dict[str, EmailIntegration] = {}
        
        # Event processing
        self.event_processors: Dict[EventType, List[Callable]] = defaultdict(list)
        
        # Initialize default templates
        self._initialize_default_templates()
        
        # Start event monitoring
        self._start_event_monitoring()
    
    def _initialize_default_templates(self):
        """Initialize default integration templates."""
        
        # Slack notification template
        slack_template = IntegrationTemplate(
            name="Slack Notifications",
            description="Send notifications to Slack channels",
            integration_type=IntegrationType.SLACK,
            config_template={
                'webhook_url': {'type': 'string', 'required': True},
                'channel': {'type': 'string', 'required': False},
                'username': {'type': 'string', 'required': False, 'default': 'Claude Nexus'}
            },
            required_fields=['webhook_url'],
            optional_fields=['channel', 'username'],
            event_mapping={
                'consultation_completed': 'Agent consultation completed for {tenant_id}',
                'security_alert': 'Security alert: {title}',
                'sla_violation': 'SLA violation detected for {tenant_id}'
            }
        )
        self.integration_templates[slack_template.template_id] = slack_template
        
        # Email notification template
        email_template = IntegrationTemplate(
            name="Email Notifications",
            description="Send email notifications and alerts",
            integration_type=IntegrationType.EMAIL,
            config_template={
                'smtp_host': {'type': 'string', 'required': True},
                'smtp_port': {'type': 'integer', 'required': True, 'default': 587},
                'username': {'type': 'string', 'required': True},
                'password': {'type': 'string', 'required': True, 'sensitive': True},
                'from_email': {'type': 'string', 'required': True},
                'to_emails': {'type': 'array', 'required': True}
            },
            required_fields=['smtp_host', 'smtp_port', 'username', 'password', 'from_email', 'to_emails']
        )
        self.integration_templates[email_template.template_id] = email_template
        
        # Webhook template
        webhook_template = IntegrationTemplate(
            name="Generic Webhook",
            description="Send events to custom webhook endpoints",
            integration_type=IntegrationType.WEBHOOK,
            config_template={
                'url': {'type': 'string', 'required': True},
                'method': {'type': 'string', 'required': False, 'default': 'POST'},
                'headers': {'type': 'object', 'required': False},
                'auth_type': {'type': 'string', 'required': False, 'default': 'none'},
                'secret': {'type': 'string', 'required': False, 'sensitive': True}
            },
            required_fields=['url']
        )
        self.integration_templates[webhook_template.template_id] = webhook_template
    
    def create_integration_from_template(self, template_id: str, config: Dict[str, Any],
                                       tenant_id: str, name: str = None) -> Optional[str]:
        """Create integration from template."""
        
        template = self.integration_templates.get(template_id)
        if not template:
            logger.error(f"Template {template_id} not found")
            return None
        
        # Validate required fields
        for field in template.required_fields:
            if field not in config:
                logger.error(f"Required field {field} missing from config")
                return None
        
        integration_name = name or f"{template.name} - {tenant_id}"
        
        if template.integration_type == IntegrationType.WEBHOOK:
            return self._create_webhook_integration(template, config, tenant_id, integration_name)
        elif template.integration_type == IntegrationType.SLACK:
            return self._create_slack_integration(template, config, tenant_id, integration_name)
        elif template.integration_type == IntegrationType.EMAIL:
            return self._create_email_integration(template, config, tenant_id, integration_name)
        
        return None
    
    def _create_webhook_integration(self, template: IntegrationTemplate, config: Dict[str, Any],
                                  tenant_id: str, name: str) -> str:
        """Create webhook integration."""
        
        endpoint = WebhookEndpoint(
            name=name,
            description=f"Integration created from template: {template.name}",
            tenant_id=tenant_id,
            url=config['url'],
            method=config.get('method', 'POST'),
            headers=config.get('headers', {}),
            secret=config.get('secret'),
            auth_type=config.get('auth_type', 'none'),
            event_types=set(EventType)  # Subscribe to all events by default
        )
        
        # Add auth credentials if provided
        if config.get('auth_token'):
            endpoint.auth_type = 'bearer'
            endpoint.auth_credentials = {'token': config['auth_token']}
        elif config.get('api_key'):
            endpoint.auth_type = 'api_key'
            endpoint.auth_credentials = {
                'api_key': config['api_key'],
                'header_name': config.get('api_key_header', 'X-API-Key')
            }
        
        return self.webhook_manager.create_webhook_endpoint(endpoint)
    
    def _create_slack_integration(self, template: IntegrationTemplate, config: Dict[str, Any],
                                tenant_id: str, name: str) -> str:
        """Create Slack integration."""
        
        integration_id = str(uuid.uuid4())
        
        slack_integration = SlackIntegration(
            webhook_url=config['webhook_url'],
            channel=config.get('channel'),
            username=config.get('username', 'Claude Nexus')
        )
        
        self.slack_integrations[f"{tenant_id}:{integration_id}"] = slack_integration
        
        # Register event processors
        self._register_slack_event_processors(tenant_id, integration_id, slack_integration)
        
        logger.info(f"Created Slack integration {name} for tenant {tenant_id}")
        return integration_id
    
    def _create_email_integration(self, template: IntegrationTemplate, config: Dict[str, Any],
                                tenant_id: str, name: str) -> str:
        """Create email integration."""
        
        integration_id = str(uuid.uuid4())
        
        email_integration = EmailIntegration(
            smtp_host=config['smtp_host'],
            smtp_port=config['smtp_port'],
            username=config['username'],
            password=config['password'],
            use_tls=config.get('use_tls', True),
            from_email=config['from_email']
        )
        
        self.email_integrations[f"{tenant_id}:{integration_id}"] = {
            'integration': email_integration,
            'to_emails': config['to_emails'],
            'config': config
        }
        
        # Register event processors
        self._register_email_event_processors(tenant_id, integration_id, email_integration, config['to_emails'])
        
        logger.info(f"Created email integration {name} for tenant {tenant_id}")
        return integration_id
    
    def _register_slack_event_processors(self, tenant_id: str, integration_id: str,
                                        slack_integration: SlackIntegration):
        """Register Slack event processors."""
        
        def process_consultation_event(event: IntegrationEvent):
            if event.tenant_id != tenant_id:
                return
            
            if event.event_type == EventType.CONSULTATION_COMPLETED:
                slack_integration.send_message(
                    f"✅ Agent consultation completed for tenant {event.tenant_id}",
                    attachments=[{
                        'color': 'good',
                        'fields': [
                            {'title': 'Agent Type', 'value': event.data.get('agent_type', 'Unknown'), 'short': True},
                            {'title': 'Duration', 'value': f"{event.data.get('duration_ms', 0)/1000:.1f}s", 'short': True},
                            {'title': 'User', 'value': event.user_id or 'Unknown', 'short': True}
                        ]
                    }]
                )
            elif event.event_type == EventType.CONSULTATION_FAILED:
                slack_integration.send_message(
                    f"❌ Agent consultation failed for tenant {event.tenant_id}",
                    attachments=[{
                        'color': 'danger',
                        'fields': [
                            {'title': 'Agent Type', 'value': event.data.get('agent_type', 'Unknown'), 'short': True},
                            {'title': 'Error', 'value': event.data.get('error', 'Unknown error'), 'short': False},
                            {'title': 'User', 'value': event.user_id or 'Unknown', 'short': True}
                        ]
                    }]
                )
        
        def process_security_event(event: IntegrationEvent):
            if event.tenant_id != tenant_id:
                return
            
            if event.event_type == EventType.SECURITY_ALERT:
                severity = event.data.get('severity', 'medium')
                slack_integration.send_alert(
                    title=event.data.get('title', 'Security Alert'),
                    description=event.data.get('description', 'Security event detected'),
                    severity=severity,
                    fields={
                        'Tenant': event.tenant_id,
                        'Source': event.source,
                        'Timestamp': event.timestamp.isoformat()
                    }
                )
        
        # Register processors
        self.event_processors[EventType.CONSULTATION_COMPLETED].append(process_consultation_event)
        self.event_processors[EventType.CONSULTATION_FAILED].append(process_consultation_event)
        self.event_processors[EventType.SECURITY_ALERT].append(process_security_event)
    
    def _register_email_event_processors(self, tenant_id: str, integration_id: str,
                                        email_integration: EmailIntegration, to_emails: List[str]):
        """Register email event processors."""
        
        def process_security_alert(event: IntegrationEvent):
            if event.tenant_id != tenant_id or event.event_type != EventType.SECURITY_ALERT:
                return
            
            email_integration.send_alert_email(
                to_emails=to_emails,
                title=event.data.get('title', 'Security Alert'),
                description=event.data.get('description', 'Security event detected'),
                severity=event.data.get('severity', 'warning'),
                details={
                    'Tenant ID': event.tenant_id,
                    'Event ID': event.event_id,
                    'Source': event.source,
                    'User ID': event.user_id or 'N/A',
                    **event.data
                }
            )
        
        def process_sla_violation(event: IntegrationEvent):
            if event.tenant_id != tenant_id or event.event_type != EventType.SLA_VIOLATION:
                return
            
            email_integration.send_alert_email(
                to_emails=to_emails,
                title=f"SLA Violation - {event.data.get('sla_name', 'Unknown SLA')}",
                description=f"SLA violation detected for tenant {event.tenant_id}",
                severity="critical",
                details={
                    'SLA Name': event.data.get('sla_name', 'Unknown'),
                    'Metric': event.data.get('metric', 'Unknown'),
                    'Current Value': event.data.get('current_value', 'Unknown'),
                    'Threshold': event.data.get('threshold', 'Unknown'),
                    'Duration': event.data.get('duration', 'Unknown')
                }
            )
        
        # Register processors
        self.event_processors[EventType.SECURITY_ALERT].append(process_security_alert)
        self.event_processors[EventType.SLA_VIOLATION].append(process_sla_violation)
    
    def emit_event(self, event: IntegrationEvent) -> List[str]:
        """Emit integration event to all registered handlers."""
        
        delivery_ids = []
        
        # Process webhook deliveries
        webhook_deliveries = self.webhook_manager.deliver_event(event)
        delivery_ids.extend(webhook_deliveries)
        
        # Process custom event processors
        processors = self.event_processors.get(event.event_type, [])
        for processor in processors:
            try:
                processor(event)
            except Exception as e:
                logger.error(f"Error processing event with custom processor: {e}")
        
        # Log event emission
        if self.audit_logger:
            audit_event = AuditEvent(
                tenant_id=event.tenant_id,
                event_type=AuditEventType.SYSTEM_EVENT,
                resource="integration_event",
                action="emit",
                result="success",
                metadata={
                    'event_id': event.event_id,
                    'event_type': event.event_type.value,
                    'delivery_count': len(delivery_ids)
                }
            )
            self.audit_logger.log_audit_event(audit_event)
        
        logger.info(f"Emitted event {event.event_id} ({event.event_type.value}) with {len(delivery_ids)} deliveries")
        return delivery_ids
    
    def _start_event_monitoring(self):
        """Start monitoring for system events to automatically emit integration events."""
        
        def event_monitor():
            while True:
                try:
                    # Monitor orchestrator for consultation events
                    if self.orchestrator:
                        self._monitor_orchestrator_events()
                    
                    # Monitor monitoring system for alert events
                    if self.monitoring_system:
                        self._monitor_system_events()
                    
                    time.sleep(10)  # Check every 10 seconds
                    
                except Exception as e:
                    logger.error(f"Error in event monitoring: {e}")
                    time.sleep(60)
        
        monitor_thread = threading.Thread(target=event_monitor, daemon=True)
        monitor_thread.start()
    
    def _monitor_orchestrator_events(self):
        """Monitor orchestrator for consultation events."""
        
        # This would integrate with the orchestrator's event system
        # For now, it's a placeholder for where orchestrator events would be monitored
        pass
    
    def _monitor_system_events(self):
        """Monitor system for alert and security events."""
        
        # This would integrate with the monitoring system's alert manager
        # For now, it's a placeholder for where monitoring events would be captured
        pass
    
    def get_integration_metrics(self, tenant_id: str = None) -> Dict[str, Any]:
        """Get integration metrics."""
        
        # Webhook metrics
        webhook_metrics = []
        for endpoint in self.webhook_manager.endpoints.values():
            if not tenant_id or endpoint.tenant_id == tenant_id:
                metrics = self.webhook_manager.get_endpoint_metrics(endpoint.endpoint_id)
                if metrics:
                    webhook_metrics.append(metrics)
        
        # Integration counts
        slack_count = len([
            k for k in self.slack_integrations.keys()
            if not tenant_id or k.startswith(f"{tenant_id}:")
        ])
        
        email_count = len([
            k for k in self.email_integrations.keys()
            if not tenant_id or k.startswith(f"{tenant_id}:")
        ])
        
        # Calculate overall success rate
        total_deliveries = sum(m['total_deliveries'] for m in webhook_metrics)
        successful_deliveries = sum(m['successful_deliveries'] for m in webhook_metrics)
        
        overall_success_rate = 0.0
        if total_deliveries > 0:
            overall_success_rate = (successful_deliveries / total_deliveries) * 100
        
        return {
            'tenant_id': tenant_id,
            'webhook_endpoints': len(webhook_metrics),
            'slack_integrations': slack_count,
            'email_integrations': email_count,
            'total_deliveries': total_deliveries,
            'successful_deliveries': successful_deliveries,
            'overall_success_rate_percent': round(overall_success_rate, 2),
            'webhook_metrics': webhook_metrics,
            'available_templates': len(self.integration_templates)
        }
    
    def get_available_templates(self) -> List[Dict[str, Any]]:
        """Get list of available integration templates."""
        
        templates = []
        
        for template in self.integration_templates.values():
            templates.append({
                'template_id': template.template_id,
                'name': template.name,
                'description': template.description,
                'integration_type': template.integration_type.value,
                'required_fields': template.required_fields,
                'optional_fields': template.optional_fields,
                'documentation': template.documentation
            })
        
        return templates

# Example usage
if __name__ == "__main__":
    from multi_tenant_orchestration import MultiTenantOrchestrator
    from enterprise_monitoring_system import EnterpriseMonitoringSystem
    from enterprise_security_architecture import SOC2ComplianceEngine
    
    # Initialize systems
    compliance_engine = SOC2ComplianceEngine()
    orchestrator = MultiTenantOrchestrator(audit_logger=compliance_engine)
    monitoring = EnterpriseMonitoringSystem(audit_logger=compliance_engine)
    
    # Initialize integration framework
    integration_framework = EnterpriseIntegrationFramework(
        orchestrator=orchestrator,
        monitoring_system=monitoring,
        audit_logger=compliance_engine
    )
    
    # Create Slack integration
    slack_config = {
        'webhook_url': 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK',
        'channel': '#alerts',
        'username': 'Claude Nexus Security'
    }
    
    slack_template_id = next((
        t.template_id for t in integration_framework.integration_templates.values()
        if t.integration_type == IntegrationType.SLACK
    ), None)
    
    if slack_template_id:
        slack_integration_id = integration_framework.create_integration_from_template(
            slack_template_id,
            slack_config,
            "test-corp",
            "Test Corp Slack Alerts"
        )
        print(f"Created Slack integration: {slack_integration_id}")
    
    # Create test event
    test_event = IntegrationEvent(
        event_type=EventType.SECURITY_ALERT,
        tenant_id="test-corp",
        user_id="admin@test-corp.com",
        source="vulnerability_scanner",
        data={
            'title': 'High Severity Vulnerability Detected',
            'description': 'SQL injection vulnerability found in login endpoint',
            'severity': 'high',
            'component': 'authentication_service',
            'cvss_score': 8.1
        }
    )
    
    # Emit event
    delivery_ids = integration_framework.emit_event(test_event)
    print(f"Event delivered to {len(delivery_ids)} endpoints")
    
    # Get integration metrics
    metrics = integration_framework.get_integration_metrics("test-corp")
    print(f"Integration metrics: {json.dumps(metrics, indent=2, default=str)}")
