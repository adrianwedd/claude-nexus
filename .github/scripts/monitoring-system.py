#!/usr/bin/env python3
"""
Comprehensive Monitoring and Alerting System for Claude Nexus
Provides enterprise-grade operational excellence with SRE practices
"""

import json
import os
import sys
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class ServiceStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class MetricData:
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str]
    threshold_critical: Optional[float] = None
    threshold_warning: Optional[float] = None

@dataclass
class Alert:
    id: str
    severity: AlertSeverity
    title: str
    description: str
    service: str
    timestamp: datetime
    resolved: bool = False
    acknowledged: bool = False
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class MonitoringSystem:
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.alerts: List[Alert] = []
        self.metrics: Dict[str, List[MetricData]] = {}
        self.service_status: Dict[str, ServiceStatus] = {}
        
    def _load_config(self, config_path: str) -> Dict:
        """Load monitoring configuration"""
        default_config = {
            "monitoring": {
                "enabled": True,
                "interval_seconds": 300,  # 5 minutes
                "retention_days": 30
            },
            "alerting": {
                "enabled": True,
                "channels": {
                    "github_issues": True,
                    "slack": False,
                    "email": False
                }
            },
            "thresholds": {
                "agent_response_time": {
                    "warning": 30,
                    "critical": 60
                },
                "workflow_success_rate": {
                    "warning": 90,
                    "critical": 80
                },
                "error_rate": {
                    "warning": 5,
                    "critical": 10
                }
            },
            "services": {
                "agent-consultation": {
                    "enabled": True,
                    "slo_availability": 99.9,
                    "slo_latency_p95": 30
                },
                "quality-gates": {
                    "enabled": True,
                    "slo_availability": 99.5,
                    "slo_latency_p95": 60
                },
                "health-checks": {
                    "enabled": True,
                    "slo_availability": 99.0,
                    "slo_latency_p95": 120
                }
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults
                    default_config.update(loaded_config)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
        
        return default_config
    
    def collect_github_metrics(self) -> List[MetricData]:
        """Collect metrics from GitHub API"""
        metrics = []
        github_token = os.getenv('GITHUB_TOKEN')
        repo = os.getenv('GITHUB_REPOSITORY', 'adrianwedd/claude-nexus')
        
        if not github_token:
            logger.warning("GITHUB_TOKEN not available, skipping GitHub metrics")
            return metrics
        
        headers = {
            'Authorization': f'token {github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        base_url = f'https://api.github.com/repos/{repo}'
        
        try:
            # Workflow runs metrics
            workflows_response = requests.get(
                f'{base_url}/actions/runs?per_page=100',
                headers=headers,
                timeout=30
            )
            
            if workflows_response.status_code == 200:
                workflows_data = workflows_response.json()
                
                # Calculate success rate
                total_runs = len(workflows_data['workflow_runs'])
                successful_runs = sum(1 for run in workflows_data['workflow_runs'] 
                                    if run['conclusion'] == 'success')
                
                if total_runs > 0:
                    success_rate = (successful_runs / total_runs) * 100
                    metrics.append(MetricData(
                        name='workflow_success_rate',
                        value=success_rate,
                        timestamp=datetime.now(),
                        tags={'service': 'github-actions'},
                        threshold_warning=self.config['thresholds']['workflow_success_rate']['warning'],
                        threshold_critical=self.config['thresholds']['workflow_success_rate']['critical']
                    ))
                
                # Calculate average duration
                recent_runs = [run for run in workflows_data['workflow_runs'][:20]]
                if recent_runs:
                    durations = []
                    for run in recent_runs:
                        if run.get('created_at') and run.get('updated_at'):
                            created = datetime.fromisoformat(run['created_at'].replace('Z', '+00:00'))
                            updated = datetime.fromisoformat(run['updated_at'].replace('Z', '+00:00'))
                            duration = (updated - created).total_seconds()
                            durations.append(duration)
                    
                    if durations:
                        avg_duration = sum(durations) / len(durations)
                        metrics.append(MetricData(
                            name='workflow_avg_duration',
                            value=avg_duration,
                            timestamp=datetime.now(),
                            tags={'service': 'github-actions'},
                            threshold_warning=1800,  # 30 minutes
                            threshold_critical=3600  # 1 hour
                        ))
            
            # Repository health metrics
            repo_response = requests.get(base_url, headers=headers, timeout=30)
            if repo_response.status_code == 200:
                repo_data = repo_response.json()
                
                # Open issues count
                metrics.append(MetricData(
                    name='open_issues_count',
                    value=repo_data.get('open_issues_count', 0),
                    timestamp=datetime.now(),
                    tags={'service': 'repository'},
                    threshold_warning=50,
                    threshold_critical=100
                ))
                
                # Repository size
                metrics.append(MetricData(
                    name='repository_size_kb',
                    value=repo_data.get('size', 0),
                    timestamp=datetime.now(),
                    tags={'service': 'repository'},
                    threshold_warning=100000,  # 100MB
                    threshold_critical=500000  # 500MB
                ))
        
        except requests.RequestException as e:
            logger.error(f"Failed to collect GitHub metrics: {e}")
            self._create_alert(
                "github_api_error",
                AlertSeverity.HIGH,
                "GitHub API Collection Failed",
                f"Failed to collect metrics from GitHub API: {e}",
                "monitoring"
            )
        
        return metrics
    
    def collect_workflow_metrics(self) -> List[MetricData]:
        """Collect metrics from workflow execution history"""
        metrics = []
        
        # Simulate agent consultation metrics
        # In production, these would come from actual workflow logs
        agent_response_times = {
            'interface-artisan': 25.5,
            'performance-virtuoso': 32.1,
            'fortress-guardian': 28.9,
            'cloud-navigator': 35.2,
            'deployment-commander': 41.3,
            'reliability-engineer': 38.7
        }
        
        for agent, response_time in agent_response_times.items():
            metrics.append(MetricData(
                name='agent_response_time',
                value=response_time,
                timestamp=datetime.now(),
                tags={'agent': agent, 'service': 'agent-consultation'},
                threshold_warning=self.config['thresholds']['agent_response_time']['warning'],
                threshold_critical=self.config['thresholds']['agent_response_time']['critical']
            ))
        
        # Quality gates metrics
        quality_gate_success_rates = {
            'security': 98.5,
            'performance': 95.2,
            'code-quality': 97.8,
            'testing': 94.1
        }
        
        for gate, success_rate in quality_gate_success_rates.items():
            metrics.append(MetricData(
                name='quality_gate_success_rate',
                value=success_rate,
                timestamp=datetime.now(),
                tags={'gate': gate, 'service': 'quality-gates'},
                threshold_warning=95.0,
                threshold_critical=90.0
            ))
        
        return metrics
    
    def collect_system_metrics(self) -> List[MetricData]:
        """Collect system-level metrics"""
        metrics = []
        
        # Simulate system resource metrics
        # In production, these would come from actual system monitoring
        system_metrics = {
            'cpu_usage_percent': 45.2,
            'memory_usage_percent': 67.8,
            'disk_usage_percent': 34.5,
            'network_latency_ms': 12.3
        }
        
        thresholds = {
            'cpu_usage_percent': {'warning': 70, 'critical': 85},
            'memory_usage_percent': {'warning': 80, 'critical': 90},
            'disk_usage_percent': {'warning': 75, 'critical': 85},
            'network_latency_ms': {'warning': 50, 'critical': 100}
        }
        
        for metric_name, value in system_metrics.items():
            threshold = thresholds.get(metric_name, {})
            metrics.append(MetricData(
                name=metric_name,
                value=value,
                timestamp=datetime.now(),
                tags={'service': 'system'},
                threshold_warning=threshold.get('warning'),
                threshold_critical=threshold.get('critical')
            ))
        
        return metrics
    
    def analyze_metrics(self, metrics: List[MetricData]) -> List[Alert]:
        """Analyze metrics and generate alerts"""
        alerts = []
        
        for metric in metrics:
            # Check critical thresholds
            if metric.threshold_critical is not None:
                if metric.value >= metric.threshold_critical:
                    alert_id = f"{metric.name}_critical_{int(time.time())}"
                    alerts.append(Alert(
                        id=alert_id,
                        severity=AlertSeverity.CRITICAL,
                        title=f"Critical: {metric.name.replace('_', ' ').title()}",
                        description=f"{metric.name} is {metric.value} (critical threshold: {metric.threshold_critical})",
                        service=metric.tags.get('service', 'unknown'),
                        timestamp=metric.timestamp,
                        metadata={
                            'metric_name': metric.name,
                            'metric_value': metric.value,
                            'threshold': metric.threshold_critical,
                            'tags': metric.tags
                        }
                    ))
            
            # Check warning thresholds
            elif metric.threshold_warning is not None:
                if metric.value >= metric.threshold_warning:
                    alert_id = f"{metric.name}_warning_{int(time.time())}"
                    alerts.append(Alert(
                        id=alert_id,
                        severity=AlertSeverity.MEDIUM,
                        title=f"Warning: {metric.name.replace('_', ' ').title()}",
                        description=f"{metric.name} is {metric.value} (warning threshold: {metric.threshold_warning})",
                        service=metric.tags.get('service', 'unknown'),
                        timestamp=metric.timestamp,
                        metadata={
                            'metric_name': metric.name,
                            'metric_value': metric.value,
                            'threshold': metric.threshold_warning,
                            'tags': metric.tags
                        }
                    ))
        
        return alerts
    
    def _create_alert(self, alert_id: str, severity: AlertSeverity, title: str, 
                     description: str, service: str, metadata: Dict = None):
        """Create a new alert"""
        alert = Alert(
            id=alert_id,
            severity=severity,
            title=title,
            description=description,
            service=service,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        self.alerts.append(alert)
        return alert
    
    def update_service_status(self, metrics: List[MetricData]):
        """Update service status based on metrics"""
        service_metrics = {}
        
        # Group metrics by service
        for metric in metrics:
            service = metric.tags.get('service', 'unknown')
            if service not in service_metrics:
                service_metrics[service] = []
            service_metrics[service].append(metric)
        
        # Determine status for each service
        for service, service_metric_list in service_metrics.items():
            critical_issues = 0
            warning_issues = 0
            
            for metric in service_metric_list:
                if (metric.threshold_critical is not None and 
                    metric.value >= metric.threshold_critical):
                    critical_issues += 1
                elif (metric.threshold_warning is not None and 
                      metric.value >= metric.threshold_warning):
                    warning_issues += 1
            
            # Determine overall service status
            if critical_issues > 0:
                self.service_status[service] = ServiceStatus.UNHEALTHY
            elif warning_issues > 0:
                self.service_status[service] = ServiceStatus.DEGRADED
            else:
                self.service_status[service] = ServiceStatus.HEALTHY
    
    def send_github_alert(self, alert: Alert) -> bool:
        """Send alert as GitHub issue"""
        github_token = os.getenv('GITHUB_TOKEN')
        repo = os.getenv('GITHUB_REPOSITORY', 'adrianwedd/claude-nexus')
        
        if not github_token:
            logger.warning("GITHUB_TOKEN not available, cannot send GitHub alert")
            return False
        
        headers = {
            'Authorization': f'token {github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        # Create issue body
        body = f"""# {alert.title}

**Severity**: {alert.severity.value.upper()}
**Service**: {alert.service}
**Timestamp**: {alert.timestamp.isoformat()}

## Description
{alert.description}

## Metadata
```json
{json.dumps(alert.metadata, indent=2, default=str)}
```

## Remediation Steps
${{
    alert.severity == AlertSeverity.CRITICAL and '''
1. **IMMEDIATE ACTION REQUIRED**
2. Check service health dashboards
3. Review recent deployments
4. Escalate to on-call engineer if needed
5. Implement temporary mitigation if possible
''' or alert.severity == AlertSeverity.HIGH and '''
1. **HIGH PRIORITY** - Address within 2 hours
2. Investigate root cause
3. Implement fix or mitigation
4. Monitor for resolution
''' or '''
1. **MEDIUM PRIORITY** - Address within 24 hours
2. Investigate during business hours
3. Plan remediation steps
4. Monitor trends
'''
}}

---
*🚨 Generated by Claude Nexus Monitoring System*
*🎭 Automated alert from specialized monitoring agents*
"""
        
        # Determine labels
        labels = ['alert', alert.severity.value, f'service-{alert.service}']
        if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]:
            labels.append('urgent')
        
        try:
            response = requests.post(
                f'https://api.github.com/repos/{repo}/issues',
                headers=headers,
                json={
                    'title': f'🚨 {alert.title}',
                    'body': body,
                    'labels': labels
                },
                timeout=30
            )
            
            if response.status_code == 201:
                logger.info(f"GitHub alert created successfully: {alert.id}")
                return True
            else:
                logger.error(f"Failed to create GitHub alert: {response.status_code} - {response.text}")
                return False
                
        except requests.RequestException as e:
            logger.error(f"Failed to send GitHub alert: {e}")
            return False
    
    def generate_monitoring_report(self) -> Dict:
        """Generate comprehensive monitoring report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_metrics': len([m for metric_list in self.metrics.values() for m in metric_list]),
                'active_alerts': len([a for a in self.alerts if not a.resolved]),
                'services_monitored': len(self.service_status),
                'healthy_services': len([s for s in self.service_status.values() if s == ServiceStatus.HEALTHY]),
                'degraded_services': len([s for s in self.service_status.values() if s == ServiceStatus.DEGRADED]),
                'unhealthy_services': len([s for s in self.service_status.values() if s == ServiceStatus.UNHEALTHY])
            },
            'service_status': {
                service: status.value for service, status in self.service_status.items()
            },
            'alerts': [
                {
                    'id': alert.id,
                    'severity': alert.severity.value,
                    'title': alert.title,
                    'service': alert.service,
                    'timestamp': alert.timestamp.isoformat(),
                    'resolved': alert.resolved
                }
                for alert in self.alerts[-10:]  # Last 10 alerts
            ],
            'metrics_summary': {}
        }
        
        # Add metrics summary
        for metric_name, metric_list in self.metrics.items():
            if metric_list:
                latest_metric = max(metric_list, key=lambda x: x.timestamp)
                report['metrics_summary'][metric_name] = {
                    'latest_value': latest_metric.value,
                    'timestamp': latest_metric.timestamp.isoformat(),
                    'tags': latest_metric.tags
                }
        
        return report
    
    def run_monitoring_cycle(self):
        """Run a complete monitoring cycle"""
        logger.info("Starting monitoring cycle...")
        
        # Collect all metrics
        all_metrics = []
        all_metrics.extend(self.collect_github_metrics())
        all_metrics.extend(self.collect_workflow_metrics())
        all_metrics.extend(self.collect_system_metrics())
        
        # Store metrics
        for metric in all_metrics:
            if metric.name not in self.metrics:
                self.metrics[metric.name] = []
            self.metrics[metric.name].append(metric)
        
        # Analyze metrics and generate alerts
        new_alerts = self.analyze_metrics(all_metrics)
        self.alerts.extend(new_alerts)
        
        # Update service status
        self.update_service_status(all_metrics)
        
        # Send alerts
        if self.config['alerting']['enabled']:
            for alert in new_alerts:
                if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]:
                    if self.config['alerting']['channels']['github_issues']:
                        self.send_github_alert(alert)
        
        # Generate report
        report = self.generate_monitoring_report()
        
        logger.info(f"Monitoring cycle completed: {len(all_metrics)} metrics collected, "
                   f"{len(new_alerts)} new alerts generated")
        
        return report

def main():
    """Main CLI interface for monitoring system"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Claude Nexus Monitoring System')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--output', choices=['json', 'github-actions'], default='json')
    parser.add_argument('--cycle', action='store_true', help='Run monitoring cycle')
    parser.add_argument('--report-only', action='store_true', help='Generate report only')
    
    args = parser.parse_args()
    
    monitor = MonitoringSystem(args.config)
    
    if args.cycle:
        report = monitor.run_monitoring_cycle()
    else:
        report = monitor.generate_monitoring_report()
    
    if args.output == 'github-actions':
        # Output for GitHub Actions
        print(f"::set-output name=monitoring_report::{json.dumps(report)}")
        print(f"::set-output name=total_alerts::{len(monitor.alerts)}")
        print(f"::set-output name=critical_alerts::{len([a for a in monitor.alerts if a.severity == AlertSeverity.CRITICAL])}")
        print(f"::set-output name=service_status::{json.dumps({s: st.value for s, st in monitor.service_status.items()})}")
    else:
        print(json.dumps(report, indent=2, default=str))

if __name__ == '__main__':
    main()