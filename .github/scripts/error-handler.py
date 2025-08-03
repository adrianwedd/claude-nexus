#!/usr/bin/env python3
"""
Enterprise-Grade Error Handling and Fallback System for Claude Nexus
Provides bulletproof operational excellence with comprehensive error recovery
"""

import json
import os
import sys
import time
import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from enum import Enum
import requests
from functools import wraps
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class ErrorCategory(Enum):
    NETWORK = "network"
    API = "api"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    VALIDATION = "validation"
    SYSTEM = "system"
    AGENT = "agent"
    WORKFLOW = "workflow"

class FallbackStrategy(Enum):
    RETRY = "retry"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    FAILOVER = "failover"
    CACHE_FALLBACK = "cache_fallback"
    DEFAULT_RESPONSE = "default_response"

@dataclass
class ErrorContext:
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    timestamp: datetime
    service: str
    operation: str
    metadata: Dict[str, Any]
    stack_trace: Optional[str] = None
    retry_count: int = 0
    resolved: bool = False

@dataclass  
class FallbackConfig:
    strategy: FallbackStrategy
    max_retries: int = 3
    retry_delay: float = 1.0
    exponential_backoff: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60
    graceful_response: Optional[Dict] = None

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    def can_execute(self) -> bool:
        if self.state == 'CLOSED':
            return True
        elif self.state == 'OPEN':
            if time.time() - self.last_failure_time > self.timeout:
                self.state = 'HALF_OPEN'
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def record_success(self):
        self.failure_count = 0
        self.state = 'CLOSED'
    
    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'

class ErrorHandler:
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.error_history: List[ErrorContext] = []
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.fallback_cache: Dict[str, Any] = {}
        
    def _load_config(self, config_path: str) -> Dict:
        """Load error handling configuration"""
        default_config = {
            "error_handling": {
                "enabled": True,
                "max_error_history": 1000,
                "error_reporting": {
                    "github_issues": True,
                    "logging": True,
                    "metrics": True
                }
            },
            "fallback_strategies": {
                "agent_consultation": {
                    "strategy": "retry",
                    "max_retries": 3,
                    "retry_delay": 2.0,
                    "exponential_backoff": True,
                    "circuit_breaker_threshold": 5,
                    "graceful_response": {
                        "status": "degraded",
                        "message": "Agent consultation temporarily unavailable",
                        "fallback_agent": "repository-surgeon"
                    }
                },
                "quality_gates": {
                    "strategy": "graceful_degradation",
                    "max_retries": 2,
                    "graceful_response": {
                        "status": "warning",
                        "message": "Quality gates running in degraded mode",
                        "gates_skipped": ["performance", "security"]
                    }
                },
                "github_api": {
                    "strategy": "retry",
                    "max_retries": 5,
                    "retry_delay": 1.0,
                    "exponential_backoff": True,
                    "circuit_breaker_threshold": 10
                },
                "monitoring": {
                    "strategy": "cache_fallback",
                    "max_retries": 2,
                    "graceful_response": {
                        "status": "cached",
                        "message": "Using cached monitoring data"
                    }
                }
            },
            "error_thresholds": {
                "critical_error_rate": 10,  # errors per hour
                "circuit_breaker_threshold": 5,
                "alert_threshold": 3
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
            except Exception as e:
                logger.warning(f"Failed to load error config from {config_path}: {e}")
        
        return default_config
    
    def categorize_error(self, error: Exception, context: Dict = None) -> ErrorCategory:
        """Categorize error based on type and context"""
        error_str = str(error).lower()
        error_type = type(error).__name__
        
        if isinstance(error, requests.exceptions.ConnectionError):
            return ErrorCategory.NETWORK
        elif isinstance(error, requests.exceptions.Timeout):
            return ErrorCategory.TIMEOUT  
        elif isinstance(error, requests.exceptions.HTTPError):
            if hasattr(error, 'response') and error.response.status_code == 401:
                return ErrorCategory.AUTHENTICATION
            elif hasattr(error, 'response') and error.response.status_code == 429:
                return ErrorCategory.RATE_LIMIT
            else:
                return ErrorCategory.API
        elif 'validation' in error_str or 'invalid' in error_str:
            return ErrorCategory.VALIDATION
        elif 'agent' in error_str:
            return ErrorCategory.AGENT
        elif 'workflow' in error_str:
            return ErrorCategory.WORKFLOW
        else:
            return ErrorCategory.SYSTEM
    
    def determine_severity(self, error: Exception, category: ErrorCategory, context: Dict = None) -> ErrorSeverity:
        """Determine error severity based on error type and context"""
        if category in [ErrorCategory.AUTHENTICATION, ErrorCategory.SYSTEM]:
            return ErrorSeverity.CRITICAL
        elif category in [ErrorCategory.API, ErrorCategory.NETWORK, ErrorCategory.AGENT]:
            return ErrorSeverity.HIGH
        elif category in [ErrorCategory.TIMEOUT, ErrorCategory.RATE_LIMIT]:
            return ErrorSeverity.MEDIUM  
        else:
            return ErrorSeverity.LOW
    
    def create_error_context(self, error: Exception, service: str, operation: str, 
                           metadata: Dict = None) -> ErrorContext:
        """Create comprehensive error context"""
        error_id = f"{service}_{operation}_{int(time.time())}"
        category = self.categorize_error(error, metadata)
        severity = self.determine_severity(error, category, metadata)
        
        return ErrorContext(
            error_id=error_id,
            category=category,
            severity=severity,
            message=str(error),
            timestamp=datetime.now(),
            service=service,
            operation=operation,
            metadata=metadata or {},
            stack_trace=traceback.format_exc()
        )
    
    def get_fallback_strategy(self, service: str) -> FallbackConfig:
        """Get fallback configuration for service"""
        service_config = self.config['fallback_strategies'].get(service, {})
        
        return FallbackConfig(
            strategy=FallbackStrategy(service_config.get('strategy', 'retry')),
            max_retries=service_config.get('max_retries', 3),
            retry_delay=service_config.get('retry_delay', 1.0),
            exponential_backoff=service_config.get('exponential_backoff', True),
            circuit_breaker_threshold=service_config.get('circuit_breaker_threshold', 5),
            circuit_breaker_timeout=service_config.get('circuit_breaker_timeout', 60),
            graceful_response=service_config.get('graceful_response')
        )
    
    def get_circuit_breaker(self, service: str) -> CircuitBreaker:
        """Get or create circuit breaker for service"""
        if service not in self.circuit_breakers:
            config = self.get_fallback_strategy(service)
            self.circuit_breakers[service] = CircuitBreaker(
                failure_threshold=config.circuit_breaker_threshold,
                timeout=config.circuit_breaker_timeout
            )
        return self.circuit_breakers[service]
    
    def retry_with_backoff(self, func: Callable, *args, max_retries: int = 3, 
                          delay: float = 1.0, exponential: bool = True, **kwargs) -> Any:
        """Retry function with exponential backoff"""
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries:
                    raise e
                
                sleep_time = delay * (2 ** attempt if exponential else 1)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
    
    def execute_with_fallback(self, func: Callable, service: str, operation: str, 
                            *args, **kwargs) -> Dict[str, Any]:
        """Execute function with comprehensive error handling and fallback"""
        fallback_config = self.get_fallback_strategy(service)
        circuit_breaker = self.get_circuit_breaker(service)
        
        # Check circuit breaker
        if not circuit_breaker.can_execute():
            logger.warning(f"Circuit breaker OPEN for {service}. Using fallback response.")
            return self._get_fallback_response(service, operation, "circuit_breaker_open")
        
        try:
            # Execute with retry logic
            if fallback_config.strategy == FallbackStrategy.RETRY:
                result = self.retry_with_backoff(
                    func, *args,
                    max_retries=fallback_config.max_retries,
                    delay=fallback_config.retry_delay,
                    exponential=fallback_config.exponential_backoff,
                    **kwargs
                )
            else:
                result = func(*args, **kwargs)
            
            # Record success
            circuit_breaker.record_success()
            return {
                'status': 'success',
                'result': result,
                'service': service,
                'operation': operation
            }
            
        except Exception as error:
            # Create error context
            error_context = self.create_error_context(error, service, operation, kwargs)
            self.error_history.append(error_context)
            
            # Record failure in circuit breaker
            circuit_breaker.record_failure()
            
            # Apply fallback strategy
            return self._apply_fallback_strategy(error_context, fallback_config)
    
    def _apply_fallback_strategy(self, error_context: ErrorContext, 
                               config: FallbackConfig) -> Dict[str, Any]:
        """Apply specific fallback strategy"""
        service = error_context.service
        operation = error_context.operation
        
        if config.strategy == FallbackStrategy.GRACEFUL_DEGRADATION:
            return self._graceful_degradation(error_context, config)
        elif config.strategy == FallbackStrategy.CACHE_FALLBACK:
            return self._cache_fallback(error_context, config)
        elif config.strategy == FallbackStrategy.FAILOVER:
            return self._failover(error_context, config)
        elif config.strategy == FallbackStrategy.DEFAULT_RESPONSE:
            return self._default_response(error_context, config)
        else:
            # Default fallback
            return self._get_fallback_response(service, operation, "error")
    
    def _graceful_degradation(self, error_context: ErrorContext, 
                            config: FallbackConfig) -> Dict[str, Any]:
        """Implement graceful degradation"""
        response = config.graceful_response or {}
        response.update({
            'status': 'degraded',
            'error_id': error_context.error_id,
            'message': f"{error_context.service} operating in degraded mode",
            'original_error': error_context.message,
            'timestamp': error_context.timestamp.isoformat()
        })
        return response
    
    def _cache_fallback(self, error_context: ErrorContext, 
                       config: FallbackConfig) -> Dict[str, Any]:
        """Use cached response as fallback"""
        cache_key = f"{error_context.service}_{error_context.operation}"
        
        if cache_key in self.fallback_cache:
            cached_response = self.fallback_cache[cache_key]
            cached_response.update({
                'status': 'cached_fallback',
                'error_id': error_context.error_id,
                'message': 'Using cached response due to service error'
            })
            return cached_response
        else:
            return self._get_fallback_response(error_context.service, error_context.operation, "cache_miss")
    
    def _failover(self, error_context: ErrorContext, config: FallbackConfig) -> Dict[str, Any]:
        """Implement failover to backup service"""
        # In a real implementation, this would failover to a backup service
        return {
            'status': 'failover',
            'error_id': error_context.error_id,
            'message': f"Failed over from {error_context.service}",
            'fallback_service': 'backup_service',
            'timestamp': error_context.timestamp.isoformat()
        }
    
    def _default_response(self, error_context: ErrorContext, 
                         config: FallbackConfig) -> Dict[str, Any]:
        """Return configured default response"""
        response = config.graceful_response or {}
        response.update({
            'status': 'default_response',
            'error_id': error_context.error_id,
            'timestamp': error_context.timestamp.isoformat()
        })
        return response
    
    def _get_fallback_response(self, service: str, operation: str, reason: str) -> Dict[str, Any]:
        """Get generic fallback response"""
        fallback_responses = {
            'agent_consultation': {
                'status': 'fallback',
                'message': 'Agent consultation temporarily unavailable',
                'fallback_agent': 'repository-surgeon',
                'recommendations': ['Review issue manually', 'Check system status', 'Retry later']
            },
            'quality_gates': {
                'status': 'warning',
                'message': 'Quality gates running in safe mode',
                'gates_active': ['basic_validation'],
                'gates_skipped': ['advanced_security', 'performance_testing']
            },
            'monitoring': {
                'status': 'limited',
                'message': 'Monitoring operating in limited mode',
                'available_metrics': ['basic_health'],
                'unavailable_metrics': ['detailed_performance']
            }
        }
        
        return fallback_responses.get(service, {
            'status': 'error',
            'message': f'{service} temporarily unavailable',
            'reason': reason,
            'service': service,
            'operation': operation
        })
    
    def report_error(self, error_context: ErrorContext):
        """Report error through configured channels"""
        if not self.config['error_handling']['error_reporting']['github_issues']:
            return
        
        # Only report high severity errors as GitHub issues
        if error_context.severity not in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]:
            return
        
        try:
            self._create_github_error_issue(error_context)
        except Exception as e:
            logger.error(f"Failed to report error to GitHub: {e}")
    
    def _create_github_error_issue(self, error_context: ErrorContext):
        """Create GitHub issue for error reporting"""
        github_token = os.getenv('GITHUB_TOKEN')
        repo = os.getenv('GITHUB_REPOSITORY', 'adrianwedd/claude-nexus')
        
        if not github_token:
            logger.warning("GITHUB_TOKEN not available, cannot create error issue")
            return
        
        headers = {
            'Authorization': f'token {github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        # Create issue body
        body = f"""# 🚨 System Error Report

**Error ID**: `{error_context.error_id}`
**Severity**: {error_context.severity.value.upper()}
**Category**: {error_context.category.value}
**Service**: {error_context.service}
**Operation**: {error_context.operation}
**Timestamp**: {error_context.timestamp.isoformat()}

## 📋 Error Details

**Message**: {error_context.message}

**Metadata**:
```json
{json.dumps(error_context.metadata, indent=2, default=str)}
```

## 🔍 Stack Trace

```
{error_context.stack_trace or 'No stack trace available'}
```

## 🔧 Automated Recovery

- **Retry Count**: {error_context.retry_count}
- **Fallback Applied**: Yes
- **Circuit Breaker Status**: Monitored
- **Recovery Actions**: Automated fallback mechanisms engaged

## 🎯 Remediation Steps

{error_context.severity == ErrorSeverity.CRITICAL and '''### 🚨 CRITICAL - Immediate Action Required
1. **URGENT**: Investigate root cause immediately
2. Check system resources and dependencies
3. Review recent deployments or changes
4. Implement emergency fixes if needed
5. Escalate to on-call engineer
6. Monitor system recovery''' or '''### ⚠️ HIGH PRIORITY - Address within 2 hours
1. Investigate error pattern and frequency
2. Check service dependencies and health
3. Review error logs for additional context
4. Implement fix or enhanced fallback
5. Monitor for recurrence'''}

## 📊 Error Context

- **Service Health**: Check monitoring dashboard
- **Recent Changes**: Review deployment history  
- **Dependencies**: Validate external service status
- **Resource Usage**: Check system resource consumption

---

*🚨 Generated by Claude Nexus Error Handling System*
*🔧 Automated error detection and reporting*
"""
        
        # Determine labels
        labels = ['error-report', error_context.severity.value, error_context.category.value, f'service-{error_context.service}']
        if error_context.severity == ErrorSeverity.CRITICAL:
            labels.extend(['critical', 'urgent'])
        
        try:
            response = requests.post(
                f'https://api.github.com/repos/{repo}/issues',
                headers=headers,
                json={
                    'title': f'🚨 {error_context.severity.value.upper()}: {error_context.service} - {error_context.message[:100]}',
                    'body': body,
                    'labels': labels
                },
                timeout=30
            )
            
            if response.status_code == 201:
                logger.info(f"Error report created: {error_context.error_id}")
            else:
                logger.error(f"Failed to create error report: {response.status_code}")
                
        except requests.RequestException as e:
            logger.error(f"Failed to create GitHub error report: {e}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        if not self.error_history:
            return {'total_errors': 0, 'error_rate': 0}
        
        now = datetime.now()
        recent_errors = [e for e in self.error_history 
                        if (now - e.timestamp).total_seconds() < 3600]  # Last hour
        
        # Error counts by category
        category_counts = {}
        for error in self.error_history:
            category = error.category.value
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Error counts by severity
        severity_counts = {}
        for error in self.error_history:
            severity = error.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Circuit breaker status
        cb_status = {service: {'state': cb.state, 'failures': cb.failure_count} 
                    for service, cb in self.circuit_breakers.items()}
        
        return {
            'total_errors': len(self.error_history),
            'recent_errors_1h': len(recent_errors),
            'error_rate_per_hour': len(recent_errors),
            'errors_by_category': category_counts,
            'errors_by_severity': severity_counts,
            'circuit_breaker_status': cb_status,
            'most_common_error': max(category_counts.items(), key=lambda x: x[1])[0] if category_counts else None,
            'last_error_time': max(e.timestamp for e in self.error_history).isoformat() if self.error_history else None
        }

# Decorator for automatic error handling
def with_error_handling(service: str, operation: str):
    """Decorator to add automatic error handling to functions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            error_handler = ErrorHandler()
            return error_handler.execute_with_fallback(func, service, operation, *args, **kwargs)
        return wrapper
    return decorator

def main():
    """Main CLI interface for error handling system"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Claude Nexus Error Handling System')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--statistics', action='store_true', help='Show error statistics')
    parser.add_argument('--test-error', choices=['network', 'api', 'timeout'], help='Test error handling')
    parser.add_argument('--output', choices=['json', 'github-actions'], default='json')
    
    args = parser.parse_args()
    
    error_handler = ErrorHandler(args.config)
    
    if args.statistics:
        stats = error_handler.get_error_statistics()
        if args.output == 'github-actions':
            print(f"::set-output name=error_statistics::{json.dumps(stats)}")
            print(f"::set-output name=error_rate::{stats['error_rate_per_hour']}")
            print(f"::set-output name=circuit_breaker_status::{json.dumps(stats['circuit_breaker_status'])}")
        else:
            print(json.dumps(stats, indent=2, default=str))
    
    elif args.test_error:
        # Test error handling with simulated errors
        if args.test_error == 'network':
            error = requests.exceptions.ConnectionError("Test network error")
        elif args.test_error == 'api':
            error = requests.exceptions.HTTPError("Test API error")
        else:
            error = requests.exceptions.Timeout("Test timeout error")
        
        error_context = error_handler.create_error_context(error, "test_service", "test_operation")
        error_handler.error_history.append(error_context)
        error_handler.report_error(error_context)
        
        print(f"Test error created: {error_context.error_id}")
    
    else:
        print("Claude Nexus Error Handling System - Ready for enterprise operations")
        print("Use --statistics to view error statistics")
        print("Use --test-error to test error handling")

if __name__ == '__main__':
    main()