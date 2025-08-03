#!/usr/bin/env python3
"""
Rate Limiting and Resource Management System for Claude Nexus
Provides sustainable operation with intelligent resource allocation
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
import threading
from collections import defaultdict, deque
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResourceType(Enum):
    API_CALLS = "api_calls"
    AGENT_CONSULTATIONS = "agent_consultations"
    WORKFLOW_EXECUTIONS = "workflow_executions"
    GITHUB_REQUESTS = "github_requests"
    MONITORING_CHECKS = "monitoring_checks"
    ALERT_NOTIFICATIONS = "alert_notifications"

class RateLimitStrategy(Enum):
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    ADAPTIVE = "adaptive"

class ResourcePriority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class RateLimitConfig:
    resource_type: ResourceType
    strategy: RateLimitStrategy
    max_requests: int
    time_window: int  # seconds
    burst_capacity: int
    priority: ResourcePriority
    enabled: bool = True

@dataclass
class ResourceUsage:
    resource_type: ResourceType
    timestamp: datetime
    count: int
    user_id: Optional[str] = None
    operation: Optional[str] = None
    metadata: Dict = None

@dataclass
class ResourceQuota:
    resource_type: ResourceType
    daily_limit: int
    hourly_limit: int
    burst_limit: int
    priority_multiplier: float = 1.0

class TokenBucket:
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate
        self.last_refill = time.time()
        self.lock = threading.Lock()
    
    def consume(self, tokens: int = 1) -> bool:
        with self.lock:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def _refill(self):
        now = time.time()
        tokens_to_add = (now - self.last_refill) * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now
    
    def available_tokens(self) -> int:
        with self.lock:
            self._refill()
            return int(self.tokens)

class SlidingWindowLimiter:
    def __init__(self, max_requests: int, window_size: int):
        self.max_requests = max_requests
        self.window_size = window_size
        self.requests = deque()
        self.lock = threading.Lock()
    
    def can_proceed(self) -> bool:
        with self.lock:
            now = time.time()
            # Remove old requests outside the window
            while self.requests and self.requests[0] <= now - self.window_size:
                self.requests.popleft()
            
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            return False
    
    def current_usage(self) -> int:
        with self.lock:
            now = time.time()
            while self.requests and self.requests[0] <= now - self.window_size:
                self.requests.popleft()
            return len(self.requests)

class ResourceManager:
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path) 
        self.rate_limiters: Dict[str, any] = {}
        self.usage_history: List[ResourceUsage] = []
        self.quotas: Dict[ResourceType, ResourceQuota] = {}
        self.resource_stats: Dict[ResourceType, Dict] = defaultdict(dict)
        self._initialize_limiters()
        self._initialize_quotas()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load rate limiting configuration"""
        default_config = {
            "rate_limits": {
                "agent_consultations": {
                    "strategy": "token_bucket",
                    "max_requests": 100,
                    "time_window": 3600,
                    "burst_capacity": 20,
                    "priority": "high",
                    "enabled": True
                },
                "github_requests": {
                    "strategy": "sliding_window", 
                    "max_requests": 5000,
                    "time_window": 3600,
                    "burst_capacity": 100,
                    "priority": "critical",
                    "enabled": True
                },
                "workflow_executions": {
                    "strategy": "adaptive",
                    "max_requests": 50,
                    "time_window": 3600,
                    "burst_capacity": 10,
                    "priority": "medium",
                    "enabled": True
                },
                "monitoring_checks": {
                    "strategy": "fixed_window",
                    "max_requests": 200,
                    "time_window": 3600,
                    "burst_capacity": 50,
                    "priority": "medium",
                    "enabled": True
                },
                "alert_notifications": {
                    "strategy": "token_bucket",
                    "max_requests": 20,
                    "time_window": 3600,
                    "burst_capacity": 5,
                    "priority": "critical",
                    "enabled": True
                }
            },
            "quotas": {
                "daily_limits": {
                    "agent_consultations": 1000,
                    "workflow_executions": 500,
                    "github_requests": 40000,
                    "monitoring_checks": 2000
                },
                "hourly_limits": {
                    "agent_consultations": 100,
                    "workflow_executions": 50,
                    "github_requests": 5000,
                    "monitoring_checks": 200
                }
            },
            "adaptive_scaling": {
                "enabled": True,
                "load_threshold": 0.8,
                "scale_factor": 1.5,
                "cooldown_period": 300
            },
            "resource_priorities": {
                "critical": 1.0,
                "high": 0.8,
                "medium": 0.6,
                "low": 0.4
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
            except Exception as e:
                logger.warning(f"Failed to load rate limit config from {config_path}: {e}")
        
        return default_config
    
    def _initialize_limiters(self):
        """Initialize rate limiters based on configuration"""
        for resource_name, config in self.config['rate_limits'].items():
            if not config.get('enabled', True):
                continue
                
            resource_type = ResourceType(resource_name)
            strategy = RateLimitStrategy(config['strategy'])
            
            limiter_key = resource_type.value
            
            if strategy == RateLimitStrategy.TOKEN_BUCKET:
                capacity = config['burst_capacity']
                refill_rate = config['max_requests'] / config['time_window']
                self.rate_limiters[limiter_key] = TokenBucket(capacity, refill_rate)
                
            elif strategy == RateLimitStrategy.SLIDING_WINDOW:
                self.rate_limiters[limiter_key] = SlidingWindowLimiter(
                    config['max_requests'], config['time_window']
                )
            
            # For fixed_window and adaptive, we'll use sliding window as base
            else:
                self.rate_limiters[limiter_key] = SlidingWindowLimiter(
                    config['max_requests'], config['time_window']
                )
    
    def _initialize_quotas(self):
        """Initialize resource quotas"""
        daily_limits = self.config['quotas']['daily_limits']
        hourly_limits = self.config['quotas']['hourly_limits']
        
        for resource_name in daily_limits:
            try:
                resource_type = ResourceType(resource_name)
                self.quotas[resource_type] = ResourceQuota(
                    resource_type=resource_type,
                    daily_limit=daily_limits[resource_name],
                    hourly_limit=hourly_limits.get(resource_name, daily_limits[resource_name] // 24),
                    burst_limit=self.config['rate_limits'][resource_name]['burst_capacity']
                )
            except (ValueError, KeyError):
                logger.warning(f"Invalid resource type in quotas: {resource_name}")
    
    def can_proceed(self, resource_type: ResourceType, count: int = 1, 
                   priority: ResourcePriority = ResourcePriority.MEDIUM,
                   user_id: str = None) -> Tuple[bool, str]:
        """Check if resource usage can proceed based on rate limits and quotas"""
        
        # Check if rate limiting is enabled for this resource
        limiter_key = resource_type.value
        if limiter_key not in self.rate_limiters:
            return True, "No rate limit configured"
        
        # Check quota limits first
        quota_check, quota_message = self._check_quotas(resource_type, count)
        if not quota_check:
            return False, quota_message
        
        # Apply priority-based adjustment
        priority_multiplier = self.config['resource_priorities'].get(priority.value, 1.0)
        effective_count = max(1, int(count / priority_multiplier))
        
        # Check rate limiter
        limiter = self.rate_limiters[limiter_key]
        
        if isinstance(limiter, TokenBucket):
            can_proceed = limiter.consume(effective_count)
            remaining = limiter.available_tokens()
        elif isinstance(limiter, SlidingWindowLimiter):
            can_proceed = limiter.can_proceed()
            remaining = self.config['rate_limits'][limiter_key]['max_requests'] - limiter.current_usage()
        else:
            can_proceed = True
            remaining = float('inf')
        
        if can_proceed:
            # Record usage
            self._record_usage(resource_type, count, user_id)
            return True, f"Request allowed. Remaining capacity: {remaining}"
        else:
            return False, f"Rate limit exceeded for {resource_type.value}. Remaining: {remaining}"
    
    def _check_quotas(self, resource_type: ResourceType, count: int) -> Tuple[bool, str]:
        """Check if request is within quota limits"""
        if resource_type not in self.quotas:
            return True, "No quota configured"
        
        quota = self.quotas[resource_type]
        now = datetime.now()
        
        # Check daily quota
        daily_usage = self._get_usage_count(resource_type, now - timedelta(days=1))
        if daily_usage + count > quota.daily_limit:
            return False, f"Daily quota exceeded: {daily_usage}/{quota.daily_limit}"
        
        # Check hourly quota  
        hourly_usage = self._get_usage_count(resource_type, now - timedelta(hours=1))
        if hourly_usage + count > quota.hourly_limit:
            return False, f"Hourly quota exceeded: {hourly_usage}/{quota.hourly_limit}"
        
        return True, "Within quota limits"
    
    def _get_usage_count(self, resource_type: ResourceType, since: datetime) -> int:
        """Get usage count for resource type since specified time"""
        return sum(usage.count for usage in self.usage_history 
                  if usage.resource_type == resource_type and usage.timestamp >= since)
    
    def _record_usage(self, resource_type: ResourceType, count: int, user_id: str = None,
                     operation: str = None, metadata: Dict = None):
        """Record resource usage"""
        usage = ResourceUsage(
            resource_type=resource_type,
            timestamp=datetime.now(),
            count=count,
            user_id=user_id,
            operation=operation,
            metadata=metadata or {}
        )
        
        self.usage_history.append(usage)
        
        # Cleanup old usage records (keep last 7 days)
        cutoff = datetime.now() - timedelta(days=7)
        self.usage_history = [u for u in self.usage_history if u.timestamp >= cutoff]
    
    def get_usage_statistics(self, resource_type: ResourceType = None) -> Dict:
        """Get comprehensive usage statistics"""
        now = datetime.now()
        
        if resource_type:
            resource_types = [resource_type]
        else:
            resource_types = list(ResourceType)
        
        statistics = {}
        
        for rt in resource_types:
            # Usage in different time windows
            last_hour = self._get_usage_count(rt, now - timedelta(hours=1))
            last_day = self._get_usage_count(rt, now - timedelta(days=1))
            last_week = self._get_usage_count(rt, now - timedelta(days=7))
            
            # Rate limiter status
            limiter_status = {}
            limiter_key = rt.value
            if limiter_key in self.rate_limiters:
                limiter = self.rate_limiters[limiter_key] 
                if isinstance(limiter, TokenBucket):
                    limiter_status = {
                        'type': 'token_bucket',
                        'available_tokens': limiter.available_tokens(),
                        'capacity': limiter.capacity
                    }
                elif isinstance(limiter, SlidingWindowLimiter):
                    limiter_status = {
                        'type': 'sliding_window',
                        'current_usage': limiter.current_usage(),
                        'max_requests': self.config['rate_limits'][limiter_key]['max_requests']
                    }
            
            # Quota status
            quota_status = {}
            if rt in self.quotas:
                quota = self.quotas[rt]
                quota_status = {
                    'daily_used': last_day,
                    'daily_limit': quota.daily_limit,
                    'daily_remaining': quota.daily_limit - last_day,
                    'hourly_used': last_hour,
                    'hourly_limit': quota.hourly_limit,
                    'hourly_remaining': quota.hourly_limit - last_hour
                }
            
            statistics[rt.value] = {
                'usage': {
                    'last_hour': last_hour,
                    'last_day': last_day,
                    'last_week': last_week
                },
                'rate_limiter': limiter_status,
                'quota': quota_status,
                'health_status': self._get_resource_health(rt)
            }
        
        return statistics
    
    def _get_resource_health(self, resource_type: ResourceType) -> str:
        """Determine resource health status"""
        if resource_type not in self.quotas:
            return "unknown"
        
        quota = self.quotas[resource_type]
        now = datetime.now()
        
        hourly_usage = self._get_usage_count(resource_type, now - timedelta(hours=1))
        daily_usage = self._get_usage_count(resource_type, now - timedelta(days=1))
        
        hourly_percentage = (hourly_usage / quota.hourly_limit) * 100
        daily_percentage = (daily_usage / quota.daily_limit) * 100
        
        max_percentage = max(hourly_percentage, daily_percentage)
        
        if max_percentage >= 90:
            return "critical"
        elif max_percentage >= 75:
            return "warning"
        elif max_percentage >= 50:
            return "moderate"
        else:
            return "healthy"
    
    def adjust_limits_adaptive(self, resource_type: ResourceType, load_factor: float):
        """Adaptively adjust rate limits based on system load"""
        if not self.config['adaptive_scaling']['enabled']:
            return
        
        threshold = self.config['adaptive_scaling']['load_threshold']
        scale_factor = self.config['adaptive_scaling']['scale_factor']
        
        limiter_key = resource_type.value
        if limiter_key not in self.rate_limiters:
            return
        
        if load_factor > threshold:
            # Increase capacity under high load
            limiter = self.rate_limiters[limiter_key]
            
            if isinstance(limiter, TokenBucket):
                new_capacity = int(limiter.capacity * scale_factor)
                limiter.capacity = min(new_capacity, limiter.capacity * 2)  # Cap at 2x
                logger.info(f"Increased {resource_type.value} token bucket capacity to {limiter.capacity}")
            
            elif isinstance(limiter, SlidingWindowLimiter):
                current_config = self.config['rate_limits'][limiter_key]
                new_max = int(current_config['max_requests'] * scale_factor)
                # Create new limiter with increased capacity
                self.rate_limiters[limiter_key] = SlidingWindowLimiter(
                    min(new_max, current_config['max_requests'] * 2),
                    current_config['time_window']
                )
                logger.info(f"Increased {resource_type.value} sliding window limit")
    
    def reset_limits(self, resource_type: ResourceType = None):
        """Reset rate limits for specified resource type or all"""
        if resource_type:
            limiter_key = resource_type.value
            if limiter_key in self.rate_limiters:
                self._reinitialize_limiter(resource_type)
        else:
            self._initialize_limiters()
        
        logger.info(f"Rate limits reset for {resource_type.value if resource_type else 'all resources'}")
    
    def _reinitialize_limiter(self, resource_type: ResourceType):
        """Reinitialize a specific rate limiter"""
        limiter_key = resource_type.value
        config = self.config['rate_limits'][limiter_key]
        strategy = RateLimitStrategy(config['strategy'])
        
        if strategy == RateLimitStrategy.TOKEN_BUCKET:
            capacity = config['burst_capacity']
            refill_rate = config['max_requests'] / config['time_window']
            self.rate_limiters[limiter_key] = TokenBucket(capacity, refill_rate)
        else:
            self.rate_limiters[limiter_key] = SlidingWindowLimiter(
                config['max_requests'], config['time_window']
            )
    
    def get_resource_recommendations(self) -> List[Dict]:
        """Get recommendations for resource optimization"""
        recommendations = []
        
        for resource_type in ResourceType:
            health = self._get_resource_health(resource_type)
            stats = self.get_usage_statistics(resource_type)
            
            if health == "critical":
                recommendations.append({
                    'resource': resource_type.value,
                    'priority': 'high',
                    'recommendation': 'Immediate attention required - resource usage critical',
                    'action': 'Increase quotas or optimize usage patterns'
                })
            elif health == "warning":
                recommendations.append({
                    'resource': resource_type.value,
                    'priority': 'medium', 
                    'recommendation': 'Monitor closely - approaching limits',
                    'action': 'Consider increasing limits or reducing usage'
                })
            
            # Check for unused resources
            rt_stats = stats.get(resource_type.value, {})
            usage = rt_stats.get('usage', {})
            if usage.get('last_day', 0) == 0:
                recommendations.append({
                    'resource': resource_type.value,
                    'priority': 'low',
                    'recommendation': 'Unused resource - consider reducing quotas',
                    'action': 'Optimize resource allocation'
                })
        
        return recommendations

# Context manager for resource tracking
class ResourceContext:
    def __init__(self, resource_manager: ResourceManager, resource_type: ResourceType, 
                 count: int = 1, priority: ResourcePriority = ResourcePriority.MEDIUM,
                 operation: str = None):
        self.resource_manager = resource_manager
        self.resource_type = resource_type
        self.count = count
        self.priority = priority
        self.operation = operation
        self.allowed = False
    
    def __enter__(self):
        can_proceed, message = self.resource_manager.can_proceed(
            self.resource_type, self.count, self.priority
        )
        
        if not can_proceed:
            raise ResourceExhaustedError(f"Resource limit exceeded: {message}")
        
        self.allowed = True
        logger.debug(f"Resource acquired: {self.resource_type.value} ({self.count})")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.allowed:
            logger.debug(f"Resource released: {self.resource_type.value}")
        return False

class ResourceExhaustedError(Exception):
    """Raised when resource limits are exceeded"""
    pass

def main():
    """Main CLI interface for resource management"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Claude Nexus Resource Management System')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--statistics', action='store_true', help='Show usage statistics')
    parser.add_argument('--resource-type', choices=[rt.value for rt in ResourceType], 
                       help='Specific resource type for statistics')
    parser.add_argument('--test-limit', choices=[rt.value for rt in ResourceType],
                       help='Test rate limiting for resource type')
    parser.add_argument('--reset-limits', choices=[rt.value for rt in ResourceType],
                       help='Reset rate limits for resource type')
    parser.add_argument('--recommendations', action='store_true', help='Get optimization recommendations')
    parser.add_argument('--output', choices=['json', 'github-actions'], default='json')
    
    args = parser.parse_args()
    
    resource_manager = ResourceManager(args.config)
    
    if args.statistics:
        resource_type = ResourceType(args.resource_type) if args.resource_type else None
        stats = resource_manager.get_usage_statistics(resource_type)
        
        if args.output == 'github-actions':
            print(f"::set-output name=resource_statistics::{json.dumps(stats)}")
            
            # Output health status for each resource
            for resource, data in stats.items():
                health = data.get('health_status', 'unknown')
                print(f"::set-output name={resource}_health::{health}")
        else:
            print(json.dumps(stats, indent=2, default=str))
    
    elif args.test_limit:
        resource_type = ResourceType(args.test_limit)
        
        # Test rate limiting
        success_count = 0
        total_attempts = 20
        
        for i in range(total_attempts):
            can_proceed, message = resource_manager.can_proceed(resource_type)
            if can_proceed:
                success_count += 1
            time.sleep(0.1)  # Small delay between requests
        
        print(f"Rate limit test for {resource_type.value}:")
        print(f"Successful requests: {success_count}/{total_attempts}")
        print(f"Success rate: {(success_count/total_attempts)*100:.1f}%")
    
    elif args.reset_limits:
        resource_type = ResourceType(args.reset_limits)
        resource_manager.reset_limits(resource_type)
        print(f"Rate limits reset for {resource_type.value}")
    
    elif args.recommendations:
        recommendations = resource_manager.get_resource_recommendations()
        if args.output == 'github-actions':
            print(f"::set-output name=recommendations::{json.dumps(recommendations)}")
        else:
            print(json.dumps(recommendations, indent=2))
    
    else:
        print("Claude Nexus Resource Management System - Ready for enterprise operations")
        print("Use --statistics to view usage statistics")
        print("Use --test-limit to test rate limiting")
        print("Use --recommendations to get optimization recommendations")

if __name__ == '__main__':
    main()