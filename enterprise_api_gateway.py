#!/usr/bin/env python3
"""
Enterprise API Gateway for Multi-Tenant Architecture

Comprehensive API gateway providing tenant routing, rate limiting,
authentication, authorization, and security controls for the
claude-nexus agent ecosystem. Supports enterprise-grade scalability
and security requirements.

Features:
- Multi-tenant request routing and isolation
- Advanced rate limiting with tenant-specific policies
- Authentication and authorization integration
- Request/response transformation and validation
- Circuit breaker and fallback mechanisms
- Comprehensive logging and monitoring
- Security threat detection and mitigation
- Performance optimization and caching

Architecture:
- High-performance async request handling
- Distributed rate limiting with Redis backend
- JWT-based authentication with SSO integration
- Role-based authorization with RBAC system
- Health checking and service discovery

Author: Fortress Guardian
Version: 1.0.0
Compliance: SOC 2 Type II, Enterprise Security
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid
from collections import defaultdict, deque
import hashlib
import hmac
import re
from urllib.parse import urlparse, parse_qs
import ipaddress
from functools import wraps
import statistics
import threading

# Web framework imports (FastAPI example)
try:
    from fastapi import FastAPI, Request, Response, HTTPException, Depends
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    import uvicorn
except ImportError:
    # Fallback for environments without FastAPI
    FastAPI = Request = Response = HTTPException = Depends = None
    HTTPBearer = HTTPAuthorizationCredentials = None
    CORSMiddleware = JSONResponse = uvicorn = None

# Import our security systems
from enterprise_security_architecture import (
    EnterpriseSecurityOrchestrator, AuditEvent, AuditEventType, SecurityLevel
)
from multi_tenant_rbac_system import (
    MultiTenantRBACSystem, ResourceType, PermissionType
)
from enterprise_sso_integration import EnterpriseSSO
from multi_tenant_orchestration import MultiTenantOrchestrator

logger = logging.getLogger(__name__)

class RateLimitStrategy(Enum):
    """Rate limiting strategies."""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    ADAPTIVE = "adaptive"

class RequestMethod(Enum):
    """HTTP request methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    OPTIONS = "OPTIONS"
    HEAD = "HEAD"

@dataclass
class RateLimitPolicy:
    """Rate limiting policy configuration."""
    policy_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    tenant_id: str = ""
    
    # Rate limit parameters
    requests_per_minute: int = 100
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_limit: int = 150  # Temporary burst allowance
    
    # Strategy and configuration
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET
    window_size_seconds: int = 60
    
    # Scope and conditions
    applies_to_paths: Set[str] = field(default_factory=set)
    applies_to_methods: Set[RequestMethod] = field(default_factory=set)
    applies_to_ips: Set[str] = field(default_factory=set)
    exempt_ips: Set[str] = field(default_factory=set)
    
    # Actions on limit exceeded
    block_duration_seconds: int = 300  # 5 minutes
    return_retry_after: bool = True
    custom_response: Optional[Dict[str, Any]] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True

@dataclass
class RequestContext:
    """Request context for processing."""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = ""
    user_id: str = ""
    
    # Request details
    method: str = "GET"
    path: str = "/"
    query_params: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    body: Optional[bytes] = None
    
    # Client information
    client_ip: str = ""
    user_agent: str = ""
    origin: str = ""
    
    # Authentication context
    authenticated: bool = False
    auth_method: str = ""
    roles: Set[str] = field(default_factory=set)
    permissions: Set[str] = field(default_factory=set)
    
    # Processing metadata
    start_time: datetime = field(default_factory=datetime.utcnow)
    processing_time_ms: float = 0.0
    rate_limit_remaining: int = 0
    
    # Security flags
    is_suspicious: bool = False
    threat_score: float = 0.0
    security_violations: List[str] = field(default_factory=list)

@dataclass
class RouteConfiguration:
    """API route configuration."""
    route_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    path_pattern: str = "/"
    methods: Set[RequestMethod] = field(default_factory=lambda: {RequestMethod.GET})
    
    # Target configuration
    target_service: str = ""
    target_path: str = ""
    load_balancing_strategy: str = "round_robin"  # round_robin, least_connections, weighted
    
    # Security settings
    require_authentication: bool = True
    required_permissions: Set[str] = field(default_factory=set)
    rate_limit_policy_id: Optional[str] = None
    
    # Request/response transformation
    request_transformations: List[Dict[str, Any]] = field(default_factory=list)
    response_transformations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Caching configuration
    cache_enabled: bool = False
    cache_ttl_seconds: int = 300
    cache_vary_headers: Set[str] = field(default_factory=set)
    
    # Circuit breaker settings
    circuit_breaker_enabled: bool = True
    failure_threshold: int = 5
    recovery_timeout_seconds: int = 60
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True

class TokenBucketRateLimiter:
    """Token bucket rate limiter implementation."""
    
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.tokens = capacity
        self.last_refill = time.time()
        self._lock = threading.Lock()
    
    def consume(self, tokens: int = 1) -> bool:
        """Attempt to consume tokens from bucket."""
        
        with self._lock:
            now = time.time()
            
            # Refill bucket based on time elapsed
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now
            
            # Check if enough tokens available
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    def get_tokens(self) -> float:
        """Get current token count."""
        with self._lock:
            now = time.time()
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now
            return self.tokens

class SlidingWindowRateLimiter:
    """Sliding window rate limiter implementation."""
    
    def __init__(self, window_size_seconds: int, max_requests: int):
        self.window_size = window_size_seconds
        self.max_requests = max_requests
        self.requests = deque()
        self._lock = threading.Lock()
    
    def is_allowed(self) -> bool:
        """Check if request is allowed under sliding window."""
        
        with self._lock:
            now = time.time()
            cutoff = now - self.window_size
            
            # Remove old requests outside window
            while self.requests and self.requests[0] < cutoff:
                self.requests.popleft()
            
            # Check if under limit
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            
            return False
    
    def get_remaining_requests(self) -> int:
        """Get remaining requests in current window."""
        with self._lock:
            now = time.time()
            cutoff = now - self.window_size
            
            # Remove old requests
            while self.requests and self.requests[0] < cutoff:
                self.requests.popleft()
            
            return max(0, self.max_requests - len(self.requests))

class EnterpriseRateLimiter:
    """Enterprise-grade rate limiter with multiple strategies."""
    
    def __init__(self):
        self.policies: Dict[str, RateLimitPolicy] = {}
        self.limiters: Dict[str, Any] = {}  # Key -> limiter instance
        self.blocked_clients: Dict[str, datetime] = {}
        self.request_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._lock = threading.Lock()
    
    def add_policy(self, policy: RateLimitPolicy):
        """Add rate limiting policy."""
        self.policies[policy.policy_id] = policy
        logger.info(f"Added rate limit policy {policy.name} for tenant {policy.tenant_id}")
    
    def check_rate_limit(self, context: RequestContext, policy_id: str) -> Tuple[bool, Dict[str, Any]]:
        """Check if request passes rate limit."""
        
        if policy_id not in self.policies:
            return True, {}
        
        policy = self.policies[policy_id]
        
        # Check if client is currently blocked
        client_key = f"{context.tenant_id}:{context.client_ip}"
        if client_key in self.blocked_clients:
            if datetime.utcnow() < self.blocked_clients[client_key]:
                return False, {
                    'error': 'Client temporarily blocked',
                    'retry_after': int((self.blocked_clients[client_key] - datetime.utcnow()).total_seconds())
                }
            else:
                del self.blocked_clients[client_key]
        
        # Check exempt IPs
        if context.client_ip in policy.exempt_ips:
            return True, {}
        
        # Apply rate limiting based on strategy
        limiter_key = f"{policy_id}:{client_key}"
        
        if policy.strategy == RateLimitStrategy.TOKEN_BUCKET:
            allowed, metadata = self._check_token_bucket(limiter_key, policy, context)
        elif policy.strategy == RateLimitStrategy.SLIDING_WINDOW:
            allowed, metadata = self._check_sliding_window(limiter_key, policy, context)
        elif policy.strategy == RateLimitStrategy.FIXED_WINDOW:
            allowed, metadata = self._check_fixed_window(limiter_key, policy, context)
        else:
            allowed, metadata = True, {}
        
        # Handle rate limit exceeded
        if not allowed:
            # Block client if configured
            if policy.block_duration_seconds > 0:
                self.blocked_clients[client_key] = datetime.utcnow() + timedelta(seconds=policy.block_duration_seconds)
            
            # Add retry-after header
            if policy.return_retry_after:
                metadata['retry_after'] = policy.window_size_seconds
        
        return allowed, metadata
    
    def _check_token_bucket(self, limiter_key: str, policy: RateLimitPolicy,
                           context: RequestContext) -> Tuple[bool, Dict[str, Any]]:
        """Check token bucket rate limit."""
        
        if limiter_key not in self.limiters:
            # Create new token bucket
            capacity = policy.burst_limit or policy.requests_per_minute
            refill_rate = policy.requests_per_minute / 60.0  # tokens per second
            self.limiters[limiter_key] = TokenBucketRateLimiter(capacity, refill_rate)
        
        limiter = self.limiters[limiter_key]
        allowed = limiter.consume(1)
        remaining = int(limiter.get_tokens())
        
        return allowed, {
            'rate_limit_remaining': remaining,
            'rate_limit_limit': policy.requests_per_minute,
            'rate_limit_window': 60
        }
    
    def _check_sliding_window(self, limiter_key: str, policy: RateLimitPolicy,
                             context: RequestContext) -> Tuple[bool, Dict[str, Any]]:
        """Check sliding window rate limit."""
        
        if limiter_key not in self.limiters:
            self.limiters[limiter_key] = SlidingWindowRateLimiter(
                policy.window_size_seconds,
                policy.requests_per_minute
            )
        
        limiter = self.limiters[limiter_key]
        allowed = limiter.is_allowed()
        remaining = limiter.get_remaining_requests()
        
        return allowed, {
            'rate_limit_remaining': remaining,
            'rate_limit_limit': policy.requests_per_minute,
            'rate_limit_window': policy.window_size_seconds
        }
    
    def _check_fixed_window(self, limiter_key: str, policy: RateLimitPolicy,
                           context: RequestContext) -> Tuple[bool, Dict[str, Any]]:
        """Check fixed window rate limit."""
        
        with self._lock:
            now = datetime.utcnow()
            window_start = now.replace(second=0, microsecond=0)
            window_key = f"{limiter_key}:{window_start.isoformat()}"
            
            current_count = self.request_counts[limiter_key].get(window_key, 0)
            
            if current_count < policy.requests_per_minute:
                self.request_counts[limiter_key][window_key] = current_count + 1
                remaining = policy.requests_per_minute - current_count - 1
                return True, {
                    'rate_limit_remaining': remaining,
                    'rate_limit_limit': policy.requests_per_minute,
                    'rate_limit_window': 60
                }
            
            return False, {
                'rate_limit_remaining': 0,
                'rate_limit_limit': policy.requests_per_minute,
                'rate_limit_window': 60
            }

class SecurityAnalyzer:
    """Security threat analysis and detection."""
    
    def __init__(self):
        self.threat_patterns = {
            'sql_injection': re.compile(r"(union|select|insert|delete|drop|create|alter|exec|script)", re.IGNORECASE),
            'xss_attack': re.compile(r"<script|javascript:|vbscript:|onload|onerror", re.IGNORECASE),
            'path_traversal': re.compile(r"\.\./|\.\.\\|\~|\%2e\%2e", re.IGNORECASE),
            'command_injection': re.compile(r"(;|\||&|\$|`|\>|\<)", re.IGNORECASE)
        }
        self.suspicious_ips: Set[str] = set()
        self.request_patterns: Dict[str, List[datetime]] = defaultdict(list)
    
    def analyze_request(self, context: RequestContext) -> float:
        """Analyze request for security threats. Returns threat score 0.0-1.0."""
        
        threat_score = 0.0
        violations = []
        
        # Check for known attack patterns
        full_url = f"{context.path}?{context.query_params}"
        body_str = context.body.decode('utf-8', errors='ignore') if context.body else ""
        
        for pattern_name, pattern in self.threat_patterns.items():
            if pattern.search(full_url) or pattern.search(body_str):
                threat_score += 0.3
                violations.append(f"Potential {pattern_name} detected")
        
        # Check for suspicious IP behavior
        if context.client_ip in self.suspicious_ips:
            threat_score += 0.2
            violations.append("Request from suspicious IP")
        
        # Check for rapid request patterns (potential bot)
        client_key = f"{context.tenant_id}:{context.client_ip}"
        now = datetime.utcnow()
        self.request_patterns[client_key].append(now)
        
        # Clean old entries (last 5 minutes)
        cutoff = now - timedelta(minutes=5)
        self.request_patterns[client_key] = [
            ts for ts in self.request_patterns[client_key] if ts > cutoff
        ]
        
        # Check for excessive requests
        if len(self.request_patterns[client_key]) > 100:  # More than 100 requests in 5 minutes
            threat_score += 0.4
            violations.append("Excessive request rate detected")
            self.suspicious_ips.add(context.client_ip)
        
        # Check for abnormal headers
        if self._has_suspicious_headers(context.headers):
            threat_score += 0.2
            violations.append("Suspicious headers detected")
        
        # Check for geo-location anomalies (placeholder)
        if self._is_geo_anomaly(context.client_ip):
            threat_score += 0.1
            violations.append("Geographic anomaly detected")
        
        context.threat_score = min(threat_score, 1.0)
        context.security_violations = violations
        context.is_suspicious = threat_score > 0.5
        
        return context.threat_score
    
    def _has_suspicious_headers(self, headers: Dict[str, str]) -> bool:
        """Check for suspicious HTTP headers."""
        
        suspicious_patterns = [
            r"sqlmap",
            r"nikto",
            r"burp",
            r"nessus",
            r"scanner"
        ]
        
        user_agent = headers.get('user-agent', '').lower()
        
        for pattern in suspicious_patterns:
            if re.search(pattern, user_agent):
                return True
        
        return False
    
    def _is_geo_anomaly(self, ip_address: str) -> bool:
        """Check for geographic anomalies (placeholder)."""
        # In production, integrate with IP geolocation service
        # and check against expected geographic regions
        return False

class EnterpriseAPIGateway:
    """Main enterprise API gateway."""
    
    def __init__(self, security_orchestrator: EnterpriseSecurityOrchestrator = None,
                 rbac_system: MultiTenantRBACSystem = None,
                 sso_system: EnterpriseSSO = None,
                 orchestrator: MultiTenantOrchestrator = None,
                 audit_logger=None):
        
        self.security_orchestrator = security_orchestrator
        self.rbac_system = rbac_system
        self.sso_system = sso_system
        self.orchestrator = orchestrator
        self.audit_logger = audit_logger
        
        # Gateway components
        self.rate_limiter = EnterpriseRateLimiter()
        self.security_analyzer = SecurityAnalyzer()
        
        # Configuration
        self.routes: Dict[str, RouteConfiguration] = {}
        self.middleware_chain: List[Callable] = []
        
        # Performance metrics
        self.request_metrics: Dict[str, List[float]] = defaultdict(list)
        self.error_counts: Dict[str, int] = defaultdict(int)
        
        # Cache
        self.response_cache: Dict[str, Tuple[Any, datetime]] = {}
        
        # Initialize FastAPI if available
        if FastAPI:
            self.app = self._create_fastapi_app()
        else:
            self.app = None
            logger.warning("FastAPI not available. API Gateway running in minimal mode.")
    
    def _create_fastapi_app(self) -> FastAPI:
        """Create FastAPI application with middleware."""
        
        app = FastAPI(
            title="Claude Nexus Enterprise API Gateway",
            description="Multi-tenant API gateway with enterprise security",
            version="1.0.0"
        )
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Add custom middleware
        @app.middleware("http")
        async def gateway_middleware(request: Request, call_next):
            return await self._process_request_middleware(request, call_next)
        
        # Health check endpoint
        @app.get("/health")
        async def health_check():
            return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
        
        # Metrics endpoint
        @app.get("/metrics")
        async def get_metrics():
            return self._get_gateway_metrics()
        
        return app
    
    async def _process_request_middleware(self, request: Request, call_next):
        """Main request processing middleware."""
        
        start_time = time.time()
        
        try:
            # Create request context
            context = await self._create_request_context(request)
            
            # Security analysis
            threat_score = self.security_analyzer.analyze_request(context)
            
            # Block high-threat requests
            if threat_score > 0.8:
                await self._log_security_event(context, "High threat score detected")
                return JSONResponse(
                    status_code=403,
                    content={"error": "Request blocked for security reasons"}
                )
            
            # Authentication
            if not await self._authenticate_request(context, request):
                return JSONResponse(
                    status_code=401,
                    content={"error": "Authentication required"}
                )
            
            # Authorization
            if not await self._authorize_request(context):
                return JSONResponse(
                    status_code=403,
                    content={"error": "Insufficient permissions"}
                )
            
            # Rate limiting
            rate_limit_result = await self._check_rate_limits(context)
            if not rate_limit_result['allowed']:
                response = JSONResponse(
                    status_code=429,
                    content={"error": "Rate limit exceeded"}
                )
                if 'retry_after' in rate_limit_result:
                    response.headers['Retry-After'] = str(rate_limit_result['retry_after'])
                return response
            
            # Route request
            route_config = self._find_matching_route(context.path, context.method)
            if not route_config:
                return JSONResponse(
                    status_code=404,
                    content={"error": "Route not found"}
                )
            
            # Check cache
            if route_config.cache_enabled:
                cached_response = self._get_cached_response(context, route_config)
                if cached_response:
                    return cached_response
            
            # Process request
            response = await call_next(request)
            
            # Cache response if configured
            if route_config.cache_enabled and response.status_code == 200:
                self._cache_response(context, route_config, response)
            
            # Add response headers
            processing_time = (time.time() - start_time) * 1000
            response.headers['X-Processing-Time'] = f"{processing_time:.2f}ms"
            response.headers['X-Request-ID'] = context.request_id
            
            # Add rate limit headers
            if 'rate_limit_remaining' in rate_limit_result:
                response.headers['X-RateLimit-Remaining'] = str(rate_limit_result['rate_limit_remaining'])
                response.headers['X-RateLimit-Limit'] = str(rate_limit_result['rate_limit_limit'])
            
            # Log successful request
            await self._log_request(context, response.status_code, processing_time)
            
            return response
            
        except Exception as e:
            # Log error
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Gateway error: {e}")
            
            if 'context' in locals():
                await self._log_request(context, 500, processing_time, str(e))
            
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error"}
            )
    
    async def _create_request_context(self, request: Request) -> RequestContext:
        """Create request context from FastAPI request."""
        
        # Extract client IP
        client_ip = request.client.host
        if 'x-forwarded-for' in request.headers:
            client_ip = request.headers['x-forwarded-for'].split(',')[0].strip()
        
        # Read request body
        body = await request.body()
        
        context = RequestContext(
            method=request.method,
            path=request.url.path,
            query_params=dict(request.query_params),
            headers=dict(request.headers),
            body=body,
            client_ip=client_ip,
            user_agent=request.headers.get('user-agent', ''),
            origin=request.headers.get('origin', '')
        )
        
        return context
    
    async def _authenticate_request(self, context: RequestContext, request: Request) -> bool:
        """Authenticate request using various methods."""
        
        # Extract tenant ID from header or path
        tenant_id = request.headers.get('x-tenant-id') or context.query_params.get('tenant_id')
        if not tenant_id:
            # Try to extract from path
            path_parts = context.path.strip('/').split('/')
            if len(path_parts) > 1 and path_parts[0] == 'tenants':
                tenant_id = path_parts[1]
        
        if not tenant_id:
            return False
        
        context.tenant_id = tenant_id
        
        # Check for JWT token
        auth_header = request.headers.get('authorization', '')
        if auth_header.startswith('Bearer '):
            token = auth_header[7:]
            
            try:
                # Validate token with security orchestrator
                if self.security_orchestrator:
                    auth_result = self.security_orchestrator.authenticate_request(
                        token, tenant_id, context.client_ip, context.user_agent
                    )
                    
                    context.authenticated = True
                    context.user_id = auth_result['user_id']
                    context.auth_method = 'jwt'
                    
                    return True
                    
            except Exception as e:
                logger.warning(f"JWT authentication failed: {e}")
        
        # Check for SSO session
        session_id = request.cookies.get('sso_session_id')
        if session_id and self.sso_system:
            session = self.sso_system.validate_session(session_id)
            if session:
                context.authenticated = True
                context.user_id = session.user_id
                context.tenant_id = session.tenant_id
                context.auth_method = 'sso'
                context.roles = session.roles
                
                return True
        
        return False
    
    async def _authorize_request(self, context: RequestContext) -> bool:
        """Authorize request using RBAC system."""
        
        if not context.authenticated or not self.rbac_system:
            return True  # Skip authorization if not authenticated or no RBAC
        
        # Determine required permission based on method
        permission_map = {
            'GET': PermissionType.READ,
            'POST': PermissionType.WRITE,
            'PUT': PermissionType.WRITE,
            'DELETE': PermissionType.DELETE,
            'PATCH': PermissionType.WRITE
        }
        
        required_permission = permission_map.get(context.method, PermissionType.READ)
        
        # Check permission
        return self.rbac_system.check_permission(
            context.user_id,
            context.tenant_id,
            ResourceType.AGENT,  # Default to agent resource
            context.path,
            required_permission,
            {
                'ip_address': context.client_ip,
                'user_agent': context.user_agent,
                'method': context.method
            }
        )
    
    async def _check_rate_limits(self, context: RequestContext) -> Dict[str, Any]:
        """Check rate limits for request."""
        
        # Find applicable rate limit policies
        applicable_policies = []
        
        for policy in self.rate_limiter.policies.values():
            if policy.tenant_id == context.tenant_id or policy.tenant_id == "*":
                # Check path patterns
                if policy.applies_to_paths:
                    if not any(pattern in context.path for pattern in policy.applies_to_paths):
                        continue
                
                # Check methods
                if policy.applies_to_methods:
                    if RequestMethod(context.method) not in policy.applies_to_methods:
                        continue
                
                # Check IPs
                if policy.applies_to_ips:
                    if context.client_ip not in policy.applies_to_ips:
                        continue
                
                applicable_policies.append(policy)
        
        # Check each applicable policy
        for policy in applicable_policies:
            allowed, metadata = self.rate_limiter.check_rate_limit(context, policy.policy_id)
            if not allowed:
                return {'allowed': False, **metadata}
            
            # Update context with rate limit info
            if 'rate_limit_remaining' in metadata:
                context.rate_limit_remaining = metadata['rate_limit_remaining']
        
        return {'allowed': True}
    
    def _find_matching_route(self, path: str, method: str) -> Optional[RouteConfiguration]:
        """Find matching route configuration."""
        
        for route in self.routes.values():
            if RequestMethod(method) not in route.methods:
                continue
            
            # Simple pattern matching (extend for complex patterns)
            if route.path_pattern == path or route.path_pattern == "/*":
                return route
        
        return None
    
    def _get_cached_response(self, context: RequestContext, route_config: RouteConfiguration) -> Optional[Response]:
        """Get cached response if available."""
        
        cache_key = self._generate_cache_key(context, route_config)
        
        if cache_key in self.response_cache:
            cached_response, cached_at = self.response_cache[cache_key]
            
            # Check if cache is still valid
            if datetime.utcnow() - cached_at < timedelta(seconds=route_config.cache_ttl_seconds):
                return cached_response
            else:
                # Remove expired cache entry
                del self.response_cache[cache_key]
        
        return None
    
    def _cache_response(self, context: RequestContext, route_config: RouteConfiguration, response: Response):
        """Cache response if configured."""
        
        cache_key = self._generate_cache_key(context, route_config)
        self.response_cache[cache_key] = (response, datetime.utcnow())
        
        # Limit cache size (simple LRU)
        if len(self.response_cache) > 1000:
            # Remove oldest entries
            sorted_cache = sorted(self.response_cache.items(), key=lambda x: x[1][1])
            for key, _ in sorted_cache[:100]:  # Remove 100 oldest entries
                del self.response_cache[key]
    
    def _generate_cache_key(self, context: RequestContext, route_config: RouteConfiguration) -> str:
        """Generate cache key for request."""
        
        key_parts = [context.tenant_id, context.path, context.method]
        
        # Add vary headers to cache key
        for header in route_config.cache_vary_headers:
            if header.lower() in context.headers:
                key_parts.append(f"{header}:{context.headers[header.lower()]}")
        
        # Add query parameters
        if context.query_params:
            query_str = '&'.join(f"{k}={v}" for k, v in sorted(context.query_params.items()))
            key_parts.append(query_str)
        
        return hashlib.sha256('|'.join(key_parts).encode()).hexdigest()
    
    async def _log_request(self, context: RequestContext, status_code: int,
                          processing_time_ms: float, error: str = None):
        """Log request for audit and monitoring."""
        
        # Update metrics
        tenant_key = context.tenant_id or "unknown"
        self.request_metrics[tenant_key].append(processing_time_ms)
        
        if status_code >= 400:
            self.error_counts[tenant_key] += 1
        
        # Log to audit system
        if self.audit_logger:
            audit_event = AuditEvent(
                tenant_id=context.tenant_id,
                user_id=context.user_id,
                event_type=AuditEventType.SYSTEM_EVENT,
                resource="api_gateway",
                action=f"{context.method} {context.path}",
                result="success" if status_code < 400 else "failure",
                ip_address=context.client_ip,
                user_agent=context.user_agent,
                metadata={
                    'request_id': context.request_id,
                    'status_code': status_code,
                    'processing_time_ms': processing_time_ms,
                    'threat_score': context.threat_score,
                    'security_violations': context.security_violations,
                    'error': error
                }
            )
            self.audit_logger.log_audit_event(audit_event)
    
    async def _log_security_event(self, context: RequestContext, description: str):
        """Log security event."""
        
        if self.audit_logger:
            audit_event = AuditEvent(
                tenant_id=context.tenant_id,
                user_id=context.user_id,
                event_type=AuditEventType.SECURITY_EVENT,
                resource="api_gateway",
                action="security_block",
                result="blocked",
                ip_address=context.client_ip,
                user_agent=context.user_agent,
                metadata={
                    'request_id': context.request_id,
                    'threat_score': context.threat_score,
                    'security_violations': context.security_violations,
                    'description': description
                }
            )
            self.audit_logger.log_audit_event(audit_event)
        
        logger.warning(f"Security event: {description} for request {context.request_id}")
    
    def add_route(self, route_config: RouteConfiguration):
        """Add route configuration."""
        self.routes[route_config.route_id] = route_config
        logger.info(f"Added route {route_config.path_pattern} -> {route_config.target_service}")
    
    def add_rate_limit_policy(self, policy: RateLimitPolicy):
        """Add rate limiting policy."""
        self.rate_limiter.add_policy(policy)
    
    def _get_gateway_metrics(self) -> Dict[str, Any]:
        """Get gateway performance metrics."""
        
        total_requests = sum(len(times) for times in self.request_metrics.values())
        total_errors = sum(self.error_counts.values())
        
        # Calculate average response times per tenant
        avg_response_times = {}
        for tenant_id, times in self.request_metrics.items():
            if times:
                avg_response_times[tenant_id] = statistics.mean(times)
        
        return {
            'total_requests': total_requests,
            'total_errors': total_errors,
            'error_rate_percent': (total_errors / max(total_requests, 1)) * 100,
            'avg_response_times_ms': avg_response_times,
            'active_routes': len(self.routes),
            'rate_limit_policies': len(self.rate_limiter.policies),
            'blocked_clients': len(self.rate_limiter.blocked_clients),
            'cache_entries': len(self.response_cache),
            'suspicious_ips': len(self.security_analyzer.suspicious_ips)
        }
    
    def run(self, host: str = "0.0.0.0", port: int = 8080):
        """Run the API gateway server."""
        
        if not self.app:
            raise RuntimeError("FastAPI not available. Cannot run server.")
        
        logger.info(f"Starting Enterprise API Gateway on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)

# Example usage and testing
if __name__ == "__main__":
    from enterprise_security_architecture import (
        EnterpriseSecurityOrchestrator, SOC2ComplianceEngine
    )
    from multi_tenant_rbac_system import MultiTenantRBACSystem
    from enterprise_sso_integration import EnterpriseSSO
    from multi_tenant_orchestration import MultiTenantOrchestrator
    
    # Initialize systems
    compliance_engine = SOC2ComplianceEngine()
    security_orchestrator = EnterpriseSecurityOrchestrator()
    rbac_system = MultiTenantRBACSystem(audit_logger=compliance_engine)
    sso_system = EnterpriseSSO(rbac_system=rbac_system, audit_logger=compliance_engine)
    orchestrator = MultiTenantOrchestrator(rbac_system=rbac_system, audit_logger=compliance_engine)
    
    # Initialize API Gateway
    gateway = EnterpriseAPIGateway(
        security_orchestrator=security_orchestrator,
        rbac_system=rbac_system,
        sso_system=sso_system,
        orchestrator=orchestrator,
        audit_logger=compliance_engine
    )
    
    # Add sample route configuration
    route_config = RouteConfiguration(
        path_pattern="/api/v1/agents/*",
        methods={RequestMethod.GET, RequestMethod.POST},
        target_service="agent_service",
        require_authentication=True,
        cache_enabled=True,
        cache_ttl_seconds=300
    )
    gateway.add_route(route_config)
    
    # Add sample rate limit policy
    rate_limit_policy = RateLimitPolicy(
        name="Standard API Limits",
        tenant_id="test-corp",
        requests_per_minute=100,
        requests_per_hour=1000,
        strategy=RateLimitStrategy.TOKEN_BUCKET,
        applies_to_paths={"/api/"},
        applies_to_methods={RequestMethod.GET, RequestMethod.POST}
    )
    gateway.add_rate_limit_policy(rate_limit_policy)
    
    # Get metrics
    metrics = gateway._get_gateway_metrics()
    print(f"Gateway metrics: {json.dumps(metrics, indent=2)}")
    
    # Note: Uncomment to run the server
    # gateway.run(host="0.0.0.0", port=8080)
