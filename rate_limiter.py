"""
Advanced Rate Limiting and Load Balancing System

This module implements sophisticated rate limiting and load balancing for payment processors:
- Token bucket and sliding window rate limiting
- Provider-specific rate limit management
- Intelligent load balancing with health checks
- Request queue management with prioritization
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union
import heapq
import logging

from payment_processor import PaymentProvider, PaymentRequest, RateLimitError

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = auto()
    WEIGHTED_ROUND_ROBIN = auto()
    LEAST_CONNECTIONS = auto()
    HEALTH_AWARE = auto()
    COST_OPTIMIZED = auto()


class RequestPriority(Enum):
    """Request priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    requests_per_second: int = 100
    burst_capacity: int = 200           # Token bucket capacity
    window_size_seconds: int = 60       # Sliding window size
    queue_max_size: int = 1000         # Maximum queued requests
    queue_timeout_seconds: int = 30     # Queue timeout
    adaptive_rate_limiting: bool = True # Enable adaptive rate limiting


@dataclass
class ProviderWeight:
    """Provider weight configuration for load balancing"""
    provider: PaymentProvider
    weight: float = 1.0                 # Higher weight = more traffic
    cost_per_transaction: float = 0.0   # Cost in cents
    max_concurrent: int = 100           # Maximum concurrent requests
    health_score: float = 1.0           # Health score (0-1)


@dataclass
class QueuedRequest:
    """Queued request with priority and timing"""
    request: PaymentRequest
    priority: RequestPriority
    queued_at: datetime
    future: asyncio.Future
    timeout_handle: Optional[asyncio.Handle] = None
    
    def __lt__(self, other):
        # Higher priority first, then FIFO within priority
        if self.priority != other.priority:
            return self.priority.value > other.priority.value
        return self.queued_at < other.queued_at


class TokenBucket:
    """Token bucket rate limiter implementation"""
    
    def __init__(self, rate: float, capacity: int):
        self.rate = rate                # Tokens per second
        self.capacity = capacity        # Maximum tokens
        self.tokens = capacity          # Current tokens
        self.last_update = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> bool:
        """Attempt to acquire tokens from the bucket"""
        async with self._lock:
            now = time.time()
            
            # Add tokens based on elapsed time
            elapsed = now - self.last_update
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_update = now
            
            # Check if we have enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    async def wait_for_tokens(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """Wait until tokens are available"""
        start_time = time.time()
        
        while True:
            if await self.acquire(tokens):
                return True
            
            # Check timeout
            if timeout and (time.time() - start_time) >= timeout:
                return False
            
            # Calculate wait time until next token
            async with self._lock:
                wait_time = max(0.01, (tokens - self.tokens) / self.rate)
            
            await asyncio.sleep(min(wait_time, 0.1))  # Cap at 100ms
    
    def get_available_tokens(self) -> int:
        """Get number of available tokens"""
        now = time.time()
        elapsed = now - self.last_update
        available = min(self.capacity, self.tokens + elapsed * self.rate)
        return int(available)


class SlidingWindowRateLimiter:
    """Sliding window rate limiter for tracking request history"""
    
    def __init__(self, window_size: int, max_requests: int):
        self.window_size = window_size
        self.max_requests = max_requests
        self.requests = deque()
        self._lock = asyncio.Lock()
    
    async def is_allowed(self) -> bool:
        """Check if a request is allowed within the sliding window"""
        async with self._lock:
            now = time.time()
            
            # Remove old requests outside the window
            while self.requests and self.requests[0] <= now - self.window_size:
                self.requests.popleft()
            
            # Check if we're under the limit
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            
            return False
    
    def get_current_count(self) -> int:
        """Get current request count in the window"""
        now = time.time()
        count = 0
        for request_time in reversed(self.requests):
            if request_time > now - self.window_size:
                count += 1
            else:
                break
        return count
    
    def time_until_next_slot(self) -> float:
        """Calculate time until next request slot is available"""
        if len(self.requests) < self.max_requests:
            return 0.0
        
        # Time until oldest request falls out of window
        now = time.time()
        oldest_request = self.requests[0]
        return max(0.0, self.window_size - (now - oldest_request))


class ProviderRateLimiter:
    """Rate limiter for a specific payment provider"""
    
    def __init__(self, provider: PaymentProvider, config: RateLimitConfig):
        self.provider = provider
        self.config = config
        self.token_bucket = TokenBucket(config.requests_per_second, config.burst_capacity)
        self.sliding_window = SlidingWindowRateLimiter(
            config.window_size_seconds, 
            config.requests_per_second * config.window_size_seconds
        )
        self.request_queue = []  # Priority queue
        self.active_requests = 0
        self.total_requests = 0
        self.rate_limited_requests = 0
        self.adaptive_rate = config.requests_per_second
        self.logger = logging.getLogger(f"{__name__}.{provider.value}")
        
    async def acquire_permit(self, 
                           priority: RequestPriority = RequestPriority.NORMAL,
                           timeout: Optional[float] = None) -> bool:
        """Acquire a permit to make a request"""
        # Check both token bucket and sliding window
        if await self.token_bucket.acquire() and await self.sliding_window.is_allowed():
            self.active_requests += 1
            self.total_requests += 1
            return True
        
        # If immediate acquisition fails, queue the request if enabled
        if self.config.queue_max_size > 0:
            return await self._queue_request(priority, timeout)
        
        # No queuing, rate limited
        self.rate_limited_requests += 1
        return False
    
    async def _queue_request(self, 
                           priority: RequestPriority,
                           timeout: Optional[float]) -> bool:
        """Queue a request when rate limited"""
        if len(self.request_queue) >= self.config.queue_max_size:
            self.rate_limited_requests += 1
            return False
        
        # Create a future for this request
        future = asyncio.Future()
        queued_request = QueuedRequest(
            request=None,  # We don't need the actual request for permit acquisition
            priority=priority,
            queued_at=datetime.utcnow(),
            future=future
        )
        
        # Set up timeout
        if timeout:
            def timeout_callback():
                if not future.done():
                    future.set_result(False)
            
            queued_request.timeout_handle = asyncio.get_event_loop().call_later(
                timeout, timeout_callback
            )
        
        # Add to priority queue
        heapq.heappush(self.request_queue, queued_request)
        
        # Start processing queue
        asyncio.create_task(self._process_queue())
        
        # Wait for result
        return await future
    
    async def _process_queue(self):
        """Process queued requests when permits become available"""
        while self.request_queue:
            # Check if we can process a request
            if not (await self.token_bucket.acquire() and await self.sliding_window.is_allowed()):
                # Wait a bit before checking again
                await asyncio.sleep(0.01)
                continue
            
            # Get highest priority request
            queued_request = heapq.heappop(self.request_queue)
            
            # Cancel timeout
            if queued_request.timeout_handle:
                queued_request.timeout_handle.cancel()
            
            # Check if request is still valid
            if queued_request.future.done():
                continue
            
            # Grant the permit
            self.active_requests += 1
            self.total_requests += 1
            queued_request.future.set_result(True)
    
    def release_permit(self):
        """Release a permit after request completion"""
        if self.active_requests > 0:
            self.active_requests -= 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get rate limiting metrics"""
        return {
            "provider": self.provider.value,
            "active_requests": self.active_requests,
            "total_requests": self.total_requests,
            "rate_limited_requests": self.rate_limited_requests,
            "queue_size": len(self.request_queue),
            "available_tokens": self.token_bucket.get_available_tokens(),
            "sliding_window_count": self.sliding_window.get_current_count(),
            "adaptive_rate": self.adaptive_rate,
            "config": {
                "requests_per_second": self.config.requests_per_second,
                "burst_capacity": self.config.burst_capacity,
                "queue_max_size": self.config.queue_max_size,
            }
        }
    
    async def adjust_rate(self, new_rate: float):
        """Dynamically adjust the rate limit"""
        if self.config.adaptive_rate_limiting:
            self.adaptive_rate = new_rate
            self.token_bucket.rate = new_rate
            self.logger.info(f"Adjusted rate limit for {self.provider.value} to {new_rate} req/s")


class LoadBalancer:
    """Intelligent load balancer for payment providers"""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.HEALTH_AWARE):
        self.strategy = strategy
        self.providers: Dict[PaymentProvider, ProviderWeight] = {}
        self.round_robin_index = 0
        self.request_counts: Dict[PaymentProvider, int] = {}
        self.response_times: Dict[PaymentProvider, List[float]] = {}
        self.logger = logging.getLogger(f"{__name__}.load_balancer")
    
    def register_provider(self, provider: PaymentProvider, weight: ProviderWeight):
        """Register a payment provider with the load balancer"""
        self.providers[provider] = weight
        self.request_counts[provider] = 0
        self.response_times[provider] = []
        self.logger.info(f"Registered provider {provider.value} with weight {weight.weight}")
    
    def update_provider_health(self, provider: PaymentProvider, health_score: float):
        """Update provider health score"""
        if provider in self.providers:
            self.providers[provider].health_score = max(0.0, min(1.0, health_score))
    
    def record_request_completion(self, provider: PaymentProvider, response_time_ms: float):
        """Record request completion for metrics"""
        if provider in self.response_times:
            self.response_times[provider].append(response_time_ms)
            # Keep only recent response times
            if len(self.response_times[provider]) > 100:
                self.response_times[provider] = self.response_times[provider][-100:]
    
    def select_provider(self, available_providers: List[PaymentProvider]) -> Optional[PaymentProvider]:
        """Select the best provider based on the load balancing strategy"""
        if not available_providers:
            return None
        
        # Filter to only registered providers
        candidates = [p for p in available_providers if p in self.providers]
        if not candidates:
            return available_providers[0] if available_providers else None
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_select(candidates)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_select(candidates)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_select(candidates)
        elif self.strategy == LoadBalancingStrategy.HEALTH_AWARE:
            return self._health_aware_select(candidates)
        elif self.strategy == LoadBalancingStrategy.COST_OPTIMIZED:
            return self._cost_optimized_select(candidates)
        else:
            return candidates[0]
    
    def _round_robin_select(self, providers: List[PaymentProvider]) -> PaymentProvider:
        """Simple round-robin selection"""
        provider = providers[self.round_robin_index % len(providers)]
        self.round_robin_index += 1
        return provider
    
    def _weighted_round_robin_select(self, providers: List[PaymentProvider]) -> PaymentProvider:
        """Weighted round-robin selection"""
        total_weight = sum(self.providers[p].weight for p in providers)
        if total_weight == 0:
            return self._round_robin_select(providers)
        
        # Calculate weighted probabilities
        weights = [(p, self.providers[p].weight / total_weight) for p in providers]
        
        # Use request counts to implement weighted round-robin
        min_ratio = float('inf')
        selected_provider = providers[0]
        
        for provider, weight in weights:
            current_ratio = self.request_counts[provider] / max(weight, 0.001)
            if current_ratio < min_ratio:
                min_ratio = current_ratio
                selected_provider = provider
        
        self.request_counts[selected_provider] += 1
        return selected_provider
    
    def _least_connections_select(self, providers: List[PaymentProvider]) -> PaymentProvider:
        """Select provider with least active connections"""
        return min(providers, key=lambda p: self.request_counts.get(p, 0))
    
    def _health_aware_select(self, providers: List[PaymentProvider]) -> PaymentProvider:
        """Select provider based on health score and performance"""
        def score_provider(provider: PaymentProvider) -> float:
            weight = self.providers[provider]
            health_score = weight.health_score
            
            # Factor in average response time
            avg_response_time = 0
            if self.response_times[provider]:
                avg_response_time = sum(self.response_times[provider]) / len(self.response_times[provider])
            
            # Normalize response time (lower is better)
            response_time_score = 1.0 / (1.0 + avg_response_time / 1000.0)  # Convert to seconds
            
            # Combine scores
            return (health_score * 0.7) + (response_time_score * 0.3)
        
        return max(providers, key=score_provider)
    
    def _cost_optimized_select(self, providers: List[PaymentProvider]) -> PaymentProvider:
        """Select provider with best cost efficiency"""
        def cost_score(provider: PaymentProvider) -> float:
            weight = self.providers[provider]
            # Lower cost is better, but factor in health
            if weight.cost_per_transaction <= 0:
                return weight.health_score
            
            cost_efficiency = weight.health_score / weight.cost_per_transaction
            return cost_efficiency
        
        return max(providers, key=cost_score)
    
    def get_load_balancing_metrics(self) -> Dict[str, Any]:
        """Get load balancing metrics"""
        provider_stats = {}
        for provider in self.providers:
            avg_response_time = 0
            if self.response_times[provider]:
                avg_response_time = sum(self.response_times[provider]) / len(self.response_times[provider])
            
            provider_stats[provider.value] = {
                "request_count": self.request_counts.get(provider, 0),
                "avg_response_time_ms": avg_response_time,
                "health_score": self.providers[provider].health_score,
                "weight": self.providers[provider].weight,
                "cost_per_transaction": self.providers[provider].cost_per_transaction,
            }
        
        return {
            "strategy": self.strategy.name,
            "providers": provider_stats,
            "round_robin_index": self.round_robin_index,
        }


class RateLimitManager:
    """Central manager for rate limiting across all payment providers"""
    
    def __init__(self):
        self.rate_limiters: Dict[PaymentProvider, ProviderRateLimiter] = {}
        self.load_balancer = LoadBalancer()
        self.logger = logging.getLogger(f"{__name__}.manager")
    
    def register_provider(self, 
                         provider: PaymentProvider, 
                         rate_config: RateLimitConfig,
                         provider_weight: ProviderWeight):
        """Register a provider with rate limiting and load balancing"""
        self.rate_limiters[provider] = ProviderRateLimiter(provider, rate_config)
        self.load_balancer.register_provider(provider, provider_weight)
        self.logger.info(f"Registered provider {provider.value} for rate limiting")
    
    async def acquire_permit(self, 
                           provider: PaymentProvider,
                           priority: RequestPriority = RequestPriority.NORMAL) -> bool:
        """Acquire a permit for the specified provider"""
        if provider not in self.rate_limiters:
            return True  # No rate limiting configured
        
        return await self.rate_limiters[provider].acquire_permit(priority)
    
    def release_permit(self, provider: PaymentProvider):
        """Release a permit for the specified provider"""
        if provider in self.rate_limiters:
            self.rate_limiters[provider].release_permit()
    
    def select_best_provider(self, available_providers: List[PaymentProvider]) -> Optional[PaymentProvider]:
        """Select the best provider using load balancing strategy"""
        return self.load_balancer.select_provider(available_providers)
    
    def record_request_completion(self, provider: PaymentProvider, response_time_ms: float, success: bool):
        """Record request completion for metrics and adaptation"""
        self.load_balancer.record_request_completion(provider, response_time_ms)
        
        # Update health score based on success/failure
        if provider in self.load_balancer.providers:
            current_health = self.load_balancer.providers[provider].health_score
            if success:
                new_health = min(1.0, current_health + 0.01)
            else:
                new_health = max(0.1, current_health - 0.05)
            
            self.load_balancer.update_provider_health(provider, new_health)
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics for all rate limiters and load balancer"""
        rate_limit_metrics = {
            provider.value: limiter.get_metrics()
            for provider, limiter in self.rate_limiters.items()
        }
        
        return {
            "rate_limiters": rate_limit_metrics,
            "load_balancer": self.load_balancer.get_load_balancing_metrics(),
        }