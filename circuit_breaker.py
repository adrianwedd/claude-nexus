"""
Circuit Breaker Implementation for Payment Processing

This module implements a sophisticated circuit breaker pattern with
configurable thresholds, automatic recovery, and detailed metrics.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Union
import logging

from payment_processor import PaymentProvider, PaymentError

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = auto()      # Normal operation
    OPEN = auto()        # Failing fast
    HALF_OPEN = auto()   # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5           # Failures before opening
    success_threshold: int = 3           # Successes to close from half-open
    timeout_seconds: int = 60           # Time to wait before half-open
    request_timeout: int = 30           # Individual request timeout
    rolling_window_size: int = 100      # Size of rolling window for metrics
    failure_rate_threshold: float = 0.5 # Failure rate to trigger opening
    min_requests: int = 10              # Minimum requests before rate calculation


@dataclass
class CircuitMetrics:
    """Circuit breaker metrics tracking"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeouts: int = 0
    rate_limit_errors: int = 0
    network_errors: int = 0
    state_changes: List[str] = field(default_factory=list)
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open"""
    def __init__(self, provider: PaymentProvider, retry_after: int):
        self.provider = provider
        self.retry_after = retry_after
        super().__init__(
            f"Circuit breaker open for {provider.value}, retry after {retry_after}s"
        )


class CircuitBreaker:
    """
    Circuit breaker implementation for payment processors
    
    Implements the circuit breaker pattern with:
    - Configurable failure thresholds
    - Automatic recovery testing
    - Detailed metrics collection
    - Rolling window failure rate calculation
    """
    
    def __init__(self, provider: PaymentProvider, config: CircuitBreakerConfig):
        self.provider = provider
        self.config = config
        self.state = CircuitState.CLOSED
        self.metrics = CircuitMetrics()
        self.last_state_change = datetime.utcnow()
        self.request_history: List[bool] = []  # True for success, False for failure
        self._lock = asyncio.Lock()
        self.logger = logging.getLogger(f"{__name__}.{provider.value}")
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function call through the circuit breaker
        
        Args:
            func: The async function to call
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            The function result
            
        Raises:
            CircuitBreakerOpenError: When circuit is open
            Exception: The original exception from the function
        """
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    await self._transition_to_half_open()
                else:
                    self._record_blocked_request()
                    raise CircuitBreakerOpenError(
                        self.provider, 
                        self._time_until_next_attempt()
                    )
            
            if self.state == CircuitState.HALF_OPEN:
                if self.metrics.consecutive_successes >= self.config.success_threshold:
                    await self._transition_to_closed()
        
        # Execute the function call
        start_time = time.time()
        try:
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.request_timeout
            )
            
            execution_time = (time.time() - start_time) * 1000
            await self._record_success(execution_time)
            return result
            
        except asyncio.TimeoutError as e:
            await self._record_failure(e, "timeout")
            raise
        except PaymentError as e:
            await self._record_failure(e, self._categorize_payment_error(e))
            raise
        except Exception as e:
            await self._record_failure(e, "unknown")
            raise
    
    async def _record_success(self, execution_time_ms: float):
        """Record a successful request"""
        async with self._lock:
            self.metrics.total_requests += 1
            self.metrics.successful_requests += 1
            self.metrics.last_success_time = datetime.utcnow()
            self.metrics.consecutive_failures = 0
            self.metrics.consecutive_successes += 1
            
            self._update_request_history(True)
            
            self.logger.debug(
                f"Success recorded for {self.provider.value}, "
                f"execution time: {execution_time_ms:.2f}ms, "
                f"consecutive successes: {self.metrics.consecutive_successes}"
            )
            
            # Transition from half-open to closed if threshold met
            if (self.state == CircuitState.HALF_OPEN and 
                self.metrics.consecutive_successes >= self.config.success_threshold):
                await self._transition_to_closed()
    
    async def _record_failure(self, error: Exception, error_type: str):
        """Record a failed request"""
        async with self._lock:
            self.metrics.total_requests += 1
            self.metrics.failed_requests += 1
            self.metrics.last_failure_time = datetime.utcnow()
            self.metrics.consecutive_successes = 0
            self.metrics.consecutive_failures += 1
            
            # Categorize the error
            if error_type == "timeout":
                self.metrics.timeouts += 1
            elif error_type == "rate_limit":
                self.metrics.rate_limit_errors += 1
            elif error_type == "network":
                self.metrics.network_errors += 1
            
            self._update_request_history(False)
            
            self.logger.warning(
                f"Failure recorded for {self.provider.value}, "
                f"error: {error}, type: {error_type}, "
                f"consecutive failures: {self.metrics.consecutive_failures}"
            )
            
            # Check if we should open the circuit
            if self._should_open_circuit():
                await self._transition_to_open()
    
    def _should_open_circuit(self) -> bool:
        """Determine if the circuit should be opened"""
        # Check consecutive failures threshold
        if self.metrics.consecutive_failures >= self.config.failure_threshold:
            return True
        
        # Check failure rate in rolling window
        if len(self.request_history) >= self.config.min_requests:
            failure_rate = self._calculate_failure_rate()
            if failure_rate >= self.config.failure_rate_threshold:
                return True
        
        return False
    
    def _calculate_failure_rate(self) -> float:
        """Calculate failure rate in the rolling window"""
        if not self.request_history:
            return 0.0
        
        failures = sum(1 for success in self.request_history if not success)
        return failures / len(self.request_history)
    
    def _update_request_history(self, success: bool):
        """Update the rolling window of request results"""
        self.request_history.append(success)
        if len(self.request_history) > self.config.rolling_window_size:
            self.request_history.pop(0)
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        time_since_open = datetime.utcnow() - self.last_state_change
        return time_since_open.total_seconds() >= self.config.timeout_seconds
    
    def _time_until_next_attempt(self) -> int:
        """Calculate seconds until next attempt is allowed"""
        time_since_open = datetime.utcnow() - self.last_state_change
        remaining = self.config.timeout_seconds - time_since_open.total_seconds()
        return max(0, int(remaining))
    
    async def _transition_to_open(self):
        """Transition circuit to OPEN state"""
        old_state = self.state
        self.state = CircuitState.OPEN
        self.last_state_change = datetime.utcnow()
        
        state_change = f"{old_state.name} -> {self.state.name}"
        self.metrics.state_changes.append(state_change)
        
        self.logger.error(
            f"Circuit breaker OPENED for {self.provider.value}, "
            f"consecutive failures: {self.metrics.consecutive_failures}, "
            f"failure rate: {self._calculate_failure_rate():.2%}"
        )
    
    async def _transition_to_half_open(self):
        """Transition circuit to HALF_OPEN state"""
        old_state = self.state
        self.state = CircuitState.HALF_OPEN
        self.last_state_change = datetime.utcnow()
        self.metrics.consecutive_successes = 0
        
        state_change = f"{old_state.name} -> {self.state.name}"
        self.metrics.state_changes.append(state_change)
        
        self.logger.info(
            f"Circuit breaker HALF-OPEN for {self.provider.value}, "
            f"testing recovery"
        )
    
    async def _transition_to_closed(self):
        """Transition circuit to CLOSED state"""
        old_state = self.state
        self.state = CircuitState.CLOSED
        self.last_state_change = datetime.utcnow()
        self.metrics.consecutive_failures = 0
        
        state_change = f"{old_state.name} -> {self.state.name}"
        self.metrics.state_changes.append(state_change)
        
        self.logger.info(
            f"Circuit breaker CLOSED for {self.provider.value}, "
            f"normal operation resumed"
        )
    
    def _record_blocked_request(self):
        """Record a request that was blocked by the circuit breaker"""
        self.logger.debug(
            f"Request blocked by circuit breaker for {self.provider.value}"
        )
    
    def _categorize_payment_error(self, error: PaymentError) -> str:
        """Categorize payment errors for metrics"""
        if isinstance(error, PaymentError):
            if "rate" in error.error_code.lower() if error.error_code else False:
                return "rate_limit"
            if "network" in error.error_code.lower() if error.error_code else False:
                return "network"
        return "payment_error"
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current circuit breaker metrics"""
        failure_rate = self._calculate_failure_rate() if self.request_history else 0.0
        
        return {
            "provider": self.provider.value,
            "state": self.state.name,
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "successful_requests": self.metrics.successful_requests,
                "failed_requests": self.metrics.failed_requests,
                "failure_rate": failure_rate,
                "consecutive_failures": self.metrics.consecutive_failures,
                "consecutive_successes": self.metrics.consecutive_successes,
                "timeouts": self.metrics.timeouts,
                "rate_limit_errors": self.metrics.rate_limit_errors,
                "network_errors": self.metrics.network_errors,
            },
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "success_threshold": self.config.success_threshold,
                "timeout_seconds": self.config.timeout_seconds,
                "failure_rate_threshold": self.config.failure_rate_threshold,
            },
            "last_state_change": self.last_state_change.isoformat(),
            "state_changes": self.metrics.state_changes[-10:],  # Last 10 changes
        }
    
    def reset(self):
        """Reset the circuit breaker to initial state"""
        self.state = CircuitState.CLOSED
        self.metrics = CircuitMetrics()
        self.request_history.clear()
        self.last_state_change = datetime.utcnow()
        self.logger.info(f"Circuit breaker reset for {self.provider.value}")


class CircuitBreakerManager:
    """Manages circuit breakers for multiple payment providers"""
    
    def __init__(self):
        self.circuit_breakers: Dict[PaymentProvider, CircuitBreaker] = {}
        self.logger = logging.getLogger(f"{__name__}.manager")
    
    def register_provider(self, provider: PaymentProvider, 
                         config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Register a payment provider with circuit breaker"""
        if config is None:
            config = CircuitBreakerConfig()
        
        circuit_breaker = CircuitBreaker(provider, config)
        self.circuit_breakers[provider] = circuit_breaker
        
        self.logger.info(f"Registered circuit breaker for {provider.value}")
        return circuit_breaker
    
    def get_circuit_breaker(self, provider: PaymentProvider) -> Optional[CircuitBreaker]:
        """Get circuit breaker for a provider"""
        return self.circuit_breakers.get(provider)
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get metrics for all circuit breakers"""
        return {
            provider.value: cb.get_metrics()
            for provider, cb in self.circuit_breakers.items()
        }
    
    def get_healthy_providers(self) -> List[PaymentProvider]:
        """Get list of providers with closed circuit breakers"""
        return [
            provider for provider, cb in self.circuit_breakers.items()
            if cb.state == CircuitState.CLOSED
        ]
    
    def reset_all(self):
        """Reset all circuit breakers"""
        for cb in self.circuit_breakers.values():
            cb.reset()
        self.logger.info("All circuit breakers reset")