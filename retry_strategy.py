"""
Intelligent Retry Strategies with Exponential Backoff

This module implements sophisticated retry mechanisms with:
- Exponential backoff with jitter
- Error-specific retry logic
- Rate limit aware retries
- Maximum retry limits and timeout controls
"""

import asyncio
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Type, Union
import logging

from payment_processor import PaymentProvider, PaymentError, RateLimitError, NetworkError, ValidationError

logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    """Retry strategy types"""
    EXPONENTIAL_BACKOFF = auto()
    LINEAR_BACKOFF = auto()
    FIXED_INTERVAL = auto()
    IMMEDIATE = auto()


@dataclass
class RetryConfig:
    """Retry configuration parameters"""
    max_attempts: int = 5
    base_delay_ms: int = 1000              # Base delay in milliseconds
    max_delay_ms: int = 60000              # Maximum delay in milliseconds
    exponential_base: float = 2.0          # Exponential backoff base
    jitter_factor: float = 0.1             # Random jitter factor (0-1)
    timeout_seconds: int = 300             # Total retry timeout
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    
    # Error-specific configurations
    retryable_errors: List[Type[Exception]] = field(default_factory=lambda: [
        RateLimitError, NetworkError, asyncio.TimeoutError, ConnectionError
    ])
    non_retryable_errors: List[Type[Exception]] = field(default_factory=lambda: [
        ValidationError
    ])
    
    # Rate limit specific
    respect_retry_after: bool = True       # Honor Retry-After headers
    rate_limit_multiplier: float = 1.5     # Extra delay for rate limits


@dataclass
class RetryAttempt:
    """Information about a retry attempt"""
    attempt_number: int
    delay_ms: int
    error: Optional[Exception]
    timestamp: datetime
    provider: PaymentProvider


@dataclass
class RetryResult:
    """Result of retry operation"""
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    attempts: List[RetryAttempt] = field(default_factory=list)
    total_duration_ms: int = 0
    final_attempt_number: int = 0


class RetryExhaustedException(Exception):
    """Raised when all retry attempts are exhausted"""
    def __init__(self, provider: PaymentProvider, attempts: List[RetryAttempt]):
        self.provider = provider
        self.attempts = attempts
        super().__init__(
            f"Retry exhausted for {provider.value} after {len(attempts)} attempts"
        )


class RetryManager:
    """
    Intelligent retry manager with exponential backoff and jitter
    
    Features:
    - Configurable retry strategies
    - Error-type specific retry logic
    - Rate limit aware delays
    - Circuit breaker integration
    - Comprehensive retry metrics
    """
    
    def __init__(self, provider: PaymentProvider, config: RetryConfig):
        self.provider = provider
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{provider.value}")
        self._active_retries: Dict[str, RetryResult] = {}
    
    async def execute_with_retry(self, 
                               func: Callable, 
                               *args, 
                               operation_id: Optional[str] = None,
                               **kwargs) -> Any:
        """
        Execute a function with intelligent retry logic
        
        Args:
            func: The async function to execute
            *args, **kwargs: Arguments to pass to the function
            operation_id: Unique identifier for this operation
            
        Returns:
            The function result
            
        Raises:
            RetryExhaustedException: When all retries are exhausted
        """
        if operation_id is None:
            operation_id = f"{func.__name__}_{int(time.time() * 1000)}"
        
        start_time = time.time()
        retry_result = RetryResult()
        self._active_retries[operation_id] = retry_result
        
        try:
            for attempt in range(1, self.config.max_attempts + 1):
                attempt_start = time.time()
                
                try:
                    # Check if we've exceeded total timeout
                    elapsed = (time.time() - start_time) * 1000
                    if elapsed > self.config.timeout_seconds * 1000:
                        raise RetryExhaustedException(self.provider, retry_result.attempts)
                    
                    self.logger.debug(
                        f"Attempt {attempt}/{self.config.max_attempts} for {operation_id}"
                    )
                    
                    # Execute the function
                    result = await func(*args, **kwargs)
                    
                    # Success!
                    retry_result.success = True
                    retry_result.result = result
                    retry_result.final_attempt_number = attempt
                    retry_result.total_duration_ms = int((time.time() - start_time) * 1000)
                    
                    if attempt > 1:
                        self.logger.info(
                            f"Operation {operation_id} succeeded on attempt {attempt}"
                        )
                    
                    return result
                    
                except Exception as error:
                    attempt_duration = int((time.time() - attempt_start) * 1000)
                    
                    # Record the attempt
                    retry_attempt = RetryAttempt(
                        attempt_number=attempt,
                        delay_ms=0,  # Will be set below if retrying
                        error=error,
                        timestamp=datetime.utcnow(),
                        provider=self.provider
                    )
                    retry_result.attempts.append(retry_attempt)
                    
                    # Check if this error is retryable
                    if not self._is_retryable_error(error):
                        self.logger.warning(
                            f"Non-retryable error in {operation_id}: {error}"
                        )
                        retry_result.error = error
                        retry_result.final_attempt_number = attempt
                        raise error
                    
                    # Check if this is the last attempt
                    if attempt >= self.config.max_attempts:
                        self.logger.error(
                            f"All retry attempts exhausted for {operation_id}"
                        )
                        retry_result.error = error
                        retry_result.final_attempt_number = attempt
                        raise RetryExhaustedException(self.provider, retry_result.attempts)
                    
                    # Calculate delay for next attempt
                    delay_ms = self._calculate_delay(attempt, error)
                    retry_attempt.delay_ms = delay_ms
                    
                    self.logger.warning(
                        f"Attempt {attempt} failed for {operation_id}, "
                        f"retrying in {delay_ms}ms. Error: {error}"
                    )
                    
                    # Wait before next attempt
                    await asyncio.sleep(delay_ms / 1000.0)
            
        finally:
            # Clean up active retry tracking
            self._active_retries.pop(operation_id, None)
            retry_result.total_duration_ms = int((time.time() - start_time) * 1000)
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """Determine if an error should trigger a retry"""
        # Check non-retryable errors first
        for error_type in self.config.non_retryable_errors:
            if isinstance(error, error_type):
                return False
        
        # Check retryable errors
        for error_type in self.config.retryable_errors:
            if isinstance(error, error_type):
                return True
        
        # Check PaymentError retryable flag
        if isinstance(error, PaymentError):
            return error.is_retryable
        
        # Default to non-retryable for unknown errors
        return False
    
    def _calculate_delay(self, attempt: int, error: Optional[Exception] = None) -> int:
        """Calculate delay for the next retry attempt"""
        if isinstance(error, RateLimitError) and error.retry_after and self.config.respect_retry_after:
            # Honor the Retry-After header from rate limit errors
            base_delay = error.retry_after * 1000
            return min(
                int(base_delay * self.config.rate_limit_multiplier),
                self.config.max_delay_ms
            )
        
        # Calculate base delay according to strategy
        if self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.config.base_delay_ms * (self.config.exponential_base ** (attempt - 1))
        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.config.base_delay_ms * attempt
        elif self.config.strategy == RetryStrategy.FIXED_INTERVAL:
            delay = self.config.base_delay_ms
        else:  # IMMEDIATE
            delay = 0
        
        # Apply jitter to prevent thundering herd
        if self.config.jitter_factor > 0:
            jitter = delay * self.config.jitter_factor * random.random()
            delay = delay + jitter
        
        # Apply rate limit multiplier for rate limit errors
        if isinstance(error, RateLimitError):
            delay *= self.config.rate_limit_multiplier
        
        # Ensure delay is within bounds
        return min(int(delay), self.config.max_delay_ms)
    
    def get_active_retries(self) -> Dict[str, RetryResult]:
        """Get currently active retry operations"""
        return self._active_retries.copy()
    
    def get_retry_metrics(self) -> Dict[str, Any]:
        """Get retry metrics for this provider"""
        active_count = len(self._active_retries)
        
        return {
            "provider": self.provider.value,
            "active_retries": active_count,
            "config": {
                "max_attempts": self.config.max_attempts,
                "base_delay_ms": self.config.base_delay_ms,
                "max_delay_ms": self.config.max_delay_ms,
                "strategy": self.config.strategy.name,
                "timeout_seconds": self.config.timeout_seconds,
            }
        }


class AdaptiveRetryManager(RetryManager):
    """
    Adaptive retry manager that adjusts retry parameters based on observed behavior
    
    Features:
    - Dynamic delay adjustment based on success rates
    - Error pattern learning
    - Provider-specific optimization
    """
    
    def __init__(self, provider: PaymentProvider, config: RetryConfig):
        super().__init__(provider, config)
        self.success_rate_window = []
        self.error_history = []
        self.adaptive_multiplier = 1.0
        self.adaptation_enabled = True
    
    async def execute_with_retry(self, func: Callable, *args, operation_id: Optional[str] = None, **kwargs) -> Any:
        """Execute with adaptive retry logic"""
        try:
            result = await super().execute_with_retry(func, *args, operation_id=operation_id, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure(e)
            raise
    
    def _record_success(self):
        """Record a successful operation for adaptation"""
        self.success_rate_window.append(True)
        self._trim_windows()
        self._adapt_parameters()
    
    def _record_failure(self, error: Exception):
        """Record a failed operation for adaptation"""
        self.success_rate_window.append(False)
        self.error_history.append(type(error).__name__)
        self._trim_windows()
        self._adapt_parameters()
    
    def _trim_windows(self):
        """Keep sliding windows at reasonable size"""
        max_window_size = 100
        if len(self.success_rate_window) > max_window_size:
            self.success_rate_window = self.success_rate_window[-max_window_size:]
        if len(self.error_history) > max_window_size:
            self.error_history = self.error_history[-max_window_size:]
    
    def _adapt_parameters(self):
        """Adapt retry parameters based on observed patterns"""
        if not self.adaptation_enabled or len(self.success_rate_window) < 10:
            return
        
        success_rate = sum(self.success_rate_window) / len(self.success_rate_window)
        
        # Adjust adaptive multiplier based on success rate
        if success_rate > 0.9:
            # High success rate - can be more aggressive
            self.adaptive_multiplier = max(0.5, self.adaptive_multiplier * 0.95)
        elif success_rate < 0.5:
            # Low success rate - be more conservative
            self.adaptive_multiplier = min(2.0, self.adaptive_multiplier * 1.1)
        
        self.logger.debug(
            f"Adaptive multiplier for {self.provider.value}: {self.adaptive_multiplier:.2f}, "
            f"success rate: {success_rate:.2%}"
        )
    
    def _calculate_delay(self, attempt: int, error: Optional[Exception] = None) -> int:
        """Calculate delay with adaptive adjustments"""
        base_delay = super()._calculate_delay(attempt, error)
        
        if self.adaptation_enabled:
            adapted_delay = int(base_delay * self.adaptive_multiplier)
            return min(adapted_delay, self.config.max_delay_ms)
        
        return base_delay
    
    def get_adaptation_metrics(self) -> Dict[str, Any]:
        """Get adaptation-specific metrics"""
        success_rate = 0.0
        if self.success_rate_window:
            success_rate = sum(self.success_rate_window) / len(self.success_rate_window)
        
        error_counts = {}
        for error_type in self.error_history:
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        return {
            "adaptive_multiplier": self.adaptive_multiplier,
            "success_rate": success_rate,
            "window_size": len(self.success_rate_window),
            "error_distribution": error_counts,
            "adaptation_enabled": self.adaptation_enabled
        }


class RetryManagerFactory:
    """Factory for creating retry managers with provider-specific configurations"""
    
    DEFAULT_CONFIGS = {
        PaymentProvider.STRIPE: RetryConfig(
            max_attempts=5,
            base_delay_ms=1000,
            max_delay_ms=32000,
            rate_limit_multiplier=2.0  # Stripe is stricter on rate limits
        ),
        PaymentProvider.PAYPAL: RetryConfig(
            max_attempts=4,
            base_delay_ms=1500,
            max_delay_ms=60000,
            rate_limit_multiplier=1.5
        ),
        PaymentProvider.SQUARE: RetryConfig(
            max_attempts=3,
            base_delay_ms=2000,
            max_delay_ms=45000,
            rate_limit_multiplier=1.2
        )
    }
    
    @classmethod
    def create_retry_manager(cls, 
                           provider: PaymentProvider, 
                           config: Optional[RetryConfig] = None,
                           adaptive: bool = True) -> RetryManager:
        """Create a retry manager for the specified provider"""
        if config is None:
            config = cls.DEFAULT_CONFIGS.get(provider, RetryConfig())
        
        if adaptive:
            return AdaptiveRetryManager(provider, config)
        else:
            return RetryManager(provider, config)
    
    @classmethod
    def create_all_managers(cls, adaptive: bool = True) -> Dict[PaymentProvider, RetryManager]:
        """Create retry managers for all supported providers"""
        return {
            provider: cls.create_retry_manager(provider, adaptive=adaptive)
            for provider in PaymentProvider
        }