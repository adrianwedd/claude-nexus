"""
Payment Orchestrator with Fallback Mechanisms

This module implements the main payment orchestrator that coordinates
between multiple payment providers with intelligent fallback strategies,
circuit breakers, retry logic, and rate limiting.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

from payment_processor import (
    PaymentProcessor, PaymentProvider, PaymentRequest, PaymentResponse, 
    PaymentStatus, PaymentError, RefundRequest
)
from circuit_breaker import CircuitBreaker, CircuitBreakerManager, CircuitBreakerOpenError
from retry_strategy import RetryManager, RetryManagerFactory, RetryExhaustedException
from rate_limiter import RateLimitManager, RateLimitConfig, ProviderWeight, RequestPriority

logger = logging.getLogger(__name__)


class FallbackStrategy(Enum):
    """Fallback strategies for provider selection"""
    PRIORITY_ORDER = auto()     # Try providers in priority order
    HEALTH_BASED = auto()       # Try healthiest providers first
    COST_OPTIMIZED = auto()     # Try cheapest providers first
    ROUND_ROBIN = auto()        # Cycle through available providers


class PaymentOutcome(Enum):
    """Final outcome of payment processing"""
    SUCCESS = auto()
    FAILED_ALL_PROVIDERS = auto()
    RATE_LIMITED = auto()
    VALIDATION_ERROR = auto()
    TIMEOUT = auto()


@dataclass
class ProviderConfig:
    """Configuration for a payment provider"""
    provider: PaymentProvider
    processor: PaymentProcessor
    priority: int = 1                   # Lower number = higher priority
    enabled: bool = True
    max_amount: Optional[float] = None  # Maximum transaction amount
    min_amount: Optional[float] = None  # Minimum transaction amount
    supported_currencies: List[str] = field(default_factory=lambda: ["USD"])
    fallback_delay_ms: int = 1000      # Delay before trying this as fallback


@dataclass
class PaymentAttempt:
    """Record of a payment attempt"""
    provider: PaymentProvider
    attempt_number: int
    start_time: datetime
    end_time: Optional[datetime] = None
    response: Optional[PaymentResponse] = None
    error: Optional[Exception] = None
    duration_ms: Optional[int] = None
    
    def complete(self, response: Optional[PaymentResponse] = None, error: Optional[Exception] = None):
        """Mark the attempt as complete"""
        self.end_time = datetime.utcnow()
        self.response = response
        self.error = error
        if self.end_time and self.start_time:
            self.duration_ms = int((self.end_time - self.start_time).total_seconds() * 1000)


@dataclass
class PaymentResult:
    """Complete payment processing result with all attempts"""
    outcome: PaymentOutcome
    final_response: Optional[PaymentResponse] = None
    final_error: Optional[Exception] = None
    attempts: List[PaymentAttempt] = field(default_factory=list)
    total_duration_ms: int = 0
    providers_tried: List[PaymentProvider] = field(default_factory=list)
    fallback_count: int = 0


class PaymentOrchestrator:
    """
    Main payment orchestrator with comprehensive fallback mechanisms
    
    Features:
    - Multi-provider fallback with configurable strategies
    - Circuit breaker integration
    - Intelligent retry logic
    - Rate limiting and load balancing
    - Real-time health monitoring
    - Comprehensive audit trails
    """
    
    def __init__(self, fallback_strategy: FallbackStrategy = FallbackStrategy.HEALTH_BASED):
        self.fallback_strategy = fallback_strategy
        self.providers: Dict[PaymentProvider, ProviderConfig] = {}
        self.circuit_breaker_manager = CircuitBreakerManager()
        self.retry_managers: Dict[PaymentProvider, RetryManager] = {}
        self.rate_limit_manager = RateLimitManager()
        self.logger = logging.getLogger(f"{__name__}.orchestrator")
        
        # Metrics
        self.total_payments = 0
        self.successful_payments = 0
        self.failed_payments = 0
        self.fallback_usage = 0
        
    def register_provider(self, 
                         config: ProviderConfig,
                         rate_config: Optional[RateLimitConfig] = None,
                         provider_weight: Optional[ProviderWeight] = None):
        """Register a payment provider with the orchestrator"""
        provider = config.provider
        self.providers[provider] = config
        
        # Set up circuit breaker
        self.circuit_breaker_manager.register_provider(provider)
        
        # Set up retry manager
        self.retry_managers[provider] = RetryManagerFactory.create_retry_manager(provider)
        
        # Set up rate limiting
        if rate_config is None:
            rate_config = RateLimitConfig(
                requests_per_second=self._get_default_rate_limit(provider)
            )
        
        if provider_weight is None:
            provider_weight = ProviderWeight(
                provider=provider,
                weight=1.0 / (config.priority if config.priority > 0 else 1)
            )
        
        self.rate_limit_manager.register_provider(provider, rate_config, provider_weight)
        
        self.logger.info(f"Registered payment provider {provider.value}")
    
    async def process_payment(self, 
                            request: PaymentRequest,
                            timeout_seconds: int = 60,
                            priority: RequestPriority = RequestPriority.NORMAL) -> PaymentResult:
        """
        Process a payment with comprehensive fallback handling
        
        Args:
            request: Payment request details
            timeout_seconds: Maximum time to spend on payment processing
            priority: Request priority for rate limiting
            
        Returns:
            PaymentResult with complete processing details
        """
        start_time = datetime.utcnow()
        result = PaymentResult(outcome=PaymentOutcome.FAILED_ALL_PROVIDERS)
        
        try:
            self.total_payments += 1
            
            # Validate request
            if not self._validate_payment_request(request):
                result.outcome = PaymentOutcome.VALIDATION_ERROR
                result.final_error = PaymentError(
                    "Invalid payment request", 
                    PaymentProvider.STRIPE,  # Default for validation errors
                    "VALIDATION_ERROR"
                )
                return result
            
            # Get eligible providers
            eligible_providers = self._get_eligible_providers(request)
            if not eligible_providers:
                result.final_error = PaymentError(
                    "No eligible payment providers available",
                    PaymentProvider.STRIPE,
                    "NO_PROVIDERS_AVAILABLE"
                )
                return result
            
            # Sort providers by fallback strategy
            provider_order = self._sort_providers_by_strategy(eligible_providers)
            
            # Try each provider with fallback logic
            for provider_index, provider in enumerate(provider_order):
                # Check timeout
                elapsed = (datetime.utcnow() - start_time).total_seconds()
                if elapsed >= timeout_seconds:
                    result.outcome = PaymentOutcome.TIMEOUT
                    break
                
                # Apply fallback delay (except for first provider)
                if provider_index > 0:
                    delay_ms = self.providers[provider].fallback_delay_ms
                    if delay_ms > 0:
                        await asyncio.sleep(delay_ms / 1000.0)
                    result.fallback_count += 1
                    self.fallback_usage += 1
                
                # Attempt payment with this provider
                attempt_result = await self._attempt_payment_with_provider(
                    provider, request, priority, timeout_seconds - elapsed
                )
                
                result.attempts.extend(attempt_result.attempts)
                result.providers_tried.append(provider)
                
                # Check if payment succeeded
                if attempt_result.outcome == PaymentOutcome.SUCCESS:
                    result.outcome = PaymentOutcome.SUCCESS
                    result.final_response = attempt_result.final_response
                    self.successful_payments += 1
                    break
                
                # Check if we should stop trying (non-retryable error)
                if attempt_result.outcome == PaymentOutcome.VALIDATION_ERROR:
                    result.outcome = PaymentOutcome.VALIDATION_ERROR
                    result.final_error = attempt_result.final_error
                    break
                
                # Continue to next provider for other failures
                self.logger.warning(
                    f"Provider {provider.value} failed for payment {request.order_id}, "
                    f"trying next provider. Error: {attempt_result.final_error}"
                )
            
            # If no provider succeeded
            if result.outcome != PaymentOutcome.SUCCESS:
                self.failed_payments += 1
                if not result.final_error and result.attempts:
                    # Use the last error as the final error
                    result.final_error = result.attempts[-1].error
            
        except Exception as e:
            self.logger.error(f"Unexpected error in payment orchestrator: {e}")
            result.final_error = e
            self.failed_payments += 1
        
        finally:
            # Calculate total duration
            end_time = datetime.utcnow()
            result.total_duration_ms = int((end_time - start_time).total_seconds() * 1000)
        
        return result
    
    async def _attempt_payment_with_provider(self,
                                           provider: PaymentProvider,
                                           request: PaymentRequest,
                                           priority: RequestPriority,
                                           timeout_seconds: float) -> PaymentResult:
        """Attempt payment with a specific provider"""
        result = PaymentResult(outcome=PaymentOutcome.FAILED_ALL_PROVIDERS)
        
        try:
            # Check if provider is enabled
            provider_config = self.providers[provider]
            if not provider_config.enabled:
                result.final_error = PaymentError(
                    f"Provider {provider.value} is disabled",
                    provider,
                    "PROVIDER_DISABLED"
                )
                return result
            
            # Acquire rate limit permit
            if not await self.rate_limit_manager.acquire_permit(provider, priority):
                result.outcome = PaymentOutcome.RATE_LIMITED
                result.final_error = PaymentError(
                    f"Rate limit exceeded for {provider.value}",
                    provider,
                    "RATE_LIMIT_EXCEEDED",
                    is_retryable=True
                )
                return result
            
            try:
                # Get circuit breaker and retry manager
                circuit_breaker = self.circuit_breaker_manager.get_circuit_breaker(provider)
                retry_manager = self.retry_managers[provider]
                
                # Execute payment with circuit breaker and retry logic
                async def payment_operation():
                    attempt = PaymentAttempt(
                        provider=provider,
                        attempt_number=len(result.attempts) + 1,
                        start_time=datetime.utcnow()
                    )
                    result.attempts.append(attempt)
                    
                    try:
                        response = await provider_config.processor.process_payment(request)
                        attempt.complete(response=response)
                        
                        # Record successful completion
                        self.rate_limit_manager.record_request_completion(
                            provider, attempt.duration_ms or 0, True
                        )
                        
                        return response
                    
                    except Exception as e:
                        attempt.complete(error=e)
                        
                        # Record failed completion
                        self.rate_limit_manager.record_request_completion(
                            provider, attempt.duration_ms or 0, False
                        )
                        
                        raise
                
                # Execute with circuit breaker protection
                if circuit_breaker:
                    response = await circuit_breaker.call(
                        retry_manager.execute_with_retry,
                        payment_operation,
                        operation_id=f"payment_{request.order_id}"
                    )
                else:
                    response = await retry_manager.execute_with_retry(
                        payment_operation,
                        operation_id=f"payment_{request.order_id}"
                    )
                
                # Success!
                result.outcome = PaymentOutcome.SUCCESS
                result.final_response = response
                
            except CircuitBreakerOpenError as e:
                result.final_error = PaymentError(
                    f"Circuit breaker open for {provider.value}",
                    provider,
                    "CIRCUIT_BREAKER_OPEN",
                    is_retryable=True
                )
            
            except RetryExhaustedException as e:
                result.final_error = PaymentError(
                    f"All retry attempts exhausted for {provider.value}",
                    provider,
                    "RETRY_EXHAUSTED"
                )
            
            except PaymentError as e:
                if not e.is_retryable:
                    result.outcome = PaymentOutcome.VALIDATION_ERROR
                result.final_error = e
            
            except Exception as e:
                result.final_error = PaymentError(
                    f"Unexpected error with {provider.value}: {str(e)}",
                    provider,
                    "UNEXPECTED_ERROR"
                )
        
        finally:
            # Always release the rate limit permit
            self.rate_limit_manager.release_permit(provider)
        
        return result
    
    def _validate_payment_request(self, request: PaymentRequest) -> bool:
        """Validate payment request"""
        try:
            if request.amount <= 0:
                return False
            if not request.currency or len(request.currency) != 3:
                return False
            if not request.customer_id or not request.payment_method_id:
                return False
            return True
        except Exception:
            return False
    
    def _get_eligible_providers(self, request: PaymentRequest) -> List[PaymentProvider]:
        """Get providers eligible for this payment request"""
        eligible = []
        
        for provider, config in self.providers.items():
            if not config.enabled:
                continue
            
            # Check amount limits
            amount = float(request.amount)
            if config.min_amount and amount < config.min_amount:
                continue
            if config.max_amount and amount > config.max_amount:
                continue
            
            # Check currency support
            if request.currency not in config.supported_currencies:
                continue
            
            eligible.append(provider)
        
        return eligible
    
    def _sort_providers_by_strategy(self, providers: List[PaymentProvider]) -> List[PaymentProvider]:
        """Sort providers according to fallback strategy"""
        if self.fallback_strategy == FallbackStrategy.PRIORITY_ORDER:
            return sorted(providers, key=lambda p: self.providers[p].priority)
        
        elif self.fallback_strategy == FallbackStrategy.HEALTH_BASED:
            # Get healthy providers first, then by health score
            healthy = self.circuit_breaker_manager.get_healthy_providers()
            healthy_in_eligible = [p for p in providers if p in healthy]
            unhealthy_in_eligible = [p for p in providers if p not in healthy]
            
            # Sort healthy by priority, unhealthy at the end
            healthy_sorted = sorted(healthy_in_eligible, key=lambda p: self.providers[p].priority)
            unhealthy_sorted = sorted(unhealthy_in_eligible, key=lambda p: self.providers[p].priority)
            
            return healthy_sorted + unhealthy_sorted
        
        elif self.fallback_strategy == FallbackStrategy.COST_OPTIMIZED:
            # Use load balancer's cost optimization
            return providers  # Load balancer will handle cost optimization
        
        elif self.fallback_strategy == FallbackStrategy.ROUND_ROBIN:
            # Use load balancer for round-robin
            best_provider = self.rate_limit_manager.select_best_provider(providers)
            if best_provider:
                # Put best provider first, then others by priority
                others = [p for p in providers if p != best_provider]
                others_sorted = sorted(others, key=lambda p: self.providers[p].priority)
                return [best_provider] + others_sorted
            else:
                return sorted(providers, key=lambda p: self.providers[p].priority)
        
        else:
            return sorted(providers, key=lambda p: self.providers[p].priority)
    
    def _get_default_rate_limit(self, provider: PaymentProvider) -> int:
        """Get default rate limit for provider"""
        defaults = {
            PaymentProvider.STRIPE: 100,
            PaymentProvider.PAYPAL: 50,
            PaymentProvider.SQUARE: 75,
        }
        return defaults.get(provider, 50)
    
    async def refund_payment(self, 
                           transaction_id: str, 
                           refund_request: RefundRequest) -> PaymentResult:
        """Process a refund with fallback handling"""
        # Implementation would be similar to process_payment
        # but for refund operations
        pass  # Placeholder for brevity
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics from all components"""
        return {
            "orchestrator": {
                "total_payments": self.total_payments,
                "successful_payments": self.successful_payments,
                "failed_payments": self.failed_payments,
                "success_rate": self.successful_payments / max(self.total_payments, 1),
                "fallback_usage": self.fallback_usage,
                "fallback_strategy": self.fallback_strategy.name,
            },
            "circuit_breakers": self.circuit_breaker_manager.get_all_metrics(),
            "rate_limiters": self.rate_limit_manager.get_all_metrics(),
            "providers": {
                provider.value: {
                    "enabled": config.enabled,
                    "priority": config.priority,
                    "max_amount": config.max_amount,
                    "min_amount": config.min_amount,
                    "supported_currencies": config.supported_currencies,
                }
                for provider, config in self.providers.items()
            }
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status"""
        healthy_providers = self.circuit_breaker_manager.get_healthy_providers()
        total_providers = len(self.providers)
        
        return {
            "healthy_providers": len(healthy_providers),
            "total_providers": total_providers,
            "health_percentage": len(healthy_providers) / max(total_providers, 1) * 100,
            "providers_status": {
                provider.value: provider in healthy_providers
                for provider in self.providers.keys()
            }
        }