"""
Comprehensive Integration Tests for Payment Processing System

This module contains extensive integration tests covering:
- Failure scenario testing
- Circuit breaker behavior
- Retry logic validation
- Rate limiting verification
- Fallback mechanism testing
- Webhook processing reliability
- End-to-end system resilience
"""

import asyncio
import json
import pytest
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from payment_processor import (
    PaymentProcessor, PaymentProvider, PaymentRequest, PaymentResponse, 
    PaymentStatus, PaymentError, RateLimitError, NetworkError, ValidationError
)
from circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerOpenError
from retry_strategy import RetryManager, RetryConfig, RetryExhaustedException
from rate_limiter import RateLimitManager, RateLimitConfig, ProviderWeight, RequestPriority
from payment_orchestrator import PaymentOrchestrator, ProviderConfig, FallbackStrategy, PaymentOutcome
from webhook_processor import WebhookEventProcessor, WebhookConfig, WebhookEvent, EventPriority
from monitoring_system import MonitoringSystem, Alert, AlertSeverity


class MockPaymentProcessor(PaymentProcessor):
    """Mock payment processor for testing"""
    
    def __init__(self, provider: PaymentProvider, config: Dict[str, Any]):
        super().__init__(provider, config)
        self.should_fail = False
        self.failure_type = None
        self.response_delay = 0
        self.call_count = 0
        self.rate_limit_after = None
        
    async def process_payment(self, request: PaymentRequest) -> PaymentResponse:
        self.call_count += 1
        
        # Simulate response delay
        if self.response_delay > 0:
            await asyncio.sleep(self.response_delay)
        
        # Simulate rate limiting
        if self.rate_limit_after and self.call_count >= self.rate_limit_after:
            raise RateLimitError(self.provider, retry_after=5)
        
        # Simulate failures
        if self.should_fail:
            if self.failure_type == "network":
                raise NetworkError(self.provider, "Connection timeout")
            elif self.failure_type == "validation":
                raise ValidationError(self.provider, "Invalid payment method")
            elif self.failure_type == "rate_limit":
                raise RateLimitError(self.provider, retry_after=10)
            else:
                raise PaymentError("Generic payment error", self.provider, is_retryable=True)
        
        # Successful response
        return PaymentResponse(
            transaction_id=f"txn_{self.provider.value}_{self.call_count}",
            status=PaymentStatus.SUCCEEDED,
            provider=self.provider,
            amount=request.amount,
            currency=request.currency,
            provider_transaction_id=f"provider_txn_{self.call_count}"
        )
    
    async def refund_payment(self, request) -> PaymentResponse:
        return PaymentResponse(
            transaction_id=f"refund_{self.provider.value}_{self.call_count}",
            status=PaymentStatus.REFUNDED,
            provider=self.provider,
            amount=request.amount or Decimal("10.00"),
            currency="USD"
        )
    
    async def get_payment_status(self, transaction_id: str) -> PaymentResponse:
        return PaymentResponse(
            transaction_id=transaction_id,
            status=PaymentStatus.SUCCEEDED,
            provider=self.provider,
            amount=Decimal("10.00"),
            currency="USD"
        )
    
    async def verify_webhook(self, event) -> bool:
        return True
    
    async def health_check(self) -> bool:
        return not self.should_fail


class TestCircuitBreakerIntegration:
    """Test circuit breaker integration with payment processing"""
    
    @pytest.fixture
    def circuit_breaker(self):
        config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout_seconds=1,
            rolling_window_size=10
        )
        return CircuitBreaker(PaymentProvider.STRIPE, config)
    
    @pytest.fixture
    def mock_processor(self):
        return MockPaymentProcessor(PaymentProvider.STRIPE, {})
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(self, circuit_breaker, mock_processor):
        """Test that circuit breaker opens after consecutive failures"""
        mock_processor.should_fail = True
        mock_processor.failure_type = "network"
        
        # Function to test
        async def failing_operation():
            return await mock_processor.process_payment(PaymentRequest(
                amount=Decimal("10.00"),
                currency="USD",
                customer_id="cust_123",
                payment_method_id="pm_123",
                order_id="order_123"
            ))
        
        # Should fail 3 times then circuit opens
        for i in range(3):
            try:
                await circuit_breaker.call(failing_operation)
            except (PaymentError, NetworkError):
                pass
        
        # Next call should be blocked by circuit breaker
        with pytest.raises(CircuitBreakerOpenError):
            await circuit_breaker.call(failing_operation)
        
        assert circuit_breaker.state.name == "OPEN"
        assert circuit_breaker.metrics.consecutive_failures == 3
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self, circuit_breaker, mock_processor):
        """Test circuit breaker recovery after timeout"""
        # First, open the circuit
        mock_processor.should_fail = True
        
        async def failing_operation():
            return await mock_processor.process_payment(PaymentRequest(
                amount=Decimal("10.00"),
                currency="USD",
                customer_id="cust_123",
                payment_method_id="pm_123",
                order_id="order_123"
            ))
        
        # Fail enough times to open circuit
        for _ in range(3):
            try:
                await circuit_breaker.call(failing_operation)
            except (PaymentError, NetworkError):
                pass
        
        assert circuit_breaker.state.name == "OPEN"
        
        # Wait for timeout
        await asyncio.sleep(1.1)
        
        # Fix the processor
        mock_processor.should_fail = False
        
        # Should transition to half-open and then closed
        result = await circuit_breaker.call(failing_operation)
        assert result is not None
        assert circuit_breaker.state.name == "HALF_OPEN"
        
        # Another success should close the circuit
        await circuit_breaker.call(failing_operation)
        assert circuit_breaker.state.name == "CLOSED"


class TestRetryStrategyIntegration:
    """Test retry strategy integration"""
    
    @pytest.fixture
    def retry_manager(self):
        config = RetryConfig(
            max_attempts=3,
            base_delay_ms=100,
            exponential_base=2.0,
            timeout_seconds=5
        )
        return RetryManager(PaymentProvider.STRIPE, config)
    
    @pytest.fixture
    def mock_processor(self):
        return MockPaymentProcessor(PaymentProvider.STRIPE, {})
    
    @pytest.mark.asyncio
    async def test_retry_on_transient_failures(self, retry_manager, mock_processor):
        """Test retries on transient failures"""
        mock_processor.should_fail = True
        mock_processor.failure_type = "network"
        
        # Fail twice, then succeed
        call_count = 0
        
        async def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise NetworkError(PaymentProvider.STRIPE, "Temporary network error")
            return {"success": True}
        
        result = await retry_manager.execute_with_retry(flaky_operation)
        
        assert result["success"] is True
        assert call_count == 3  # Failed twice, succeeded on third attempt
    
    @pytest.mark.asyncio
    async def test_no_retry_on_validation_errors(self, retry_manager, mock_processor):
        """Test no retries on validation errors"""
        async def validation_error_operation():
            raise ValidationError(PaymentProvider.STRIPE, "Invalid payment method")
        
        with pytest.raises(ValidationError):
            await retry_manager.execute_with_retry(validation_error_operation)
        
        # Should not retry validation errors
    
    @pytest.mark.asyncio
    async def test_retry_exhaustion(self, retry_manager, mock_processor):
        """Test behavior when all retries are exhausted"""
        async def always_failing_operation():
            raise NetworkError(PaymentProvider.STRIPE, "Network error")
        
        with pytest.raises(RetryExhaustedException):
            await retry_manager.execute_with_retry(always_failing_operation)


class TestRateLimitingIntegration:
    """Test rate limiting integration"""
    
    @pytest.fixture
    def rate_limit_manager(self):
        manager = RateLimitManager()
        
        config = RateLimitConfig(
            requests_per_second=5,  # Low limit for testing
            burst_capacity=5,
            queue_max_size=10
        )
        
        weight = ProviderWeight(
            provider=PaymentProvider.STRIPE,
            weight=1.0
        )
        
        manager.register_provider(PaymentProvider.STRIPE, config, weight)
        return manager
    
    @pytest.mark.asyncio
    async def test_rate_limiting_blocks_excess_requests(self, rate_limit_manager):
        """Test that rate limiting blocks excess requests"""
        # Rapid-fire requests should hit rate limit
        permits_granted = 0
        permits_denied = 0
        
        tasks = []
        for _ in range(20):  # Request more than rate limit
            task = asyncio.create_task(
                rate_limit_manager.acquire_permit(PaymentProvider.STRIPE)
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        permits_granted = sum(1 for result in results if result)
        permits_denied = sum(1 for result in results if not result)
        
        assert permits_granted <= 10  # Should not exceed burst capacity + queue
        assert permits_denied > 0  # Some requests should be denied
    
    @pytest.mark.asyncio
    async def test_rate_limit_recovery(self, rate_limit_manager):
        """Test that rate limiting recovers over time"""
        # Use up the burst capacity
        for _ in range(5):
            await rate_limit_manager.acquire_permit(PaymentProvider.STRIPE)
        
        # Next request should be denied
        assert not await rate_limit_manager.acquire_permit(PaymentProvider.STRIPE)
        
        # Wait for token bucket to refill
        await asyncio.sleep(1.0)
        
        # Should be able to get permits again
        assert await rate_limit_manager.acquire_permit(PaymentProvider.STRIPE)


class TestPaymentOrchestratorIntegration:
    """Test payment orchestrator with all resilience patterns"""
    
    @pytest.fixture
    def orchestrator(self):
        orchestrator = PaymentOrchestrator(FallbackStrategy.HEALTH_BASED)
        
        # Register multiple providers
        for provider in [PaymentProvider.STRIPE, PaymentProvider.PAYPAL, PaymentProvider.SQUARE]:
            config = ProviderConfig(
                provider=provider,
                processor=MockPaymentProcessor(provider, {}),
                priority=provider.value,  # Different priorities
                enabled=True
            )
            
            rate_config = RateLimitConfig(requests_per_second=100)
            weight = ProviderWeight(provider=provider, weight=1.0)
            
            orchestrator.register_provider(config, rate_config, weight)
        
        return orchestrator
    
    @pytest.mark.asyncio
    async def test_successful_payment_processing(self, orchestrator):
        """Test successful payment processing"""
        request = PaymentRequest(
            amount=Decimal("100.00"),
            currency="USD",
            customer_id="cust_123",
            payment_method_id="pm_123",
            order_id="order_123"
        )
        
        result = await orchestrator.process_payment(request)
        
        assert result.outcome == PaymentOutcome.SUCCESS
        assert result.final_response is not None
        assert result.final_response.status == PaymentStatus.SUCCEEDED
        assert len(result.providers_tried) == 1  # Should succeed with first provider
    
    @pytest.mark.asyncio
    async def test_fallback_on_provider_failure(self, orchestrator):
        """Test fallback to next provider when first fails"""
        # Make Stripe fail
        stripe_processor = None
        for provider, config in orchestrator.providers.items():
            if provider == PaymentProvider.STRIPE:
                stripe_processor = config.processor
                break
        
        stripe_processor.should_fail = True
        stripe_processor.failure_type = "network"
        
        request = PaymentRequest(
            amount=Decimal("100.00"),
            currency="USD",
            customer_id="cust_123",
            payment_method_id="pm_123",
            order_id="order_123"
        )
        
        result = await orchestrator.process_payment(request)
        
        assert result.outcome == PaymentOutcome.SUCCESS
        assert len(result.providers_tried) > 1  # Should have tried multiple providers
        assert result.fallback_count > 0
    
    @pytest.mark.asyncio
    async def test_all_providers_fail(self, orchestrator):
        """Test behavior when all providers fail"""
        # Make all providers fail
        for provider, config in orchestrator.providers.items():
            config.processor.should_fail = True
            config.processor.failure_type = "network"
        
        request = PaymentRequest(
            amount=Decimal("100.00"),
            currency="USD",
            customer_id="cust_123",
            payment_method_id="pm_123",
            order_id="order_123"
        )
        
        result = await orchestrator.process_payment(request)
        
        assert result.outcome == PaymentOutcome.FAILED_ALL_PROVIDERS
        assert len(result.providers_tried) == len(orchestrator.providers)
        assert result.final_error is not None
    
    @pytest.mark.asyncio
    async def test_validation_error_no_fallback(self, orchestrator):
        """Test that validation errors don't trigger fallback"""
        # Make Stripe return validation error
        stripe_processor = None
        for provider, config in orchestrator.providers.items():
            if provider == PaymentProvider.STRIPE:
                stripe_processor = config.processor
                break
        
        stripe_processor.should_fail = True
        stripe_processor.failure_type = "validation"
        
        request = PaymentRequest(
            amount=Decimal("100.00"),
            currency="USD",
            customer_id="cust_123",
            payment_method_id="pm_123",
            order_id="order_123"
        )
        
        result = await orchestrator.process_payment(request)
        
        assert result.outcome == PaymentOutcome.VALIDATION_ERROR
        assert len(result.providers_tried) == 1  # Should not try other providers


class TestWebhookProcessorIntegration:
    """Test webhook processor integration"""
    
    @pytest.fixture
    def webhook_processor(self):
        config = WebhookConfig(
            max_retries=3,
            retry_delay_base_ms=100,
            processing_timeout_seconds=5,
            signature_verification=False  # Disable for testing
        )
        
        processor = WebhookEventProcessor(config, {})
        
        # Register test handlers
        async def test_handler(event):
            if event.event_type == "payment.succeeded":
                return {"processed": True}
            elif event.event_type == "payment.failed":
                raise Exception("Processing error")
            else:
                return {"processed": True}
        
        processor.register_event_handler("payment.succeeded", test_handler)
        processor.register_event_handler("payment.failed", test_handler)
        
        return processor
    
    @pytest.mark.asyncio
    async def test_successful_webhook_processing(self, webhook_processor):
        """Test successful webhook event processing"""
        event = WebhookEvent(
            event_id="evt_123",
            provider=PaymentProvider.STRIPE,
            event_type="payment.succeeded",
            transaction_id="txn_123",
            status=PaymentStatus.SUCCEEDED,
            payload={"amount": 1000, "currency": "usd"},
            signature="test_signature"
        )
        
        await webhook_processor.start_processing()
        
        # Process the event
        accepted = await webhook_processor.process_webhook(event, b'{"test": "data"}')
        assert accepted is True
        
        # Wait for processing
        await asyncio.sleep(0.5)
        
        metrics = webhook_processor.get_metrics()
        assert metrics["events_received"] == 1
        assert metrics["events_processed"] == 1
        assert metrics["events_failed"] == 0
        
        await webhook_processor.stop_processing()
    
    @pytest.mark.asyncio
    async def test_webhook_retry_on_failure(self, webhook_processor):
        """Test webhook retry logic on processing failures"""
        event = WebhookEvent(
            event_id="evt_456",
            provider=PaymentProvider.STRIPE,
            event_type="payment.failed",  # This will trigger an error in our handler
            transaction_id="txn_456",
            status=PaymentStatus.FAILED,
            payload={"error": "card_declined"},
            signature="test_signature"
        )
        
        await webhook_processor.start_processing()
        
        # Process the failing event
        accepted = await webhook_processor.process_webhook(event, b'{"test": "data"}')
        assert accepted is True
        
        # Wait for processing and retries
        await asyncio.sleep(2.0)
        
        metrics = webhook_processor.get_metrics()
        assert metrics["events_received"] == 1
        assert metrics["events_failed"] > 0
        
        # Event should eventually end up in dead letter queue
        dead_letter_events = webhook_processor.get_dead_letter_events()
        assert len(dead_letter_events) > 0
        
        await webhook_processor.stop_processing()
    
    @pytest.mark.asyncio
    async def test_webhook_deduplication(self, webhook_processor):
        """Test webhook event deduplication"""
        event = WebhookEvent(
            event_id="evt_duplicate",
            provider=PaymentProvider.STRIPE,
            event_type="payment.succeeded",
            transaction_id="txn_duplicate",
            status=PaymentStatus.SUCCEEDED,
            payload={"amount": 1000},
            signature="test_signature"
        )
        
        await webhook_processor.start_processing()
        
        # Process the same event twice
        accepted1 = await webhook_processor.process_webhook(event, b'{"test": "data"}')
        accepted2 = await webhook_processor.process_webhook(event, b'{"test": "data"}')
        
        assert accepted1 is True
        assert accepted2 is True  # Duplicate is accepted but not re-processed
        
        # Wait for processing
        await asyncio.sleep(0.5)
        
        metrics = webhook_processor.get_metrics()
        assert metrics["events_received"] == 2
        assert metrics["events_duplicate"] == 1
        
        await webhook_processor.stop_processing()


class TestEndToEndIntegration:
    """End-to-end integration tests"""
    
    @pytest.fixture
    def complete_system(self):
        """Setup complete payment processing system"""
        # Payment orchestrator
        orchestrator = PaymentOrchestrator(FallbackStrategy.HEALTH_BASED)
        
        # Monitoring system
        monitoring = MonitoringSystem()
        
        # Register providers with different reliability characteristics
        providers_config = [
            (PaymentProvider.STRIPE, {"failure_rate": 0.1, "latency_ms": 500}),
            (PaymentProvider.PAYPAL, {"failure_rate": 0.05, "latency_ms": 800}),
            (PaymentProvider.SQUARE, {"failure_rate": 0.15, "latency_ms": 300}),
        ]
        
        for provider, characteristics in providers_config:
            processor = MockPaymentProcessor(provider, characteristics)
            config = ProviderConfig(
                provider=provider,
                processor=processor,
                priority=len(providers_config) - providers_config.index((provider, characteristics)),
                enabled=True
            )
            
            rate_config = RateLimitConfig(requests_per_second=50)
            weight = ProviderWeight(provider=provider, weight=1.0)
            
            orchestrator.register_provider(config, rate_config, weight)
        
        return {
            "orchestrator": orchestrator,
            "monitoring": monitoring
        }
    
    @pytest.mark.asyncio
    async def test_high_volume_processing(self, complete_system):
        """Test system behavior under high volume"""
        orchestrator = complete_system["orchestrator"]
        monitoring = complete_system["monitoring"]
        
        await monitoring.start_monitoring()
        
        # Generate many payment requests
        requests = []
        for i in range(100):
            request = PaymentRequest(
                amount=Decimal(f"{10 + i}.00"),
                currency="USD",
                customer_id=f"cust_{i}",
                payment_method_id=f"pm_{i}",
                order_id=f"order_{i}"
            )
            requests.append(request)
        
        # Process payments concurrently
        start_time = time.time()
        tasks = [orchestrator.process_payment(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        # Analyze results
        successful_results = [r for r in results if isinstance(r, PaymentResult) and r.outcome == PaymentOutcome.SUCCESS]
        failed_results = [r for r in results if isinstance(r, PaymentResult) and r.outcome != PaymentOutcome.SUCCESS]
        
        success_rate = len(successful_results) / len(results)
        avg_processing_time = sum(r.total_duration_ms for r in successful_results) / len(successful_results)
        total_processing_time = end_time - start_time
        
        print(f"Processed {len(requests)} payments in {total_processing_time:.2f}s")
        print(f"Success rate: {success_rate:.2%}")
        print(f"Average processing time: {avg_processing_time:.2f}ms")
        print(f"Throughput: {len(requests) / total_processing_time:.2f} payments/second")
        
        # Verify system targets
        assert success_rate >= 0.95  # At least 95% success rate
        assert avg_processing_time <= 2000  # Under 2 seconds average
        
        # Check system metrics
        metrics = orchestrator.get_comprehensive_metrics()
        assert metrics["orchestrator"]["success_rate"] >= 0.95
        
        await monitoring.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_provider_failure_scenario(self, complete_system):
        """Test system resilience when a provider fails completely"""
        orchestrator = complete_system["orchestrator"]
        
        # Make Stripe fail completely
        stripe_config = orchestrator.providers[PaymentProvider.STRIPE]
        stripe_config.processor.should_fail = True
        stripe_config.processor.failure_type = "network"
        
        # Process payments
        requests = []
        for i in range(20):
            request = PaymentRequest(
                amount=Decimal("50.00"),
                currency="USD",
                customer_id=f"cust_{i}",
                payment_method_id=f"pm_{i}",
                order_id=f"order_{i}"
            )
            requests.append(request)
        
        tasks = [orchestrator.process_payment(req) for req in requests]
        results = await asyncio.gather(*tasks)
        
        # Should still have high success rate due to fallback
        successful_results = [r for r in results if r.outcome == PaymentOutcome.SUCCESS]
        success_rate = len(successful_results) / len(results)
        
        assert success_rate >= 0.90  # Should maintain 90%+ success rate
        
        # Verify fallback was used
        fallback_used = sum(1 for r in results if r.fallback_count > 0)
        assert fallback_used > 0
    
    @pytest.mark.asyncio
    async def test_system_recovery_after_outage(self, complete_system):
        """Test system recovery after complete outage"""
        orchestrator = complete_system["orchestrator"]
        
        # Simulate complete outage
        for provider, config in orchestrator.providers.items():
            config.processor.should_fail = True
            config.processor.failure_type = "network"
        
        # Try to process payment during outage
        request = PaymentRequest(
            amount=Decimal("25.00"),
            currency="USD",
            customer_id="cust_outage",
            payment_method_id="pm_outage",
            order_id="order_outage"
        )
        
        result = await orchestrator.process_payment(request)
        assert result.outcome == PaymentOutcome.FAILED_ALL_PROVIDERS
        
        # Simulate recovery
        for provider, config in orchestrator.providers.items():
            config.processor.should_fail = False
        
        # Wait for circuit breakers to reset
        await asyncio.sleep(2)
        
        # Try payment again
        result = await orchestrator.process_payment(request)
        assert result.outcome == PaymentOutcome.SUCCESS


# Test utilities
def create_sample_payment_request(order_id: str = None) -> PaymentRequest:
    """Create a sample payment request for testing"""
    return PaymentRequest(
        amount=Decimal("10.00"),
        currency="USD",
        customer_id="cust_test",
        payment_method_id="pm_test",
        order_id=order_id or f"order_{int(time.time())}"
    )


def create_sample_webhook_event(event_type: str = "payment.succeeded") -> WebhookEvent:
    """Create a sample webhook event for testing"""
    return WebhookEvent(
        event_id=f"evt_{int(time.time())}",
        provider=PaymentProvider.STRIPE,
        event_type=event_type,
        transaction_id="txn_test",
        status=PaymentStatus.SUCCEEDED,
        payload={"amount": 1000, "currency": "usd"},
        signature="test_signature"
    )


if __name__ == "__main__":
    # Run basic integration test
    async def main():
        print("Running basic integration tests...")
        
        # Test circuit breaker
        circuit_breaker = CircuitBreaker(
            PaymentProvider.STRIPE,
            CircuitBreakerConfig(failure_threshold=2, timeout_seconds=1)
        )
        
        mock_processor = MockPaymentProcessor(PaymentProvider.STRIPE, {})
        mock_processor.should_fail = True
        
        # Test circuit opening
        for i in range(3):
            try:
                await circuit_breaker.call(mock_processor.process_payment, create_sample_payment_request())
            except Exception as e:
                print(f"Attempt {i+1} failed: {e}")
        
        print(f"Circuit breaker state: {circuit_breaker.state.name}")
        print("Basic integration test completed!")
    
    asyncio.run(main())