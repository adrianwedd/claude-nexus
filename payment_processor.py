"""
Core Payment Processor Interface and Abstract Implementation

This module defines the foundational interfaces for payment processing
with built-in resilience patterns and error handling strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum, auto
from typing import Dict, List, Optional, Union, Any
import asyncio
import logging
import uuid

logger = logging.getLogger(__name__)


class PaymentStatus(Enum):
    """Payment processing status enumeration"""
    PENDING = auto()
    PROCESSING = auto()
    SUCCEEDED = auto()
    FAILED = auto()
    CANCELLED = auto()
    REFUNDED = auto()
    PARTIALLY_REFUNDED = auto()


class PaymentProvider(Enum):
    """Supported payment providers"""
    STRIPE = "stripe"
    PAYPAL = "paypal"
    SQUARE = "square"


class PaymentError(Exception):
    """Base payment processing error"""
    def __init__(self, message: str, provider: PaymentProvider, 
                 error_code: Optional[str] = None, 
                 is_retryable: bool = False):
        super().__init__(message)
        self.provider = provider
        self.error_code = error_code
        self.is_retryable = is_retryable
        self.timestamp = datetime.utcnow()


class RateLimitError(PaymentError):
    """Rate limit exceeded error"""
    def __init__(self, provider: PaymentProvider, retry_after: Optional[int] = None):
        super().__init__(
            f"Rate limit exceeded for {provider.value}",
            provider=provider,
            error_code="RATE_LIMIT_EXCEEDED",
            is_retryable=True
        )
        self.retry_after = retry_after


class NetworkError(PaymentError):
    """Network connectivity error"""
    def __init__(self, provider: PaymentProvider, underlying_error: str):
        super().__init__(
            f"Network error for {provider.value}: {underlying_error}",
            provider=provider,
            error_code="NETWORK_ERROR",
            is_retryable=True
        )


class ValidationError(PaymentError):
    """Payment validation error"""
    def __init__(self, provider: PaymentProvider, message: str):
        super().__init__(
            message,
            provider=provider,
            error_code="VALIDATION_ERROR",
            is_retryable=False
        )


@dataclass
class PaymentRequest:
    """Payment processing request"""
    amount: Decimal
    currency: str
    customer_id: str
    payment_method_id: str
    order_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    idempotency_key: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def __post_init__(self):
        if self.amount <= 0:
            raise ValueError("Payment amount must be positive")
        if not self.currency or len(self.currency) != 3:
            raise ValueError("Currency must be a 3-letter code")


@dataclass
class PaymentResponse:
    """Payment processing response"""
    transaction_id: str
    status: PaymentStatus
    provider: PaymentProvider
    amount: Decimal
    currency: str
    provider_transaction_id: Optional[str] = None
    fees: Optional[Decimal] = None
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RefundRequest:
    """Refund processing request"""
    transaction_id: str
    amount: Optional[Decimal] = None  # None for full refund
    reason: Optional[str] = None
    idempotency_key: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class WebhookEvent:
    """Webhook event data structure"""
    event_id: str
    provider: PaymentProvider
    event_type: str
    transaction_id: str
    status: PaymentStatus
    payload: Dict[str, Any]
    signature: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    processed: bool = False
    retry_count: int = 0


class PaymentProcessor(ABC):
    """Abstract base class for payment processors"""
    
    def __init__(self, provider: PaymentProvider, config: Dict[str, Any]):
        self.provider = provider
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{provider.value}")
    
    @abstractmethod
    async def process_payment(self, request: PaymentRequest) -> PaymentResponse:
        """Process a payment request"""
        pass
    
    @abstractmethod
    async def refund_payment(self, request: RefundRequest) -> PaymentResponse:
        """Process a refund request"""
        pass
    
    @abstractmethod
    async def get_payment_status(self, transaction_id: str) -> PaymentResponse:
        """Get current payment status"""
        pass
    
    @abstractmethod
    async def verify_webhook(self, event: WebhookEvent) -> bool:
        """Verify webhook signature and authenticity"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the payment provider is healthy"""
        pass
    
    def get_rate_limit(self) -> int:
        """Get the rate limit for this provider (requests per second)"""
        return self.config.get('rate_limit', 100)
    
    def get_timeout(self) -> int:
        """Get the timeout for requests in seconds"""
        return self.config.get('timeout', 30)
    
    def is_retryable_error(self, error: Exception) -> bool:
        """Determine if an error is retryable"""
        if isinstance(error, PaymentError):
            return error.is_retryable
        if isinstance(error, (asyncio.TimeoutError, ConnectionError)):
            return True
        return False