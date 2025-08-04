"""
Webhook Reliability System with Event Processing

This module implements a robust webhook processing system with:
- Idempotent event processing
- Reliable delivery guarantees
- Dead letter queues for failed events
- Event ordering and deduplication
- Signature verification and security
"""

import asyncio
import hashlib
import hmac
import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Union
import logging

from payment_processor import PaymentProvider, PaymentStatus, WebhookEvent

logger = logging.getLogger(__name__)


class EventStatus(Enum):
    """Webhook event processing status"""
    PENDING = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    FAILED = auto()
    DEAD_LETTER = auto()
    DUPLICATE = auto()


class EventPriority(Enum):
    """Event processing priority"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class WebhookConfig:
    """Webhook processing configuration"""
    max_retries: int = 5
    retry_delay_base_ms: int = 1000
    retry_delay_max_ms: int = 60000
    processing_timeout_seconds: int = 30
    signature_verification: bool = True
    idempotency_window_hours: int = 24
    dead_letter_threshold: int = 10
    batch_size: int = 100
    max_queue_size: int = 10000
    ordering_enabled: bool = True
    deduplication_enabled: bool = True


@dataclass
class ProcessedEvent:
    """Record of a processed webhook event"""
    event_id: str
    provider: PaymentProvider
    event_type: str
    status: EventStatus
    processing_attempts: int = 0
    first_attempt: datetime = field(default_factory=datetime.utcnow)
    last_attempt: Optional[datetime] = None
    completion_time: Optional[datetime] = None
    error_message: Optional[str] = None
    processing_duration_ms: Optional[int] = None
    handler_result: Optional[Any] = None


@dataclass
class QueuedWebhookEvent:
    """Webhook event with queueing metadata"""
    event: WebhookEvent
    priority: EventPriority
    queued_at: datetime
    retry_count: int = 0
    next_retry_at: Optional[datetime] = None
    processing_timeout: Optional[datetime] = None
    
    def __lt__(self, other):
        # Higher priority first, then FIFO within priority
        if self.priority != other.priority:
            return self.priority.value > other.priority.value
        return self.queued_at < other.queued_at


class WebhookSignatureVerifier:
    """Webhook signature verification for different providers"""
    
    def __init__(self, secrets: Dict[PaymentProvider, str]):
        self.secrets = secrets
        self.logger = logging.getLogger(f"{__name__}.verifier")
    
    def verify_signature(self, event: WebhookEvent, request_body: bytes) -> bool:
        """Verify webhook signature based on provider"""
        if event.provider not in self.secrets:
            self.logger.warning(f"No secret configured for {event.provider.value}")
            return False
        
        secret = self.secrets[event.provider]
        
        try:
            if event.provider == PaymentProvider.STRIPE:
                return self._verify_stripe_signature(event.signature, request_body, secret)
            elif event.provider == PaymentProvider.PAYPAL:
                return self._verify_paypal_signature(event.signature, request_body, secret)
            elif event.provider == PaymentProvider.SQUARE:
                return self._verify_square_signature(event.signature, request_body, secret)
            else:
                self.logger.error(f"Unknown provider for signature verification: {event.provider}")
                return False
        
        except Exception as e:
            self.logger.error(f"Signature verification error for {event.provider.value}: {e}")
            return False
    
    def _verify_stripe_signature(self, signature: str, body: bytes, secret: str) -> bool:
        """Verify Stripe webhook signature"""
        try:
            # Stripe signature format: t=timestamp,v1=signature
            elements = signature.split(',')
            timestamp = None
            signatures = []
            
            for element in elements:
                key, value = element.split('=', 1)
                if key == 't':
                    timestamp = int(value)
                elif key.startswith('v'):
                    signatures.append(value)
            
            if not timestamp or not signatures:
                return False
            
            # Check timestamp (reject if older than 5 minutes)
            current_time = int(time.time())
            if current_time - timestamp > 300:
                return False
            
            # Verify signature
            payload = f"{timestamp}.{body.decode('utf-8')}"
            expected_signature = hmac.new(
                secret.encode('utf-8'),
                payload.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            return any(hmac.compare_digest(expected_signature, sig) for sig in signatures)
        
        except Exception:
            return False
    
    def _verify_paypal_signature(self, signature: str, body: bytes, secret: str) -> bool:
        """Verify PayPal webhook signature"""
        try:
            # PayPal uses HMAC-SHA256
            expected_signature = hmac.new(
                secret.encode('utf-8'),
                body,
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(signature, expected_signature)
        
        except Exception:
            return False
    
    def _verify_square_signature(self, signature: str, body: bytes, secret: str) -> bool:
        """Verify Square webhook signature"""
        try:
            # Square uses HMAC-SHA1
            expected_signature = hmac.new(
                secret.encode('utf-8'),
                body,
                hashlib.sha1
            ).hexdigest()
            
            return hmac.compare_digest(signature, expected_signature)
        
        except Exception:
            return False


class EventDeduplicator:
    """Handles event deduplication and idempotency"""
    
    def __init__(self, window_hours: int = 24):
        self.window_hours = window_hours
        self.processed_events: Dict[str, ProcessedEvent] = {}
        self.event_timestamps: deque = deque()
        self._lock = asyncio.Lock()
    
    async def is_duplicate(self, event: WebhookEvent) -> bool:
        """Check if event is a duplicate within the idempotency window"""
        async with self._lock:
            await self._cleanup_old_events()
            return event.event_id in self.processed_events
    
    async def record_event(self, event: WebhookEvent, status: EventStatus):
        """Record an event as processed"""
        async with self._lock:
            processed_event = ProcessedEvent(
                event_id=event.event_id,
                provider=event.provider,
                event_type=event.event_type,
                status=status
            )
            
            self.processed_events[event.event_id] = processed_event
            self.event_timestamps.append((event.event_id, datetime.utcnow()))
    
    async def _cleanup_old_events(self):
        """Remove events outside the idempotency window"""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.window_hours)
        
        while self.event_timestamps and self.event_timestamps[0][1] < cutoff_time:
            event_id, _ = self.event_timestamps.popleft()
            self.processed_events.pop(event_id, None)
    
    def get_processed_count(self) -> int:
        """Get number of processed events in memory"""
        return len(self.processed_events)


class EventOrdering:
    """Handles ordered event processing per transaction"""
    
    def __init__(self):
        self.transaction_queues: Dict[str, deque] = defaultdict(deque)
        self.processing_transactions: Set[str] = set()
        self._lock = asyncio.Lock()
    
    async def queue_event(self, event: WebhookEvent) -> bool:
        """Queue event for ordered processing"""
        async with self._lock:
            transaction_id = event.transaction_id
            
            # Add to transaction queue
            self.transaction_queues[transaction_id].append(event)
            
            # Return True if this transaction is ready for processing
            return transaction_id not in self.processing_transactions
    
    async def get_next_event(self, transaction_id: str) -> Optional[WebhookEvent]:
        """Get next event for a transaction"""
        async with self._lock:
            if transaction_id in self.transaction_queues and self.transaction_queues[transaction_id]:
                return self.transaction_queues[transaction_id].popleft()
            return None
    
    async def mark_transaction_processing(self, transaction_id: str):
        """Mark transaction as being processed"""
        async with self._lock:
            self.processing_transactions.add(transaction_id)
    
    async def mark_transaction_complete(self, transaction_id: str):
        """Mark transaction processing as complete"""
        async with self._lock:
            self.processing_transactions.discard(transaction_id)
            
            # Clean up empty queues
            if transaction_id in self.transaction_queues and not self.transaction_queues[transaction_id]:
                del self.transaction_queues[transaction_id]
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        return {
            "transaction_queues": len(self.transaction_queues),
            "processing_transactions": len(self.processing_transactions),
            "total_queued_events": sum(len(queue) for queue in self.transaction_queues.values()),
        }


class WebhookEventProcessor:
    """
    Comprehensive webhook event processor with reliability guarantees
    
    Features:
    - Signature verification
    - Event deduplication
    - Ordered processing
    - Retry logic with exponential backoff
    - Dead letter queue
    - Performance monitoring
    """
    
    def __init__(self, config: WebhookConfig, webhook_secrets: Dict[PaymentProvider, str]):
        self.config = config
        self.signature_verifier = WebhookSignatureVerifier(webhook_secrets)
        self.deduplicator = EventDeduplicator(config.idempotency_window_hours)
        self.event_ordering = EventOrdering()
        
        # Event handlers registry
        self.event_handlers: Dict[str, Callable] = {}
        
        # Processing queues
        self.event_queue: List[QueuedWebhookEvent] = []
        self.dead_letter_queue: List[QueuedWebhookEvent] = []
        self.processing_pool = set()
        
        # Metrics
        self.metrics = {
            "events_received": 0,
            "events_processed": 0,
            "events_failed": 0,
            "events_duplicate": 0,
            "events_dead_letter": 0,
            "signature_failures": 0,
            "processing_time_ms": [],
        }
        
        self.logger = logging.getLogger(f"{__name__}.processor")
        self._shutdown_event = asyncio.Event()
        self._processor_task: Optional[asyncio.Task] = None
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """Register an event handler for a specific event type"""
        self.event_handlers[event_type] = handler
        self.logger.info(f"Registered handler for event type: {event_type}")
    
    async def process_webhook(self, 
                            event: WebhookEvent, 
                            request_body: bytes,
                            priority: EventPriority = EventPriority.NORMAL) -> bool:
        """
        Process incoming webhook event
        
        Args:
            event: Webhook event data
            request_body: Raw request body for signature verification
            priority: Processing priority
            
        Returns:
            True if event was accepted for processing, False otherwise
        """
        self.metrics["events_received"] += 1
        
        try:
            # Verify signature if enabled
            if self.config.signature_verification:
                if not self.signature_verifier.verify_signature(event, request_body):
                    self.metrics["signature_failures"] += 1
                    self.logger.warning(f"Signature verification failed for event {event.event_id}")
                    return False
            
            # Check for duplicates
            if self.config.deduplication_enabled:
                if await self.deduplicator.is_duplicate(event):
                    self.metrics["events_duplicate"] += 1
                    self.logger.debug(f"Duplicate event detected: {event.event_id}")
                    return True  # Return True since duplicate is not an error
            
            # Check queue capacity
            if len(self.event_queue) >= self.config.max_queue_size:
                self.logger.error("Event queue is full, rejecting event")
                return False
            
            # Queue event for processing
            queued_event = QueuedWebhookEvent(
                event=event,
                priority=priority,
                queued_at=datetime.utcnow()
            )
            
            # Handle ordering if enabled
            if self.config.ordering_enabled:
                can_process = await self.event_ordering.queue_event(event)
                if not can_process:
                    self.logger.debug(f"Event {event.event_id} queued for ordered processing")
            
            self.event_queue.append(queued_event)
            self.event_queue.sort()  # Maintain priority order
            
            self.logger.debug(f"Event {event.event_id} queued for processing")
            return True
        
        except Exception as e:
            self.logger.error(f"Error processing webhook event {event.event_id}: {e}")
            return False
    
    async def start_processing(self):
        """Start the event processing loop"""
        if self._processor_task is None or self._processor_task.done():
            self._shutdown_event.clear()
            self._processor_task = asyncio.create_task(self._process_events_loop())
            self.logger.info("Webhook event processor started")
    
    async def stop_processing(self):
        """Stop the event processing loop"""
        self._shutdown_event.set()
        if self._processor_task:
            await self._processor_task
        self.logger.info("Webhook event processor stopped")
    
    async def _process_events_loop(self):
        """Main event processing loop"""
        while not self._shutdown_event.is_set():
            try:
                # Process batch of events
                await self._process_event_batch()
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.01)
            
            except Exception as e:
                self.logger.error(f"Error in event processing loop: {e}")
                await asyncio.sleep(1.0)  # Longer delay on error
    
    async def _process_event_batch(self):
        """Process a batch of events"""
        batch_size = min(self.config.batch_size, len(self.event_queue))
        if batch_size == 0:
            return
        
        # Get batch of events
        batch = self.event_queue[:batch_size]
        self.event_queue = self.event_queue[batch_size:]
        
        # Process events concurrently
        tasks = [self._process_single_event(queued_event) for queued_event in batch]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _process_single_event(self, queued_event: QueuedWebhookEvent):
        """Process a single webhook event"""
        event = queued_event.event
        start_time = time.time()
        
        try:
            # Check if we should retry this event
            if queued_event.next_retry_at and datetime.utcnow() < queued_event.next_retry_at:
                # Re-queue for later
                self.event_queue.append(queued_event)
                return
            
            # Handle ordered processing
            if self.config.ordering_enabled:
                await self.event_ordering.mark_transaction_processing(event.transaction_id)
            
            try:
                # Get event handler
                handler = self.event_handlers.get(event.event_type)
                if not handler:
                    raise ValueError(f"No handler registered for event type: {event.event_type}")
                
                # Execute handler with timeout
                result = await asyncio.wait_for(
                    handler(event),
                    timeout=self.config.processing_timeout_seconds
                )
                
                # Success
                processing_time = int((time.time() - start_time) * 1000)
                self.metrics["processing_time_ms"].append(processing_time)
                self.metrics["events_processed"] += 1
                
                # Record successful processing
                if self.config.deduplication_enabled:
                    await self.deduplicator.record_event(event, EventStatus.COMPLETED)
                
                self.logger.debug(
                    f"Successfully processed event {event.event_id} "
                    f"in {processing_time}ms"
                )
            
            except Exception as e:
                # Handle processing failure
                await self._handle_processing_failure(queued_event, e)
            
            finally:
                # Mark transaction processing complete
                if self.config.ordering_enabled:
                    await self.event_ordering.mark_transaction_complete(event.transaction_id)
        
        except Exception as e:
            self.logger.error(f"Unexpected error processing event {event.event_id}: {e}")
            self.metrics["events_failed"] += 1
    
    async def _handle_processing_failure(self, queued_event: QueuedWebhookEvent, error: Exception):
        """Handle event processing failure with retry logic"""
        event = queued_event.event
        queued_event.retry_count += 1
        
        self.logger.warning(
            f"Event {event.event_id} processing failed "
            f"(attempt {queued_event.retry_count}): {error}"
        )
        
        # Check if we should retry
        if queued_event.retry_count < self.config.max_retries:
            # Calculate retry delay with exponential backoff
            delay_ms = min(
                self.config.retry_delay_base_ms * (2 ** (queued_event.retry_count - 1)),
                self.config.retry_delay_max_ms
            )
            
            queued_event.next_retry_at = datetime.utcnow() + timedelta(milliseconds=delay_ms)
            
            # Re-queue for retry
            self.event_queue.append(queued_event)
            self.event_queue.sort()
            
            self.logger.info(
                f"Event {event.event_id} will be retried in {delay_ms}ms "
                f"(attempt {queued_event.retry_count + 1})"
            )
        else:
            # Move to dead letter queue
            self.dead_letter_queue.append(queued_event)
            self.metrics["events_dead_letter"] += 1
            
            # Record as failed
            if self.config.deduplication_enabled:
                await self.deduplicator.record_event(event, EventStatus.DEAD_LETTER)
            
            self.logger.error(
                f"Event {event.event_id} moved to dead letter queue "
                f"after {queued_event.retry_count} attempts"
            )
        
        self.metrics["events_failed"] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive processing metrics"""
        avg_processing_time = 0
        if self.metrics["processing_time_ms"]:
            avg_processing_time = sum(self.metrics["processing_time_ms"]) / len(self.metrics["processing_time_ms"])
        
        return {
            "events_received": self.metrics["events_received"],
            "events_processed": self.metrics["events_processed"],
            "events_failed": self.metrics["events_failed"],
            "events_duplicate": self.metrics["events_duplicate"],
            "events_dead_letter": self.metrics["events_dead_letter"],
            "signature_failures": self.metrics["signature_failures"],
            "success_rate": self.metrics["events_processed"] / max(self.metrics["events_received"], 1),
            "queue_size": len(self.event_queue),
            "dead_letter_size": len(self.dead_letter_queue),
            "avg_processing_time_ms": avg_processing_time,
            "deduplicator_stats": {
                "processed_events_count": self.deduplicator.get_processed_count(),
            },
            "ordering_stats": self.event_ordering.get_queue_stats(),
        }
    
    def get_dead_letter_events(self) -> List[QueuedWebhookEvent]:
        """Get events in the dead letter queue"""
        return self.dead_letter_queue.copy()
    
    async def reprocess_dead_letter_event(self, event_id: str) -> bool:
        """Reprocess a specific event from dead letter queue"""
        for i, queued_event in enumerate(self.dead_letter_queue):
            if queued_event.event.event_id == event_id:
                # Reset retry count and move back to main queue
                queued_event.retry_count = 0
                queued_event.next_retry_at = None
                
                self.event_queue.append(queued_event)
                self.event_queue.sort()
                
                del self.dead_letter_queue[i]
                
                self.logger.info(f"Reprocessing dead letter event {event_id}")
                return True
        
        return False


# Example event handlers for different payment events
async def handle_payment_completed(event: WebhookEvent) -> Any:
    """Example handler for payment completion events"""
    logger.info(f"Payment completed: {event.transaction_id}")
    # Update payment status in database
    # Send confirmation email
    # Update inventory
    return {"status": "processed"}


async def handle_payment_failed(event: WebhookEvent) -> Any:
    """Example handler for payment failure events"""
    logger.info(f"Payment failed: {event.transaction_id}")
    # Update payment status
    # Notify customer
    # Trigger retry logic if applicable
    return {"status": "processed"}


async def handle_refund_processed(event: WebhookEvent) -> Any:
    """Example handler for refund events"""
    logger.info(f"Refund processed: {event.transaction_id}")
    # Update refund status
    # Send refund confirmation
    # Update accounting
    return {"status": "processed"}