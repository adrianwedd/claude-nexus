#!/usr/bin/env python3
"""
Intelligent Agent Selection and Routing Optimization System
=========================================================

Advanced ML-powered system for optimal agent selection, dynamic routing,
and multi-agent collaboration pattern optimization for the Claude-Nexus
agent ecosystem.

Key Features:
- Context-aware agent selection with ML scoring
- Dynamic load balancing and routing optimization
- Multi-agent collaboration pattern prediction
- Historical success pattern analysis
- Real-time performance-based routing decisions
- Adaptive learning from user feedback

Author: Intelligence Orchestrator (Claude-Nexus ML Team)
Date: 2025-08-04
Version: 1.0.0
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import hashlib
import threading
import time
from collections import defaultdict, deque
import math


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('intelligent_routing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TaskComplexity(Enum):
    """Task complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


class CollaborationType(Enum):
    """Multi-agent collaboration types"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"
    PEER_REVIEW = "peer_review"
    HANDOFF = "handoff"


class RoutingStrategy(Enum):
    """Routing strategy types"""
    PERFORMANCE_OPTIMIZED = "performance_optimized"
    LOAD_BALANCED = "load_balanced"
    COST_OPTIMIZED = "cost_optimized"
    QUALITY_FOCUSED = "quality_focused"
    HYBRID = "hybrid"


@dataclass
class TaskContext:
    """Enhanced task context for routing decisions"""
    task_id: str
    task_type: str
    complexity: TaskComplexity
    priority: int  # 1-10 scale
    domain_keywords: List[str]
    estimated_duration_minutes: int
    requires_collaboration: bool
    user_preferences: Dict[str, Any]
    deadline: Optional[datetime]
    security_level: int  # 1-5 scale
    timestamp: datetime


@dataclass
class AgentCapability:
    """Agent capability profile"""
    agent_type: str
    specialization_domains: List[str]
    current_performance_score: float
    historical_success_rate: float
    average_response_time_ms: int
    current_workload: int
    max_concurrent_tasks: int
    expertise_keywords: List[str]
    collaboration_affinity: Dict[str, float]  # Scores for working with other agents
    cost_per_minute: float
    last_updated: datetime


@dataclass
class RoutingDecision:
    """Routing decision with explanation"""
    task_id: str
    selected_agents: List[str]
    collaboration_pattern: CollaborationType
    confidence_score: float
    expected_performance: float
    estimated_cost: float
    estimated_duration_minutes: int
    routing_rationale: Dict[str, Any]
    alternative_options: List[Dict[str, Any]]
    timestamp: datetime


@dataclass
class CollaborationPattern:
    """Multi-agent collaboration pattern"""
    pattern_id: str
    agents: List[str]
    sequence: List[Dict[str, Any]]
    success_rate: float
    average_duration_minutes: int
    typical_handoff_points: List[str]
    quality_score: float
    cost_efficiency: float
    usage_count: int
    last_used: datetime


class IntelligentRoutingSystem:
    """Intelligent Agent Selection and Routing Optimization System"""
    
    def __init__(self, config_file: str = "routing_config.json"):
        self.config_file = config_file
        self.config = self._load_config()
        
        # Agent and task management
        self.agent_capabilities = {}
        self.task_queue = deque()
        self.active_tasks = {}
        self.completed_tasks = []
        
        # Routing intelligence
        self.success_patterns = {}
        self.collaboration_patterns = {}
        self.performance_history = defaultdict(list)
        self.routing_decisions = []
        
        # Real-time monitoring
        self.current_workloads = defaultdict(int)
        self.response_times = defaultdict(deque)
        self.quality_scores = defaultdict(deque)
        
        # Learning and optimization
        self.feedback_buffer = []
        self.adaptation_weights = {}
        
        # Threading for real-time processing
        self._stop_monitoring = threading.Event()
        self._monitoring_thread = None
        
        self._initialize_agents()
        self._load_historical_patterns()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load routing system configuration"""
        default_config = {
            "agents": {
                "reliability-engineer": {
                    "specialization_domains": ["reliability", "architecture", "operations"],
                    "max_concurrent_tasks": 3,
                    "cost_per_minute": 0.5,
                    "expertise_keywords": ["P0", "P1", "SLA", "architecture", "reliability", "monitoring"]
                },
                "fortress-guardian": {
                    "specialization_domains": ["security", "compliance", "risk"],
                    "max_concurrent_tasks": 2,
                    "cost_per_minute": 0.7,
                    "expertise_keywords": ["CVSS", "vulnerability", "security", "compliance", "threat"]
                },
                "performance-virtuoso": {
                    "specialization_domains": ["performance", "optimization", "scalability"],
                    "max_concurrent_tasks": 4,
                    "cost_per_minute": 0.6,
                    "expertise_keywords": ["latency", "throughput", "optimization", "performance", "scalability"]
                },
                "interface-artisan": {
                    "specialization_domains": ["ui", "ux", "design", "frontend"],
                    "max_concurrent_tasks": 3,
                    "cost_per_minute": 0.45,
                    "expertise_keywords": ["UI", "UX", "design", "interface", "accessibility"]
                },
                "data-architect": {
                    "specialization_domains": ["data", "analytics", "ml", "ai"],
                    "max_concurrent_tasks": 2,
                    "cost_per_minute": 0.8,
                    "expertise_keywords": ["data", "analytics", "ML", "AI", "pipeline"]
                }
            },
            "routing_strategies": {
                "default_strategy": RoutingStrategy.HYBRID.value,
                "performance_weight": 0.4,
                "cost_weight": 0.2,
                "quality_weight": 0.3,
                "load_balance_weight": 0.1
            },
            "collaboration_settings": {
                "enable_multi_agent": True,
                "max_agents_per_task": 3,
                "collaboration_threshold": 0.7,  # When to suggest collaboration
                "handoff_delay_seconds": 5
            },
            "learning_parameters": {
                "adaptation_rate": 0.1,
                "history_window_days": 30,
                "min_samples_for_learning": 10,
                "feedback_weight": 0.3
            },
            "performance_targets": {
                "response_time_ms": 5000,
                "quality_score": 0.85,
                "success_rate": 0.9,
                "cost_efficiency": 0.8
            }
        }
        
        try:
            with open(self.config_file, 'r') as f:
                loaded_config = json.load(f)
                return {**default_config, **loaded_config}
        except FileNotFoundError:
            logger.info(f"Config file not found, using defaults. Creating {self.config_file}")
            self._save_config(default_config)
            return default_config
    
    def _save_config(self, config: Dict[str, Any]):
        """Save configuration to file"""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
    
    def _initialize_agents(self):
        """Initialize agent capability profiles"""
        for agent_type, config in self.config["agents"].items():
            self.agent_capabilities[agent_type] = AgentCapability(
                agent_type=agent_type,
                specialization_domains=config["specialization_domains"],
                current_performance_score=0.8,  # Initial score
                historical_success_rate=0.85,   # Initial rate
                average_response_time_ms=5000,  # Initial estimate
                current_workload=0,
                max_concurrent_tasks=config["max_concurrent_tasks"],
                expertise_keywords=config["expertise_keywords"],
                collaboration_affinity={},  # Will be learned
                cost_per_minute=config["cost_per_minute"],
                last_updated=datetime.now()
            )
            
            # Initialize response time and quality tracking
            self.response_times[agent_type] = deque(maxlen=100)
            self.quality_scores[agent_type] = deque(maxlen=100)
    
    def _load_historical_patterns(self):
        """Load historical collaboration patterns and success data"""
        try:
            with open("routing_patterns.json", 'r') as f:
                patterns_data = json.load(f)
                
                for pattern_data in patterns_data.get("collaboration_patterns", []):
                    pattern = CollaborationPattern(
                        pattern_id=pattern_data["pattern_id"],
                        agents=pattern_data["agents"],
                        sequence=pattern_data["sequence"],
                        success_rate=pattern_data["success_rate"],
                        average_duration_minutes=pattern_data["average_duration_minutes"],
                        typical_handoff_points=pattern_data["typical_handoff_points"],
                        quality_score=pattern_data["quality_score"],
                        cost_efficiency=pattern_data["cost_efficiency"],
                        usage_count=pattern_data["usage_count"],
                        last_used=datetime.fromisoformat(pattern_data["last_used"])
                    )
                    self.collaboration_patterns[pattern.pattern_id] = pattern
                    
                logger.info(f"Loaded {len(self.collaboration_patterns)} collaboration patterns")
                
        except FileNotFoundError:
            logger.info("No historical patterns found, starting fresh")
    
    def select_optimal_agent(self, task_context: TaskContext, 
                            available_agents: Optional[List[str]] = None) -> RoutingDecision:
        """Select optimal agent(s) for given task context"""
        try:
            if available_agents is None:
                available_agents = list(self.agent_capabilities.keys())
            
            # Filter available agents based on current workload
            eligible_agents = [
                agent for agent in available_agents
                if self.current_workloads[agent] < self.agent_capabilities[agent].max_concurrent_tasks
            ]
            
            if not eligible_agents:
                # All agents at capacity - select least loaded
                eligible_agents = [min(available_agents, key=lambda x: self.current_workloads[x])]
            
            # Calculate agent scores
            agent_scores = {}
            explanations = {}
            
            for agent in eligible_agents:
                score, explanation = self._calculate_agent_score(agent, task_context)
                agent_scores[agent] = score
                explanations[agent] = explanation
            
            # Determine if multi-agent collaboration is needed
            requires_collaboration = self._should_use_collaboration(task_context, agent_scores)
            
            if requires_collaboration:
                return self._select_collaborative_agents(task_context, agent_scores, explanations)
            else:
                return self._select_single_agent(task_context, agent_scores, explanations)
                
        except Exception as e:
            logger.error(f"Agent selection error: {e}")
            return self._fallback_selection(task_context, available_agents)
    
    def _calculate_agent_score(self, agent_type: str, task_context: TaskContext) -> Tuple[float, Dict[str, Any]]:
        """Calculate comprehensive agent suitability score"""
        capability = self.agent_capabilities[agent_type]
        
        # Domain expertise match
        domain_score = self._calculate_domain_match(capability, task_context)
        
        # Keyword relevance
        keyword_score = self._calculate_keyword_relevance(capability, task_context)
        
        # Performance factors
        performance_score = capability.current_performance_score
        
        # Workload and availability
        workload_penalty = self.current_workloads[agent_type] / capability.max_concurrent_tasks
        availability_score = 1.0 - workload_penalty
        
        # Response time factor
        response_time_score = self._calculate_response_time_score(agent_type)
        
        # Cost efficiency
        cost_score = self._calculate_cost_efficiency(capability, task_context)
        
        # Priority and urgency considerations
        urgency_bonus = min(0.2, task_context.priority / 50)  # Up to 20% bonus for high priority
        
        # Weighted combination
        weights = self.config["routing_strategies"]
        final_score = (
            domain_score * 0.25 +
            keyword_score * 0.2 +
            performance_score * weights["performance_weight"] +
            availability_score * weights["load_balance_weight"] +
            response_time_score * 0.1 +
            cost_score * weights["cost_weight"] +
            urgency_bonus
        )
        
        explanation = {
            "domain_match": domain_score,
            "keyword_relevance": keyword_score,
            "performance": performance_score,
            "availability": availability_score,
            "response_time": response_time_score,
            "cost_efficiency": cost_score,
            "urgency_bonus": urgency_bonus,
            "final_score": final_score,
            "current_workload": self.current_workloads[agent_type],
            "max_capacity": capability.max_concurrent_tasks
        }
        
        return final_score, explanation
    
    def _calculate_domain_match(self, capability: AgentCapability, task_context: TaskContext) -> float:
        """Calculate domain expertise match score"""
        task_domains = set(task_context.domain_keywords)
        agent_domains = set(capability.specialization_domains)
        
        if not task_domains:
            return 0.5  # Neutral score if no domain specified
        
        intersection = task_domains.intersection(agent_domains)
        union = task_domains.union(agent_domains)
        
        # Jaccard similarity with bonus for exact matches
        jaccard = len(intersection) / len(union) if union else 0
        exact_match_bonus = len(intersection) / len(task_domains) * 0.3
        
        return min(1.0, jaccard + exact_match_bonus)
    
    def _calculate_keyword_relevance(self, capability: AgentCapability, task_context: TaskContext) -> float:
        """Calculate keyword relevance score"""
        task_keywords = set(kw.lower() for kw in task_context.domain_keywords)
        agent_keywords = set(kw.lower() for kw in capability.expertise_keywords)
        
        if not task_keywords:
            return 0.5
        
        matches = task_keywords.intersection(agent_keywords)
        return len(matches) / len(task_keywords) if task_keywords else 0
    
    def _calculate_response_time_score(self, agent_type: str) -> float:
        """Calculate response time performance score"""
        recent_times = self.response_times[agent_type]
        if not recent_times:
            return 0.8  # Default score
        
        avg_time = sum(recent_times) / len(recent_times)
        target_time = self.config["performance_targets"]["response_time_ms"]
        
        # Score inversely proportional to response time
        if avg_time <= target_time:
            return 1.0
        else:
            return max(0.1, target_time / avg_time)
    
    def _calculate_cost_efficiency(self, capability: AgentCapability, task_context: TaskContext) -> float:
        """Calculate cost efficiency score"""
        task_budget = task_context.user_preferences.get('budget_per_minute', 1.0)
        
        if capability.cost_per_minute <= task_budget:
            return 1.0 - (capability.cost_per_minute / task_budget) * 0.5
        else:
            return max(0.1, task_budget / capability.cost_per_minute)
    
    def _should_use_collaboration(self, task_context: TaskContext, agent_scores: Dict[str, float]) -> bool:
        """Determine if multi-agent collaboration should be used"""
        if not self.config["collaboration_settings"]["enable_multi_agent"]:
            return False
        
        if task_context.requires_collaboration:
            return True
        
        # Check if task complexity suggests collaboration
        if task_context.complexity in [TaskComplexity.COMPLEX, TaskComplexity.EXPERT]:
            return True
        
        # Check if no single agent has high confidence
        max_score = max(agent_scores.values()) if agent_scores else 0
        collaboration_threshold = self.config["collaboration_settings"]["collaboration_threshold"]
        
        return max_score < collaboration_threshold
    
    def _select_collaborative_agents(self, task_context: TaskContext, 
                                   agent_scores: Dict[str, float],
                                   explanations: Dict[str, Dict[str, Any]]) -> RoutingDecision:
        """Select multiple agents for collaborative task execution"""
        max_agents = self.config["collaboration_settings"]["max_agents_per_task"]
        
        # Find best collaboration pattern
        best_pattern = self._find_best_collaboration_pattern(task_context, agent_scores)
        
        if best_pattern:
            selected_agents = best_pattern.agents
            collaboration_type = CollaborationType.SEQUENTIAL  # Pattern-based
        else:
            # Select top scoring agents
            sorted_agents = sorted(agent_scores.items(), key=lambda x: x[1], reverse=True)
            selected_agents = [agent for agent, _ in sorted_agents[:max_agents]]
            collaboration_type = self._determine_collaboration_type(task_context, selected_agents)
        
        # Calculate collaborative performance prediction
        individual_scores = [agent_scores[agent] for agent in selected_agents]
        collaboration_bonus = 0.1 * len(selected_agents)  # Synergy bonus
        expected_performance = min(1.0, np.mean(individual_scores) + collaboration_bonus)
        
        # Calculate costs and duration
        total_cost = sum(self.agent_capabilities[agent].cost_per_minute for agent in selected_agents)
        estimated_duration = task_context.estimated_duration_minutes
        if collaboration_type == CollaborationType.PARALLEL:
            estimated_duration = int(estimated_duration * 0.7)  # Parallel efficiency
        elif collaboration_type == CollaborationType.SEQUENTIAL:
            estimated_duration = int(estimated_duration * 1.2)  # Sequential overhead
        
        estimated_cost = total_cost * estimated_duration
        
        routing_rationale = {
            "collaboration_reason": "High complexity task requiring multi-agent expertise",
            "selected_pattern": best_pattern.pattern_id if best_pattern else "dynamic_selection",
            "agent_explanations": {agent: explanations[agent] for agent in selected_agents},
            "collaboration_type": collaboration_type.value,
            "synergy_factors": self._calculate_synergy_factors(selected_agents)
        }
        
        # Generate alternatives
        alternatives = self._generate_alternative_options(task_context, agent_scores, selected_agents)
        
        return RoutingDecision(
            task_id=task_context.task_id,
            selected_agents=selected_agents,
            collaboration_pattern=collaboration_type,
            confidence_score=expected_performance,
            expected_performance=expected_performance,
            estimated_cost=estimated_cost,
            estimated_duration_minutes=estimated_duration,
            routing_rationale=routing_rationale,
            alternative_options=alternatives,
            timestamp=datetime.now()
        )
    
    def _select_single_agent(self, task_context: TaskContext,
                           agent_scores: Dict[str, float],
                           explanations: Dict[str, Dict[str, Any]]) -> RoutingDecision:
        """Select single best agent for task execution"""
        best_agent = max(agent_scores.keys(), key=lambda x: agent_scores[x])
        best_score = agent_scores[best_agent]
        
        capability = self.agent_capabilities[best_agent]
        estimated_cost = capability.cost_per_minute * task_context.estimated_duration_minutes
        
        routing_rationale = {
            "selection_reason": "Single agent sufficient for task complexity",
            "agent_explanation": explanations[best_agent],
            "confidence_factors": {
                "domain_expertise": explanations[best_agent]["domain_match"],
                "availability": explanations[best_agent]["availability"],
                "performance_history": capability.historical_success_rate
            }
        }
        
        alternatives = self._generate_alternative_options(task_context, agent_scores, [best_agent])
        
        return RoutingDecision(
            task_id=task_context.task_id,
            selected_agents=[best_agent],
            collaboration_pattern=CollaborationType.SEQUENTIAL,  # Single agent
            confidence_score=best_score,
            expected_performance=best_score,
            estimated_cost=estimated_cost,
            estimated_duration_minutes=task_context.estimated_duration_minutes,
            routing_rationale=routing_rationale,
            alternative_options=alternatives,
            timestamp=datetime.now()
        )
    
    def _find_best_collaboration_pattern(self, task_context: TaskContext,
                                       agent_scores: Dict[str, float]) -> Optional[CollaborationPattern]:
        """Find best collaboration pattern for task"""
        if not self.collaboration_patterns:
            return None
        
        best_pattern = None
        best_score = 0
        
        for pattern in self.collaboration_patterns.values():
            # Check if pattern agents are available and suitable
            pattern_score = 0
            available_agents = 0
            
            for agent in pattern.agents:
                if agent in agent_scores:
                    pattern_score += agent_scores[agent]
                    available_agents += 1
            
            if available_agents == len(pattern.agents):
                # All agents available, calculate pattern score
                avg_agent_score = pattern_score / len(pattern.agents)
                pattern_score = (
                    avg_agent_score * 0.4 +
                    pattern.success_rate * 0.3 +
                    pattern.quality_score * 0.2 +
                    pattern.cost_efficiency * 0.1
                )
                
                if pattern_score > best_score:
                    best_score = pattern_score
                    best_pattern = pattern
        
        return best_pattern
    
    def _determine_collaboration_type(self, task_context: TaskContext, 
                                    selected_agents: List[str]) -> CollaborationType:
        """Determine optimal collaboration type for selected agents"""
        if len(selected_agents) == 1:
            return CollaborationType.SEQUENTIAL
        
        # Analyze task context to determine collaboration type
        if task_context.complexity == TaskComplexity.EXPERT:
            return CollaborationType.HIERARCHICAL
        elif task_context.priority >= 8:
            return CollaborationType.PARALLEL
        elif "review" in task_context.task_type.lower():
            return CollaborationType.PEER_REVIEW
        else:
            return CollaborationType.SEQUENTIAL
    
    def _calculate_synergy_factors(self, selected_agents: List[str]) -> Dict[str, float]:
        """Calculate synergy factors between selected agents"""
        synergy_factors = {}
        
        for i, agent1 in enumerate(selected_agents):
            for agent2 in selected_agents[i+1:]:
                # Check collaboration affinity
                affinity1 = self.agent_capabilities[agent1].collaboration_affinity.get(agent2, 0.5)
                affinity2 = self.agent_capabilities[agent2].collaboration_affinity.get(agent1, 0.5)
                
                synergy_score = (affinity1 + affinity2) / 2
                synergy_factors[f"{agent1}-{agent2}"] = synergy_score
        
        return synergy_factors
    
    def _generate_alternative_options(self, task_context: TaskContext,
                                    agent_scores: Dict[str, float],
                                    selected_agents: List[str]) -> List[Dict[str, Any]]:
        """Generate alternative routing options"""
        alternatives = []
        
        # Get top 3 alternatives
        sorted_agents = sorted(
            [(agent, score) for agent, score in agent_scores.items() if agent not in selected_agents],
            key=lambda x: x[1], reverse=True
        )
        
        for agent, score in sorted_agents[:3]:
            capability = self.agent_capabilities[agent]
            cost = capability.cost_per_minute * task_context.estimated_duration_minutes
            
            alternatives.append({
                "agent": agent,
                "confidence": score,
                "estimated_cost": cost,
                "estimated_duration": task_context.estimated_duration_minutes,
                "reason": f"Alternative option with {score:.1%} confidence"
            })
        
        return alternatives
    
    def update_agent_performance(self, agent_type: str, performance_data: Dict[str, Any]):
        """Update agent performance metrics"""
        if agent_type not in self.agent_capabilities:
            return
        
        capability = self.agent_capabilities[agent_type]
        
        # Update performance metrics
        if "response_time_ms" in performance_data:
            self.response_times[agent_type].append(performance_data["response_time_ms"])
            capability.average_response_time_ms = int(np.mean(self.response_times[agent_type]))
        
        if "quality_score" in performance_data:
            self.quality_scores[agent_type].append(performance_data["quality_score"])
        
        if "specialization_score" in performance_data:
            capability.current_performance_score = performance_data["specialization_score"]
        
        # Update success rate with exponential moving average
        if "task_successful" in performance_data:
            alpha = self.config["learning_parameters"]["adaptation_rate"]
            old_rate = capability.historical_success_rate
            new_success = 1.0 if performance_data["task_successful"] else 0.0
            capability.historical_success_rate = (1 - alpha) * old_rate + alpha * new_success
        
        capability.last_updated = datetime.now()
        
        # Store in performance history
        self.performance_history[agent_type].append({
            "timestamp": datetime.now().isoformat(),
            "performance_data": performance_data
        })
        
        # Trim history
        if len(self.performance_history[agent_type]) > 1000:
            self.performance_history[agent_type] = self.performance_history[agent_type][-1000:]
    
    def record_task_completion(self, task_id: str, outcome: Dict[str, Any]):
        """Record task completion and learn from outcome"""
        try:
            # Find the routing decision
            routing_decision = None
            for decision in self.routing_decisions:
                if decision.task_id == task_id:
                    routing_decision = decision
                    break
            
            if not routing_decision:
                logger.warning(f"No routing decision found for task {task_id}")
                return
            
            # Update agent workloads
            for agent in routing_decision.selected_agents:
                self.current_workloads[agent] = max(0, self.current_workloads[agent] - 1)
            
            # Learn from outcome
            self._learn_from_outcome(routing_decision, outcome)
            
            # Update collaboration patterns if multi-agent
            if len(routing_decision.selected_agents) > 1:
                self._update_collaboration_pattern(routing_decision, outcome)
            
            # Store completed task
            completion_record = {
                "task_id": task_id,
                "routing_decision": asdict(routing_decision),
                "outcome": outcome,
                "completion_timestamp": datetime.now().isoformat()
            }
            self.completed_tasks.append(completion_record)
            
            # Trim completed tasks history
            if len(self.completed_tasks) > 10000:
                self.completed_tasks = self.completed_tasks[-10000:]
            
            logger.info(f"Task {task_id} completion recorded successfully")
            
        except Exception as e:
            logger.error(f"Error recording task completion: {e}")
    
    def _learn_from_outcome(self, routing_decision: RoutingDecision, outcome: Dict[str, Any]):
        """Learn from task outcome to improve future routing decisions"""
        actual_performance = outcome.get("quality_score", 0.8)
        actual_duration = outcome.get("duration_minutes", routing_decision.estimated_duration_minutes)
        actual_cost = outcome.get("cost", routing_decision.estimated_cost)
        
        # Calculate prediction accuracy
        performance_error = abs(actual_performance - routing_decision.expected_performance)
        duration_error = abs(actual_duration - routing_decision.estimated_duration_minutes) / routing_decision.estimated_duration_minutes
        cost_error = abs(actual_cost - routing_decision.estimated_cost) / routing_decision.estimated_cost if routing_decision.estimated_cost > 0 else 0
        
        # Update adaptation weights based on prediction accuracy
        adaptation_rate = self.config["learning_parameters"]["adaptation_rate"]
        
        for agent in routing_decision.selected_agents:
            if agent not in self.adaptation_weights:
                self.adaptation_weights[agent] = {
                    "performance_weight": 1.0,
                    "duration_weight": 1.0,
                    "cost_weight": 1.0
                }
            
            # Adjust weights based on prediction errors
            if performance_error < 0.1:  # Good prediction
                self.adaptation_weights[agent]["performance_weight"] *= (1 + adaptation_rate)
            else:  # Poor prediction
                self.adaptation_weights[agent]["performance_weight"] *= (1 - adaptation_rate * 0.5)
            
            # Keep weights in reasonable bounds
            for weight_key in self.adaptation_weights[agent]:
                self.adaptation_weights[agent][weight_key] = max(0.1, min(2.0, self.adaptation_weights[agent][weight_key]))
    
    def _update_collaboration_pattern(self, routing_decision: RoutingDecision, outcome: Dict[str, Any]):
        """Update collaboration patterns based on outcome"""
        if len(routing_decision.selected_agents) < 2:
            return
        
        pattern_id = self._generate_pattern_id(routing_decision.selected_agents, routing_decision.collaboration_pattern)
        
        quality_score = outcome.get("quality_score", 0.8)
        duration_minutes = outcome.get("duration_minutes", routing_decision.estimated_duration_minutes)
        cost = outcome.get("cost", routing_decision.estimated_cost)
        success = outcome.get("successful", True)
        
        if pattern_id in self.collaboration_patterns:
            # Update existing pattern
            pattern = self.collaboration_patterns[pattern_id]
            
            # Exponential moving average for metrics
            alpha = 0.1
            pattern.success_rate = (1 - alpha) * pattern.success_rate + alpha * (1.0 if success else 0.0)
            pattern.quality_score = (1 - alpha) * pattern.quality_score + alpha * quality_score
            pattern.average_duration_minutes = int((1 - alpha) * pattern.average_duration_minutes + alpha * duration_minutes)
            
            pattern.usage_count += 1
            pattern.last_used = datetime.now()
            
        else:
            # Create new pattern
            self.collaboration_patterns[pattern_id] = CollaborationPattern(
                pattern_id=pattern_id,
                agents=routing_decision.selected_agents,
                sequence=[{"agent": agent, "step": i} for i, agent in enumerate(routing_decision.selected_agents)],
                success_rate=1.0 if success else 0.0,
                average_duration_minutes=duration_minutes,
                typical_handoff_points=[],
                quality_score=quality_score,
                cost_efficiency=0.8,  # Initial estimate
                usage_count=1,
                last_used=datetime.now()
            )
    
    def _generate_pattern_id(self, agents: List[str], collaboration_type: CollaborationType) -> str:
        """Generate unique pattern ID for collaboration"""
        agents_str = "-".join(sorted(agents))
        pattern_str = f"{agents_str}_{collaboration_type.value}"
        return hashlib.md5(pattern_str.encode()).hexdigest()[:8]
    
    def start_task(self, task_context: TaskContext) -> RoutingDecision:
        """Start a new task with optimal routing"""
        try:
            # Make routing decision
            routing_decision = self.select_optimal_agent(task_context)
            
            # Update agent workloads
            for agent in routing_decision.selected_agents:
                self.current_workloads[agent] += 1
            
            # Store routing decision
            self.routing_decisions.append(routing_decision)
            
            # Add to active tasks
            self.active_tasks[task_context.task_id] = {
                "task_context": task_context,
                "routing_decision": routing_decision,
                "start_time": datetime.now()
            }
            
            logger.info(f"Task {task_context.task_id} started with agents: {routing_decision.selected_agents}")
            
            return routing_decision
            
        except Exception as e:
            logger.error(f"Error starting task {task_context.task_id}: {e}")
            return self._fallback_selection(task_context, list(self.agent_capabilities.keys()))
    
    def _fallback_selection(self, task_context: TaskContext, available_agents: List[str]) -> RoutingDecision:
        """Fallback routing decision when main logic fails"""
        # Simple round-robin selection
        if not available_agents:
            available_agents = list(self.agent_capabilities.keys())
        
        selected_agent = available_agents[0]  # Simplest fallback
        capability = self.agent_capabilities[selected_agent]
        
        return RoutingDecision(
            task_id=task_context.task_id,
            selected_agents=[selected_agent],
            collaboration_pattern=CollaborationType.SEQUENTIAL,
            confidence_score=0.5,
            expected_performance=0.7,
            estimated_cost=capability.cost_per_minute * task_context.estimated_duration_minutes,
            estimated_duration_minutes=task_context.estimated_duration_minutes,
            routing_rationale={"method": "fallback_selection", "reason": "Error in main routing logic"},
            alternative_options=[],
            timestamp=datetime.now()
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "agents": {},
            "current_workloads": dict(self.current_workloads),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "collaboration_patterns": len(self.collaboration_patterns),
            "routing_decisions": len(self.routing_decisions),
            "performance_summary": {}
        }
        
        # Agent status
        for agent_type, capability in self.agent_capabilities.items():
            recent_response_times = list(self.response_times[agent_type])
            recent_quality_scores = list(self.quality_scores[agent_type])
            
            status["agents"][agent_type] = {
                "current_performance_score": capability.current_performance_score,
                "historical_success_rate": capability.historical_success_rate,
                "current_workload": self.current_workloads[agent_type],
                "max_capacity": capability.max_concurrent_tasks,
                "utilization": self.current_workloads[agent_type] / capability.max_concurrent_tasks,
                "avg_response_time_ms": int(np.mean(recent_response_times)) if recent_response_times else capability.average_response_time_ms,
                "avg_quality_score": float(np.mean(recent_quality_scores)) if recent_quality_scores else 0.8,
                "specialization_domains": capability.specialization_domains,
                "cost_per_minute": capability.cost_per_minute
            }
        
        # Performance summary
        if self.routing_decisions:
            recent_decisions = self.routing_decisions[-100:]  # Last 100 decisions
            
            confidence_scores = [d.confidence_score for d in recent_decisions]
            expected_performances = [d.expected_performance for d in recent_decisions]
            
            status["performance_summary"] = {
                "avg_confidence": float(np.mean(confidence_scores)),
                "avg_expected_performance": float(np.mean(expected_performances)),
                "single_agent_tasks": len([d for d in recent_decisions if len(d.selected_agents) == 1]),
                "multi_agent_tasks": len([d for d in recent_decisions if len(d.selected_agents) > 1]),
                "collaboration_rate": len([d for d in recent_decisions if len(d.selected_agents) > 1]) / len(recent_decisions)
            }
        
        return status
    
    def save_state(self, filename: str = "routing_system_state.json"):
        """Save routing system state"""
        state = {
            "config": self.config,
            "agent_capabilities": {
                agent: {
                    **asdict(capability),
                    "last_updated": capability.last_updated.isoformat()
                }
                for agent, capability in self.agent_capabilities.items()
            },
            "collaboration_patterns": {
                pattern_id: {
                    **asdict(pattern),
                    "last_used": pattern.last_used.isoformat()
                }
                for pattern_id, pattern in self.collaboration_patterns.items()
            },
            "performance_history": {
                agent: history[-100:]  # Keep last 100 entries
                for agent, history in self.performance_history.items()
            },
            "routing_decisions": [
                {
                    **asdict(decision),
                    "timestamp": decision.timestamp.isoformat()
                }
                for decision in self.routing_decisions[-1000:]  # Keep last 1000
            ],
            "completed_tasks": self.completed_tasks[-1000:],  # Keep last 1000
            "current_workloads": dict(self.current_workloads),
            "adaptation_weights": self.adaptation_weights,
            "last_updated": datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"Routing system state saved to {filename}")


def main():
    """Main function for testing Intelligent Routing System"""
    print("🎯 Intelligent Agent Selection and Routing Optimization System")
    print("=" * 60)
    
    # Initialize routing system
    routing_system = IntelligentRoutingSystem()
    
    # Test single agent selection
    print("\n🔍 Testing Single Agent Selection...")
    task_context = TaskContext(
        task_id="test_001",
        task_type="performance optimization",
        complexity=TaskComplexity.MODERATE,
        priority=7,
        domain_keywords=["performance", "latency", "optimization"],
        estimated_duration_minutes=30,
        requires_collaboration=False,
        user_preferences={"budget_per_minute": 0.8},
        deadline=datetime.now() + timedelta(hours=2),
        security_level=2,
        timestamp=datetime.now()
    )
    
    decision = routing_system.select_optimal_agent(task_context)
    print(f"   Selected Agent: {decision.selected_agents[0]}")
    print(f"   Confidence: {decision.confidence_score:.1%}")
    print(f"   Expected Performance: {decision.expected_performance:.1%}")
    print(f"   Estimated Cost: ${decision.estimated_cost:.2f}")
    print(f"   Estimated Duration: {decision.estimated_duration_minutes} minutes")
    
    # Test multi-agent collaboration
    print("\n🤝 Testing Multi-Agent Collaboration...")
    complex_task = TaskContext(
        task_id="test_002",
        task_type="security performance audit",
        complexity=TaskComplexity.EXPERT,
        priority=9,
        domain_keywords=["security", "performance", "audit", "compliance"],
        estimated_duration_minutes=60,
        requires_collaboration=True,
        user_preferences={"budget_per_minute": 1.2},
        deadline=datetime.now() + timedelta(hours=4),
        security_level=5,
        timestamp=datetime.now()
    )
    
    collaboration_decision = routing_system.select_optimal_agent(complex_task)
    print(f"   Selected Agents: {', '.join(collaboration_decision.selected_agents)}")
    print(f"   Collaboration Type: {collaboration_decision.collaboration_pattern.value}")
    print(f"   Confidence: {collaboration_decision.confidence_score:.1%}")
    print(f"   Expected Performance: {collaboration_decision.expected_performance:.1%}")
    print(f"   Estimated Cost: ${collaboration_decision.estimated_cost:.2f}")
    
    # Start tasks and simulate execution
    print("\n🚀 Starting Tasks...")
    routing_system.start_task(task_context)
    routing_system.start_task(complex_task)
    
    # Update agent performance
    print("\n📊 Updating Agent Performance...")
    performance_update = {
        "response_time_ms": 4500,
        "quality_score": 0.92,
        "specialization_score": 0.88,
        "task_successful": True
    }
    routing_system.update_agent_performance("performance-virtuoso", performance_update)
    
    # Simulate task completion
    print("\n✅ Recording Task Completion...")
    outcome = {
        "quality_score": 0.89,
        "duration_minutes": 28,
        "cost": 14.0,
        "successful": True,
        "user_satisfaction": 0.95
    }
    routing_system.record_task_completion("test_001", outcome)
    
    # Get system status
    print("\n📈 System Status...")
    status = routing_system.get_system_status()
    
    print(f"   Active Tasks: {status['active_tasks']}")
    print(f"   Completed Tasks: {status['completed_tasks']}")
    print(f"   Collaboration Patterns: {status['collaboration_patterns']}")
    
    print("\n   Agent Utilization:")
    for agent, info in status["agents"].items():
        utilization = info["utilization"] * 100
        print(f"   • {agent}: {utilization:.0f}% ({info['current_workload']}/{info['max_capacity']})")
    
    if status["performance_summary"]:
        perf = status["performance_summary"]
        print(f"\n   Performance Summary:")
        print(f"   • Average Confidence: {perf['avg_confidence']:.1%}")
        print(f"   • Average Expected Performance: {perf['avg_expected_performance']:.1%}")
        print(f"   • Collaboration Rate: {perf['collaboration_rate']:.1%}")
    
    # Save state
    routing_system.save_state()
    print(f"\n💾 Routing system state saved successfully")
    
    print(f"\n✅ Intelligent Routing System initialized and tested successfully!")
    print(f"   • Context-aware agent selection operational")
    print(f"   • Dynamic load balancing enabled")
    print(f"   • Multi-agent collaboration patterns active")
    print(f"   • Performance-based routing decisions implemented")
    print(f"   • Adaptive learning from feedback operational")
    
    return routing_system


if __name__ == "__main__":
    system = main()