#!/usr/bin/env python3
"""
Agent Performance Metrics Framework for Claude-Nexus Ecosystem
================================================================

Comprehensive measurement and analytics system for evaluating agent effectiveness,
specialization value, and multi-agent workflow performance.

Author: Claude-Nexus Performance Team
Date: 2025-08-03
Version: 1.0.0
"""

import json
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import statistics
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Enumeration of all specialized agents in the claude-nexus ecosystem"""
    RELIABILITY_ENGINEER = "reliability-engineer"
    CODE_SOVEREIGN = "code-sovereign"
    PERFORMANCE_VIRTUOSO = "performance-virtuoso"
    CLOUD_NAVIGATOR = "cloud-navigator"
    FORTRESS_GUARDIAN = "fortress-guardian"
    INTEGRATION_MAESTRO = "integration-maestro"
    DATA_ARCHITECT = "data-architect"
    DATA_FLOW_ARCHITECT = "data-flow-architect"
    DEPLOYMENT_COMMANDER = "deployment-commander"
    INTERFACE_ARTISAN = "interface-artisan"
    MOBILE_PLATFORM_SPECIALIST = "mobile-platform-specialist"
    QUALITY_ASSURANCE_ENGINEER = "quality-assurance-engineer"
    DEVEX_CURATOR = "devex-curator"
    INTELLIGENCE_ORCHESTRATOR = "intelligence-orchestrator"
    KNOWLEDGE_CURATOR = "knowledge-curator"
    REPOSITORY_SURGEON = "repository-surgeon"


class MetricType(Enum):
    """Types of metrics collected for agent performance analysis"""
    RESPONSE_QUALITY = "response_quality"
    SPECIALIZATION_VALUE = "specialization_value"
    TASK_COMPLETION = "task_completion"
    COLLABORATION_EFFECTIVENESS = "collaboration_effectiveness"
    USER_SATISFACTION = "user_satisfaction"
    TECHNICAL_ACCURACY = "technical_accuracy"


@dataclass
class AgentPerformanceMetric:
    """Individual performance metric for an agent execution"""
    agent_type: AgentType
    metric_type: MetricType
    value: float  # 0.0 - 1.0 scale
    timestamp: datetime
    task_id: str
    execution_time_ms: int
    context_size: int
    output_length: int
    specialization_keywords: List[str]
    user_feedback: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "agent_type": self.agent_type.value,
            "metric_type": self.metric_type.value,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "task_id": self.task_id,
            "execution_time_ms": self.execution_time_ms,
            "context_size": self.context_size,
            "output_length": self.output_length,
            "specialization_keywords": self.specialization_keywords,
            "user_feedback": self.user_feedback
        }


@dataclass
class MultiAgentWorkflowMetric:
    """Performance metrics for multi-agent collaboration workflows"""
    workflow_id: str
    agents_involved: List[AgentType]
    total_execution_time_ms: int
    handoff_efficiency: float  # 0.0 - 1.0
    output_quality: float  # 0.0 - 1.0
    collaboration_score: float  # 0.0 - 1.0
    business_value_delivered: float  # 0.0 - 1.0
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "workflow_id": self.workflow_id,
            "agents_involved": [agent.value for agent in self.agents_involved],
            "total_execution_time_ms": self.total_execution_time_ms,
            "handoff_efficiency": self.handoff_efficiency,
            "output_quality": self.output_quality,
            "collaboration_score": self.collaboration_score,
            "business_value_delivered": self.business_value_delivered,
            "timestamp": self.timestamp.isoformat()
        }


class AgentPerformanceAnalyzer:
    """Comprehensive analytics engine for agent performance measurement"""
    
    def __init__(self, data_store_path: str = "agent_metrics.json"):
        self.data_store_path = data_store_path
        self.metrics: List[AgentPerformanceMetric] = []
        self.workflow_metrics: List[MultiAgentWorkflowMetric] = []
        self.specialization_baselines = self._initialize_specialization_baselines()
        
    def _initialize_specialization_baselines(self) -> Dict[AgentType, Dict[str, Any]]:
        """Initialize expected specialization characteristics for each agent"""
        return {
            AgentType.RELIABILITY_ENGINEER: {
                "keywords": ["architecture", "reliability", "P0", "P1", "monitoring", "SLA", "operational"],
                "expected_response_time_ms": 15000,  # Complex analysis
                "expected_output_length": 2000,
                "specialization_indicators": ["priority classification", "system analysis", "architectural"]
            },
            AgentType.PERFORMANCE_VIRTUOSO: {
                "keywords": ["latency", "throughput", "optimization", "bottleneck", "scalability", "ms", "performance"],
                "expected_response_time_ms": 12000,
                "expected_output_length": 1800,
                "specialization_indicators": ["quantified metrics", "before/after", "optimization"]
            },
            AgentType.FORTRESS_GUARDIAN: {
                "keywords": ["security", "vulnerability", "CVSS", "authentication", "encryption", "compliance"],
                "expected_response_time_ms": 18000,  # Thorough security analysis
                "expected_output_length": 2200,
                "specialization_indicators": ["threat model", "CVSS scoring", "security controls"]
            },
            AgentType.INTERFACE_ARTISAN: {
                "keywords": ["accessibility", "WCAG", "UX", "responsive", "Core Web Vitals", "frontend"],
                "expected_response_time_ms": 14000,
                "expected_output_length": 1600,
                "specialization_indicators": ["accessibility compliance", "user experience", "design system"]
            },
            AgentType.DATA_ARCHITECT: {
                "keywords": ["schema", "validation", "migration", "integrity", "data quality"],
                "expected_response_time_ms": 16000,
                "expected_output_length": 1900,
                "specialization_indicators": ["data integrity", "schema evolution", "validation framework"]
            },
            AgentType.INTEGRATION_MAESTRO: {
                "keywords": ["API", "resilience", "circuit breaker", "retry", "fallback", "integration"],
                "expected_response_time_ms": 13000,
                "expected_output_length": 1700,
                "specialization_indicators": ["resilience patterns", "API integration", "failure handling"]
            }
        }
    
    def calculate_specialization_score(self, agent_type: AgentType, response_text: str, 
                                     execution_time_ms: int, output_length: int) -> float:
        """Calculate how well an agent demonstrated its specialization"""
        if agent_type not in self.specialization_baselines:
            return 0.5  # Default score for unmapped agents
        
        baseline = self.specialization_baselines[agent_type]
        score_components = []
        
        # 1. Keyword presence score (40% weight)
        keyword_score = self._calculate_keyword_score(response_text, baseline["keywords"])
        score_components.append(("keyword_presence", keyword_score, 0.4))
        
        # 2. Specialization indicators score (30% weight)
        indicator_score = self._calculate_indicator_score(response_text, baseline["specialization_indicators"])
        score_components.append(("specialization_indicators", indicator_score, 0.3))
        
        # 3. Response depth score (20% weight)
        depth_score = self._calculate_depth_score(output_length, baseline["expected_output_length"])
        score_components.append(("response_depth", depth_score, 0.2))
        
        # 4. Execution efficiency score (10% weight)
        efficiency_score = self._calculate_efficiency_score(execution_time_ms, baseline["expected_response_time_ms"])
        score_components.append(("execution_efficiency", efficiency_score, 0.1))
        
        # Calculate weighted average
        total_score = sum(score * weight for _, score, weight in score_components)
        
        logger.info(f"Specialization score for {agent_type.value}: {total_score:.3f}")
        for component, score, weight in score_components:
            logger.debug(f"  {component}: {score:.3f} (weight: {weight})")
        
        return min(1.0, max(0.0, total_score))
    
    def _calculate_keyword_score(self, text: str, keywords: List[str]) -> float:
        """Calculate score based on presence of specialization keywords"""
        text_lower = text.lower()
        found_keywords = sum(1 for keyword in keywords if keyword.lower() in text_lower)
        return found_keywords / len(keywords) if keywords else 0.0
    
    def _calculate_indicator_score(self, text: str, indicators: List[str]) -> float:
        """Calculate score based on presence of specialization indicators"""
        text_lower = text.lower()
        found_indicators = sum(1 for indicator in indicators if indicator.lower() in text_lower)
        return found_indicators / len(indicators) if indicators else 0.0
    
    def _calculate_depth_score(self, actual_length: int, expected_length: int) -> float:
        """Calculate score based on response depth/comprehensiveness"""
        if expected_length == 0:
            return 1.0
        
        ratio = actual_length / expected_length
        if ratio >= 0.8:  # At least 80% of expected length
            return 1.0
        elif ratio >= 0.5:  # At least 50% of expected length
            return 0.7
        elif ratio >= 0.3:  # At least 30% of expected length
            return 0.4
        else:
            return 0.1
    
    def _calculate_efficiency_score(self, actual_time_ms: int, expected_time_ms: int) -> float:
        """Calculate score based on execution efficiency"""
        if expected_time_ms == 0:
            return 1.0
        
        ratio = actual_time_ms / expected_time_ms
        if ratio <= 1.0:  # Faster than expected
            return 1.0
        elif ratio <= 1.5:  # Within 50% of expected
            return 0.8
        elif ratio <= 2.0:  # Within 100% of expected
            return 0.6
        else:
            return 0.3
    
    def record_agent_performance(self, agent_type: AgentType, task_description: str,
                               response_text: str, execution_time_ms: int,
                               user_feedback: Optional[str] = None) -> AgentPerformanceMetric:
        """Record performance metrics for a single agent execution"""
        
        # Generate unique task ID
        task_id = hashlib.md5(f"{task_description}{datetime.now().isoformat()}".encode()).hexdigest()[:12]
        
        # Calculate specialization score
        specialization_score = self.calculate_specialization_score(
            agent_type, response_text, execution_time_ms, len(response_text)
        )
        
        # Extract specialization keywords found
        baseline = self.specialization_baselines.get(agent_type, {})
        keywords_found = [kw for kw in baseline.get("keywords", []) 
                         if kw.lower() in response_text.lower()]
        
        # Create performance metric
        metric = AgentPerformanceMetric(
            agent_type=agent_type,
            metric_type=MetricType.SPECIALIZATION_VALUE,
            value=specialization_score,
            timestamp=datetime.now(),
            task_id=task_id,
            execution_time_ms=execution_time_ms,
            context_size=len(task_description),
            output_length=len(response_text),
            specialization_keywords=keywords_found,
            user_feedback=user_feedback
        )
        
        self.metrics.append(metric)
        logger.info(f"Recorded performance metric: {agent_type.value} scored {specialization_score:.3f}")
        
        return metric
    
    def record_workflow_performance(self, workflow_id: str, agents_involved: List[AgentType],
                                  total_execution_time_ms: int, output_quality: float,
                                  business_value: float) -> MultiAgentWorkflowMetric:
        """Record performance metrics for multi-agent workflow"""
        
        # Calculate handoff efficiency (simplified - could be enhanced with actual handoff analysis)
        handoff_efficiency = min(1.0, 1.0 - (len(agents_involved) - 1) * 0.1)  # Penalty for each handoff
        
        # Calculate collaboration score
        collaboration_score = (output_quality + business_value + handoff_efficiency) / 3
        
        workflow_metric = MultiAgentWorkflowMetric(
            workflow_id=workflow_id,
            agents_involved=agents_involved,
            total_execution_time_ms=total_execution_time_ms,
            handoff_efficiency=handoff_efficiency,
            output_quality=output_quality,
            collaboration_score=collaboration_score,
            business_value_delivered=business_value,
            timestamp=datetime.now()
        )
        
        self.workflow_metrics.append(workflow_metric)
        logger.info(f"Recorded workflow metric: {workflow_id} scored {collaboration_score:.3f}")
        
        return workflow_metric
    
    def generate_agent_performance_report(self, agent_type: Optional[AgentType] = None,
                                        days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive performance report for agent(s)"""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        relevant_metrics = [m for m in self.metrics if m.timestamp >= cutoff_date]
        
        if agent_type:
            relevant_metrics = [m for m in relevant_metrics if m.agent_type == agent_type]
        
        if not relevant_metrics:
            return {"error": "No metrics found for the specified criteria"}
        
        # Group metrics by agent type
        agent_groups = {}
        for metric in relevant_metrics:
            if metric.agent_type not in agent_groups:
                agent_groups[metric.agent_type] = []
            agent_groups[metric.agent_type].append(metric)
        
        report = {
            "report_generated": datetime.now().isoformat(),
            "period_days": days,
            "total_executions": len(relevant_metrics),
            "agent_performance": {}
        }
        
        for agent, metrics in agent_groups.items():
            values = [m.value for m in metrics]
            execution_times = [m.execution_time_ms for m in metrics]
            
            agent_report = {
                "executions": len(metrics),
                "average_specialization_score": statistics.mean(values),
                "median_specialization_score": statistics.median(values),
                "min_specialization_score": min(values),
                "max_specialization_score": max(values),
                "average_execution_time_ms": statistics.mean(execution_times),
                "median_execution_time_ms": statistics.median(execution_times),
                "performance_trend": self._calculate_performance_trend(metrics),
                "most_common_keywords": self._get_most_common_keywords(metrics)
            }
            
            report["agent_performance"][agent.value] = agent_report
        
        return report
    
    def _calculate_performance_trend(self, metrics: List[AgentPerformanceMetric]) -> str:
        """Calculate performance trend over time"""
        if len(metrics) < 3:
            return "insufficient_data"
        
        # Sort by timestamp
        sorted_metrics = sorted(metrics, key=lambda m: m.timestamp)
        
        # Split into first and second half
        mid_point = len(sorted_metrics) // 2
        first_half = sorted_metrics[:mid_point]
        second_half = sorted_metrics[mid_point:]
        
        first_avg = statistics.mean(m.value for m in first_half)
        second_avg = statistics.mean(m.value for m in second_half)
        
        if second_avg > first_avg + 0.05:
            return "improving"
        elif second_avg < first_avg - 0.05:
            return "declining"
        else:
            return "stable"
    
    def _get_most_common_keywords(self, metrics: List[AgentPerformanceMetric]) -> List[str]:
        """Get most commonly found specialization keywords"""
        keyword_counts = {}
        for metric in metrics:
            for keyword in metric.specialization_keywords:
                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        # Return top 5 most common keywords
        sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
        return [kw for kw, count in sorted_keywords[:5]]
    
    def generate_ecosystem_health_report(self) -> Dict[str, Any]:
        """Generate overall ecosystem health and performance report"""
        
        recent_metrics = [m for m in self.metrics 
                         if m.timestamp >= datetime.now() - timedelta(days=7)]
        recent_workflows = [w for w in self.workflow_metrics 
                          if w.timestamp >= datetime.now() - timedelta(days=7)]
        
        if not recent_metrics:
            return {"error": "No recent metrics available"}
        
        # Calculate ecosystem-wide metrics
        all_scores = [m.value for m in recent_metrics]
        workflow_scores = [w.collaboration_score for w in recent_workflows]
        
        # Agent coverage analysis
        active_agents = set(m.agent_type for m in recent_metrics)
        total_agents = len(AgentType)
        coverage_percentage = (len(active_agents) / total_agents) * 100
        
        # Performance distribution
        high_performers = len([s for s in all_scores if s >= 0.8])
        medium_performers = len([s for s in all_scores if 0.6 <= s < 0.8])
        low_performers = len([s for s in all_scores if s < 0.6])
        
        report = {
            "ecosystem_health_score": statistics.mean(all_scores) if all_scores else 0.0,
            "agent_coverage_percentage": coverage_percentage,
            "total_executions_7_days": len(recent_metrics),
            "total_workflows_7_days": len(recent_workflows),
            "performance_distribution": {
                "high_performers": high_performers,
                "medium_performers": medium_performers,
                "low_performers": low_performers
            },
            "average_workflow_score": statistics.mean(workflow_scores) if workflow_scores else 0.0,
            "top_performing_agents": self._get_top_performing_agents(recent_metrics),
            "underperforming_agents": self._get_underperforming_agents(recent_metrics),
            "recommendations": self._generate_improvement_recommendations(recent_metrics)
        }
        
        return report
    
    def _get_top_performing_agents(self, metrics: List[AgentPerformanceMetric]) -> List[Dict[str, Any]]:
        """Identify top performing agents"""
        agent_scores = {}
        for metric in metrics:
            if metric.agent_type not in agent_scores:
                agent_scores[metric.agent_type] = []
            agent_scores[metric.agent_type].append(metric.value)
        
        agent_averages = {agent: statistics.mean(scores) 
                         for agent, scores in agent_scores.items()}
        
        sorted_agents = sorted(agent_averages.items(), key=lambda x: x[1], reverse=True)
        
        return [{"agent": agent.value, "average_score": score} 
                for agent, score in sorted_agents[:3]]
    
    def _get_underperforming_agents(self, metrics: List[AgentPerformanceMetric]) -> List[Dict[str, Any]]:
        """Identify underperforming agents"""
        agent_scores = {}
        for metric in metrics:
            if metric.agent_type not in agent_scores:
                agent_scores[metric.agent_type] = []
            agent_scores[metric.agent_type].append(metric.value)
        
        agent_averages = {agent: statistics.mean(scores) 
                         for agent, scores in agent_scores.items()}
        
        underperformers = [(agent, score) for agent, score in agent_averages.items() if score < 0.6]
        
        return [{"agent": agent.value, "average_score": score, "improvement_needed": True} 
                for agent, score in underperformers]
    
    def _generate_improvement_recommendations(self, metrics: List[AgentPerformanceMetric]) -> List[str]:
        """Generate actionable recommendations for ecosystem improvement"""
        recommendations = []
        
        # Analyze overall performance
        avg_score = statistics.mean(m.value for m in metrics)
        
        if avg_score < 0.7:
            recommendations.append("Overall ecosystem performance below target (0.7). Consider agent optimization.")
        
        # Analyze agent coverage
        active_agents = set(m.agent_type for m in metrics)
        if len(active_agents) < len(AgentType) * 0.8:
            recommendations.append("Low agent utilization detected. Consider promoting underused specialist agents.")
        
        # Analyze execution times
        avg_execution_time = statistics.mean(m.execution_time_ms for m in metrics)
        if avg_execution_time > 20000:  # 20 seconds
            recommendations.append("High average execution times. Consider performance optimization.")
        
        # Analyze specialization effectiveness
        specialization_scores = [m.value for m in metrics if m.metric_type == MetricType.SPECIALIZATION_VALUE]
        if specialization_scores and statistics.mean(specialization_scores) < 0.75:
            recommendations.append("Specialization effectiveness below target. Review agent prompt engineering.")
        
        return recommendations
    
    def save_metrics(self):
        """Save all metrics to persistent storage"""
        data = {
            "metrics": [metric.to_dict() for metric in self.metrics],
            "workflow_metrics": [workflow.to_dict() for workflow in self.workflow_metrics],
            "last_updated": datetime.now().isoformat()
        }
        
        with open(self.data_store_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(self.metrics)} metrics and {len(self.workflow_metrics)} workflow metrics")
    
    def load_metrics(self):
        """Load metrics from persistent storage"""
        try:
            with open(self.data_store_path, 'r') as f:
                data = json.load(f)
            
            # Load individual metrics
            for metric_data in data.get("metrics", []):
                metric = AgentPerformanceMetric(
                    agent_type=AgentType(metric_data["agent_type"]),
                    metric_type=MetricType(metric_data["metric_type"]),
                    value=metric_data["value"],
                    timestamp=datetime.fromisoformat(metric_data["timestamp"]),
                    task_id=metric_data["task_id"],
                    execution_time_ms=metric_data["execution_time_ms"],
                    context_size=metric_data["context_size"],
                    output_length=metric_data["output_length"],
                    specialization_keywords=metric_data["specialization_keywords"],
                    user_feedback=metric_data.get("user_feedback")
                )
                self.metrics.append(metric)
            
            # Load workflow metrics
            for workflow_data in data.get("workflow_metrics", []):
                workflow = MultiAgentWorkflowMetric(
                    workflow_id=workflow_data["workflow_id"],
                    agents_involved=[AgentType(agent) for agent in workflow_data["agents_involved"]],
                    total_execution_time_ms=workflow_data["total_execution_time_ms"],
                    handoff_efficiency=workflow_data["handoff_efficiency"],
                    output_quality=workflow_data["output_quality"],
                    collaboration_score=workflow_data["collaboration_score"],
                    business_value_delivered=workflow_data["business_value_delivered"],
                    timestamp=datetime.fromisoformat(workflow_data["timestamp"])
                )
                self.workflow_metrics.append(workflow)
            
            logger.info(f"Loaded {len(self.metrics)} metrics and {len(self.workflow_metrics)} workflow metrics")
            
        except FileNotFoundError:
            logger.info("No existing metrics file found. Starting with empty metrics.")
        except Exception as e:
            logger.error(f"Error loading metrics: {e}")


# Example usage and testing functions
def test_agent_performance_measurement():
    """Test the agent performance measurement system"""
    
    analyzer = AgentPerformanceAnalyzer()
    
    # Simulate agent executions with different performance characteristics
    test_scenarios = [
        {
            "agent": AgentType.RELIABILITY_ENGINEER,
            "task": "Analyze system architecture for microservices deployment",
            "response": "Based on my methodical analysis of your microservices architecture, I've identified P0 priority issues in your service mesh configuration. The system demonstrates excellent operational patterns but requires monitoring improvements for production reliability...",
            "execution_time": 14500,
            "feedback": "Very thorough architectural analysis"
        },
        {
            "agent": AgentType.PERFORMANCE_VIRTUOSO,
            "task": "Optimize database query performance",
            "response": "Performance analysis complete. Current latency: 450ms, target: <100ms. Identified N+1 query bottlenecks. Optimization strategy: implement connection pooling (70% improvement), add database indexes (40% improvement), Redis caching (85% improvement)...",
            "execution_time": 11200,
            "feedback": "Excellent quantified analysis"
        },
        {
            "agent": AgentType.FORTRESS_GUARDIAN,
            "task": "Security audit of authentication system",
            "response": "Security vulnerability assessment complete. CVSS Score: 7.8 (High). Identified authentication bypass vulnerability. Threat model shows elevated risk for credential stuffing attacks. Recommend immediate implementation of rate limiting and MFA...",
            "execution_time": 16800,
            "feedback": "Comprehensive security analysis"
        }
    ]
    
    # Record performance metrics
    for scenario in test_scenarios:
        metric = analyzer.record_agent_performance(
            agent_type=scenario["agent"],
            task_description=scenario["task"],
            response_text=scenario["response"],
            execution_time_ms=scenario["execution_time"],
            user_feedback=scenario["feedback"]
        )
        print(f"Recorded metric for {scenario['agent'].value}: {metric.value:.3f}")
    
    # Test multi-agent workflow
    workflow_metric = analyzer.record_workflow_performance(
        workflow_id="ecommerce_optimization_001",
        agents_involved=[
            AgentType.PERFORMANCE_VIRTUOSO,
            AgentType.FORTRESS_GUARDIAN,
            AgentType.INTERFACE_ARTISAN
        ],
        total_execution_time_ms=45000,
        output_quality=0.92,
        business_value=0.88
    )
    print(f"Recorded workflow metric: {workflow_metric.collaboration_score:.3f}")
    
    # Generate performance report
    report = analyzer.generate_agent_performance_report()
    print("\n=== AGENT PERFORMANCE REPORT ===")
    print(json.dumps(report, indent=2))
    
    # Generate ecosystem health report
    health_report = analyzer.generate_ecosystem_health_report()
    print("\n=== ECOSYSTEM HEALTH REPORT ===")
    print(json.dumps(health_report, indent=2))
    
    # Save metrics
    analyzer.save_metrics()
    
    return analyzer


if __name__ == "__main__":
    # Run performance measurement test
    print("Testing Agent Performance Metrics Framework...")
    analyzer = test_agent_performance_measurement()
    print("\nAgent Performance Metrics Framework test completed successfully!")