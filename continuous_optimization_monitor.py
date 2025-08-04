#!/usr/bin/env python3
"""
Continuous Agent Optimization Monitoring System
==============================================

Real-time monitoring and alerting system for maintaining agent performance
at 75%+ specialization scores with automated improvement recommendations.

Author: Performance Virtuoso (Claude-Nexus Optimization Team)
Date: 2025-08-03
Version: 1.0.0
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import statistics
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent_optimization_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert levels for performance monitoring"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class PerformanceAlert:
    """Performance monitoring alert"""
    agent_type: str
    alert_level: AlertLevel
    metric: str
    current_value: float
    threshold: float
    message: str
    timestamp: datetime
    recommendation: str


class ContinuousOptimizationMonitor:
    """Continuous monitoring system for agent performance optimization"""
    
    def __init__(self, config_file: str = "optimization_config.json"):
        self.config_file = config_file
        self.config = self._load_config()
        self.alerts = []
        self.performance_history = {}
        
    def _load_config(self) -> Dict[str, Any]:
        """Load monitoring configuration"""
        default_config = {
            "monitoring_interval_minutes": 60,
            "performance_thresholds": {
                "critical_threshold": 0.60,  # Below 60% triggers critical alert
                "warning_threshold": 0.70,   # Below 70% triggers warning
                "target_threshold": 0.75     # Target performance level
            },
            "agents_to_monitor": [
                "reliability-engineer",
                "fortress-guardian", 
                "performance-virtuoso"
            ],
            "optimization_targets": {
                "reliability-engineer": {
                    "baseline": 0.406,
                    "current_target": 0.81,
                    "keywords": ["architecture", "reliability", "P0", "P1", "P2", "monitoring", "SLA", "operational"],
                    "indicators": ["priority classification", "system analysis", "architectural", "operational context", "SLA impact assessment"]
                },
                "fortress-guardian": {
                    "baseline": 0.487,
                    "current_target": 0.94,
                    "keywords": ["security", "vulnerability", "CVSS", "authentication", "encryption", "compliance", "penetration", "threat"],
                    "indicators": ["threat model", "CVSS scoring", "security controls", "vulnerability assessment", "penetration testing"]
                },
                "performance-virtuoso": {
                    "baseline": 0.506,
                    "current_target": 0.86,
                    "keywords": ["latency", "throughput", "optimization", "bottleneck", "scalability", "ms", "performance", "monitoring"],
                    "indicators": ["quantified metrics", "before/after", "optimization", "scalability assessment", "performance monitoring"]
                }
            },
            "alert_settings": {
                "email_alerts": False,
                "slack_alerts": False,
                "console_alerts": True,
                "log_alerts": True
            }
        }
        
        try:
            with open(self.config_file, 'r') as f:
                loaded_config = json.load(f)
                # Merge with defaults
                return {**default_config, **loaded_config}
        except FileNotFoundError:
            logger.info(f"Config file not found, using defaults. Creating {self.config_file}")
            self._save_config(default_config)
            return default_config
    
    def _save_config(self, config: Dict[str, Any]):
        """Save monitoring configuration"""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def check_agent_performance(self, agent_type: str, current_score: float, 
                              response_text: str = "") -> List[PerformanceAlert]:
        """Check individual agent performance and generate alerts"""
        alerts = []
        thresholds = self.config["performance_thresholds"]
        
        # Check performance thresholds
        if current_score < thresholds["critical_threshold"]:
            alert = PerformanceAlert(
                agent_type=agent_type,
                alert_level=AlertLevel.CRITICAL,
                metric="specialization_score",
                current_value=current_score,
                threshold=thresholds["critical_threshold"],
                message=f"CRITICAL: {agent_type} performance below {thresholds['critical_threshold']:.0%}",
                timestamp=datetime.now(),
                recommendation=self._generate_critical_recommendation(agent_type, current_score)
            )
            alerts.append(alert)
            
        elif current_score < thresholds["warning_threshold"]:
            alert = PerformanceAlert(
                agent_type=agent_type,
                alert_level=AlertLevel.WARNING,
                metric="specialization_score",
                current_value=current_score,
                threshold=thresholds["warning_threshold"],
                message=f"WARNING: {agent_type} performance below {thresholds['warning_threshold']:.0%}",
                timestamp=datetime.now(),
                recommendation=self._generate_warning_recommendation(agent_type, current_score)
            )
            alerts.append(alert)
            
        elif current_score >= thresholds["target_threshold"]:
            alert = PerformanceAlert(
                agent_type=agent_type,
                alert_level=AlertLevel.INFO,
                metric="specialization_score",
                current_value=current_score,
                threshold=thresholds["target_threshold"],
                message=f"SUCCESS: {agent_type} exceeding target performance {thresholds['target_threshold']:.0%}",
                timestamp=datetime.now(),
                recommendation="Continue monitoring. Consider expanding optimization to other agents."
            )
            alerts.append(alert)
        
        # Check keyword coverage if response text provided
        if response_text and agent_type in self.config["optimization_targets"]:
            keyword_coverage = self._analyze_keyword_coverage(agent_type, response_text)
            if keyword_coverage < 0.6:  # Less than 60% keyword coverage
                alert = PerformanceAlert(
                    agent_type=agent_type,
                    alert_level=AlertLevel.WARNING,
                    metric="keyword_coverage",
                    current_value=keyword_coverage,
                    threshold=0.6,
                    message=f"LOW KEYWORD COVERAGE: {agent_type} only {keyword_coverage:.0%} keyword utilization",
                    timestamp=datetime.now(),
                    recommendation=self._generate_keyword_recommendation(agent_type, keyword_coverage)
                )
                alerts.append(alert)
        
        # Store alerts
        self.alerts.extend(alerts)
        
        # Log alerts
        for alert in alerts:
            if alert.alert_level == AlertLevel.CRITICAL:
                logger.critical(f"{alert.message} - {alert.recommendation}")
            elif alert.alert_level == AlertLevel.WARNING:
                logger.warning(f"{alert.message} - {alert.recommendation}")
            else:
                logger.info(f"{alert.message}")
        
        return alerts
    
    def _analyze_keyword_coverage(self, agent_type: str, response_text: str) -> float:
        """Analyze keyword coverage in response text"""
        if agent_type not in self.config["optimization_targets"]:
            return 0.0
        
        keywords = self.config["optimization_targets"][agent_type]["keywords"]
        text_lower = response_text.lower()
        found_keywords = sum(1 for keyword in keywords if keyword.lower() in text_lower)
        
        return found_keywords / len(keywords) if keywords else 0.0
    
    def _generate_critical_recommendation(self, agent_type: str, current_score: float) -> str:
        """Generate recommendation for critical performance issues"""
        target_info = self.config["optimization_targets"].get(agent_type, {})
        baseline = target_info.get("baseline", 0.5)
        
        if current_score < baseline:
            return f"IMMEDIATE ACTION REQUIRED: Performance regression detected. Review recent prompt changes and revert if necessary. Implement emergency optimization focusing on keyword density and specialization indicators."
        else:
            return f"OPTIMIZATION NEEDED: Enhance prompt with missing specialization keywords and indicators. Focus on the 4 scoring components: keywords (40%), indicators (30%), depth (20%), efficiency (10%)."
    
    def _generate_warning_recommendation(self, agent_type: str, current_score: float) -> str:
        """Generate recommendation for warning-level performance issues"""
        return f"MINOR OPTIMIZATION: Fine-tune prompt to include more specialization keywords and methodology indicators. Current score of {current_score:.1%} is above baseline but below target."
    
    def _generate_keyword_recommendation(self, agent_type: str, coverage: float) -> str:
        """Generate recommendation for low keyword coverage"""
        target_info = self.config["optimization_targets"].get(agent_type, {})
        keywords = target_info.get("keywords", [])
        
        return f"KEYWORD OPTIMIZATION: Enhance prompt to include more of these specialization keywords: {', '.join(keywords[:3])}... Focus on natural integration of domain-specific terminology."
    
    def update_performance_history(self, agent_type: str, score: float, 
                                 metadata: Optional[Dict[str, Any]] = None):
        """Update performance history for trend analysis"""
        if agent_type not in self.performance_history:
            self.performance_history[agent_type] = []
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "score": score,
            "metadata": metadata or {}
        }
        
        self.performance_history[agent_type].append(entry)
        
        # Keep only last 100 entries per agent
        if len(self.performance_history[agent_type]) > 100:
            self.performance_history[agent_type] = self.performance_history[agent_type][-100:]
    
    def analyze_performance_trends(self, agent_type: str, days: int = 7) -> Dict[str, Any]:
        """Analyze performance trends over specified period"""
        if agent_type not in self.performance_history:
            return {"error": "No performance history found"}
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_entries = [
            entry for entry in self.performance_history[agent_type]
            if datetime.fromisoformat(entry["timestamp"]) >= cutoff_date
        ]
        
        if not recent_entries:
            return {"error": "No recent performance data"}
        
        scores = [entry["score"] for entry in recent_entries]
        
        # Calculate trend
        if len(scores) >= 3:
            mid_point = len(scores) // 2
            first_half_avg = statistics.mean(scores[:mid_point])
            second_half_avg = statistics.mean(scores[mid_point:])
            
            if second_half_avg > first_half_avg + 0.05:
                trend = "improving"
            elif second_half_avg < first_half_avg - 0.05:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        return {
            "agent_type": agent_type,
            "period_days": days,
            "data_points": len(recent_entries),
            "current_score": scores[-1] if scores else 0,
            "average_score": statistics.mean(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "trend": trend,
            "trend_direction": second_half_avg - first_half_avg if len(scores) >= 3 else 0
        }
    
    def generate_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization monitoring report"""
        report = {
            "report_timestamp": datetime.now().isoformat(),
            "monitoring_config": {
                "target_threshold": self.config["performance_thresholds"]["target_threshold"],
                "warning_threshold": self.config["performance_thresholds"]["warning_threshold"],
                "critical_threshold": self.config["performance_thresholds"]["critical_threshold"]
            },
            "agent_status": {},
            "recent_alerts": [],
            "trend_analysis": {},
            "recommendations": []
        }
        
        # Analyze each monitored agent
        for agent_type in self.config["agents_to_monitor"]:
            # Get recent alerts
            recent_alerts = [
                {
                    "level": alert.alert_level.value,
                    "metric": alert.metric,
                    "current_value": alert.current_value,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat()
                }
                for alert in self.alerts
                if alert.agent_type == agent_type and 
                   alert.timestamp >= datetime.now() - timedelta(hours=24)
            ]
            
            # Get trend analysis
            trend_analysis = self.analyze_performance_trends(agent_type)
            
            # Determine current status
            target_score = self.config["optimization_targets"][agent_type]["current_target"]
            current_score = trend_analysis.get("current_score", 0)
            
            if current_score >= target_score:
                status = "optimal"
            elif current_score >= self.config["performance_thresholds"]["target_threshold"]:
                status = "good"
            elif current_score >= self.config["performance_thresholds"]["warning_threshold"]:
                status = "warning"
            else:
                status = "critical"
            
            report["agent_status"][agent_type] = {
                "status": status,
                "current_score": current_score,
                "target_score": target_score,
                "recent_alerts": len(recent_alerts),
                "trend": trend_analysis.get("trend", "unknown")
            }
            
            report["recent_alerts"].extend(recent_alerts)
            report["trend_analysis"][agent_type] = trend_analysis
        
        # Generate global recommendations
        report["recommendations"] = self._generate_global_recommendations(report)
        
        return report
    
    def _generate_global_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate global optimization recommendations"""
        recommendations = []
        
        # Check overall system health
        agent_statuses = list(report["agent_status"].values())
        optimal_count = sum(1 for status in agent_statuses if status["status"] == "optimal")
        critical_count = sum(1 for status in agent_statuses if status["status"] == "critical")
        
        if critical_count > 0:
            recommendations.append(f"URGENT: {critical_count} agent(s) require immediate optimization attention")
        
        if optimal_count == len(agent_statuses):
            recommendations.append("EXCELLENT: All agents performing at optimal levels. Consider expanding optimization to additional agents.")
        elif optimal_count >= len(agent_statuses) * 0.67:
            recommendations.append("GOOD: Majority of agents optimized. Focus remaining optimization efforts on underperforming agents.")
        
        # Check for declining trends
        declining_agents = [
            agent for agent, analysis in report["trend_analysis"].items()
            if analysis.get("trend") == "declining"
        ]
        
        if declining_agents:
            recommendations.append(f"TREND ALERT: Performance declining for {', '.join(declining_agents)}. Review recent changes and implement corrective measures.")
        
        # Check alert frequency
        total_alerts = len(report["recent_alerts"])
        if total_alerts > 10:
            recommendations.append("HIGH ALERT FREQUENCY: Consider adjusting monitoring thresholds or implementing automated optimization.")
        
        return recommendations
    
    def save_monitoring_state(self, filename: str = "monitoring_state.json"):
        """Save current monitoring state"""
        state = {
            "config": self.config,
            "performance_history": self.performance_history,
            "alerts": [
                {
                    "agent_type": alert.agent_type,
                    "alert_level": alert.alert_level.value,
                    "metric": alert.metric,
                    "current_value": alert.current_value,
                    "threshold": alert.threshold,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                    "recommendation": alert.recommendation
                }
                for alert in self.alerts
            ],
            "last_updated": datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Monitoring state saved to {filename}")
    
    def load_monitoring_state(self, filename: str = "monitoring_state.json"):
        """Load monitoring state from file"""
        try:
            with open(filename, 'r') as f:
                state = json.load(f)
            
            self.config = state.get("config", self.config)
            self.performance_history = state.get("performance_history", {})
            
            # Reconstruct alerts
            self.alerts = []
            for alert_data in state.get("alerts", []):
                alert = PerformanceAlert(
                    agent_type=alert_data["agent_type"],
                    alert_level=AlertLevel(alert_data["alert_level"]),
                    metric=alert_data["metric"],
                    current_value=alert_data["current_value"],
                    threshold=alert_data["threshold"],
                    message=alert_data["message"],
                    timestamp=datetime.fromisoformat(alert_data["timestamp"]),
                    recommendation=alert_data["recommendation"]
                )
                self.alerts.append(alert)
            
            logger.info(f"Monitoring state loaded from {filename}")
            
        except FileNotFoundError:
            logger.info(f"No existing monitoring state found at {filename}")
        except Exception as e:
            logger.error(f"Error loading monitoring state: {e}")
    
    def start_continuous_monitoring(self):
        """Start continuous monitoring loop"""
        logger.info("Starting continuous agent optimization monitoring...")
        
        interval_minutes = self.config["monitoring_interval_minutes"]
        
        while True:
            try:
                # This would integrate with actual agent performance data
                # For now, we'll simulate monitoring
                logger.info("Performing monitoring check...")
                
                # Generate and save monitoring report
                report = self.generate_optimization_report()
                
                # Log summary
                logger.info(f"Monitoring complete. {len(report['recent_alerts'])} recent alerts.")
                
                # Save state
                self.save_monitoring_state()
                
                # Wait for next monitoring cycle
                time.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(60)  # Wait 1 minute before retrying


def main():
    """Main monitoring function for testing"""
    monitor = ContinuousOptimizationMonitor()
    
    # Simulate some performance checks
    print("🔍 Testing Continuous Optimization Monitor")
    print("=" * 50)
    
    # Test current optimized performance
    test_scenarios = [
        ("reliability-engineer", 0.81, "P0 priority architectural analysis with SLA monitoring and operational excellence"),
        ("fortress-guardian", 0.94, "CVSS vulnerability assessment with threat modeling and penetration testing"),
        ("performance-virtuoso", 0.86, "Latency optimization with throughput analysis and scalability monitoring")
    ]
    
    for agent_type, score, response in test_scenarios:
        print(f"\n📊 Checking {agent_type} performance...")
        alerts = monitor.check_agent_performance(agent_type, score, response)
        monitor.update_performance_history(agent_type, score)
        
        print(f"   Score: {score:.0%}")
        print(f"   Alerts: {len(alerts)}")
        for alert in alerts:
            print(f"   • {alert.alert_level.value.upper()}: {alert.message}")
    
    # Generate comprehensive report
    print(f"\n📈 Generating optimization report...")
    report = monitor.generate_optimization_report()
    
    print(f"\n🎯 OPTIMIZATION MONITORING SUMMARY")
    print(f"=" * 40)
    for agent, status in report["agent_status"].items():
        print(f"{agent}: {status['status'].upper()} ({status['current_score']:.0%})")
    
    print(f"\n💡 Recommendations:")
    for rec in report["recommendations"]:
        print(f"• {rec}")
    
    # Save monitoring state
    monitor.save_monitoring_state()
    
    print(f"\n💾 Monitoring state saved successfully")
    
    return monitor


if __name__ == "__main__":
    monitor = main()