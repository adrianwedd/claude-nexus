#!/usr/bin/env python3
"""
Intelligent Agent Selection Algorithm for Claude Nexus Ecosystem
Analyzes PR/issue content and selects the most appropriate specialized agents
"""

import json
import re
import sys
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AgentType(Enum):
    # Core Engineering
    RELIABILITY_ENGINEER = "reliability-engineer"
    CODE_SOVEREIGN = "code-sovereign"
    PERFORMANCE_VIRTUOSO = "performance-virtuoso"
    
    # Infrastructure & Operations
    CLOUD_NAVIGATOR = "cloud-navigator"
    DEPLOYMENT_COMMANDER = "deployment-commander"
    DEVEX_CURATOR = "devex-curator"
    
    # Security & Quality
    FORTRESS_GUARDIAN = "fortress-guardian"
    QUALITY_ASSURANCE_ENGINEER = "quality-assurance-engineer"
    
    # Integration & Data
    INTEGRATION_MAESTRO = "integration-maestro"
    DATA_ARCHITECT = "data-architect"
    DATA_FLOW_ARCHITECT = "data-flow-architect"
    
    # User Experience & Documentation
    INTERFACE_ARTISAN = "interface-artisan"
    MOBILE_PLATFORM_SPECIALIST = "mobile-platform-specialist"
    KNOWLEDGE_CURATOR = "knowledge-curator"
    
    # Advanced Capabilities
    INTELLIGENCE_ORCHESTRATOR = "intelligence-orchestrator"
    REPOSITORY_SURGEON = "repository-surgeon"

@dataclass
class AgentMatch:
    agent: AgentType
    confidence: float
    reasoning: List[str]
    priority: int  # 1=high, 2=medium, 3=low

class AgentSelector:
    def __init__(self):
        self.file_patterns = self._initialize_file_patterns()
        self.keyword_patterns = self._initialize_keyword_patterns()
        self.agent_descriptions = self._initialize_agent_descriptions()
        
    def _initialize_file_patterns(self) -> Dict[AgentType, List[str]]:
        """Define file patterns that indicate specific agent expertise"""
        return {
            AgentType.INTERFACE_ARTISAN: [
                r'.*\.(css|scss|sass|less|styl)$',
                r'.*\.(html|htm|jsx|tsx|vue|svelte)$',
                r'.*/components/.*',
                r'.*/styles/.*',
                r'.*\.stories\.(js|ts|jsx|tsx)$'
            ],
            AgentType.MOBILE_PLATFORM_SPECIALIST: [
                r'.*\.(swift|kt|java|dart)$',
                r'.*/android/.*',
                r'.*/ios/.*',
                r'.*\.podspec$',
                r'.*build\.gradle$',
                r'.*pubspec\.yaml$'
            ],
            AgentType.PERFORMANCE_VIRTUOSO: [
                r'.*benchmark.*',
                r'.*performance.*',
                r'.*\.perf\.',
                r'.*lighthouse.*',
                r'.*webpack\.config.*'
            ],
            AgentType.FORTRESS_GUARDIAN: [
                r'.*security.*',
                r'.*auth.*',
                r'.*oauth.*',
                r'.*jwt.*',
                r'.*\.env.*',
                r'.*secrets.*'
            ],
            AgentType.CLOUD_NAVIGATOR: [
                r'.*k8s/.*',
                r'.*kubernetes/.*',
                r'.*Dockerfile.*',
                r'.*docker-compose.*',
                r'.*helm/.*',
                r'.*\.yaml$',
                r'.*\.yml$'
            ],
            AgentType.DEPLOYMENT_COMMANDER: [
                r'.*\.github/workflows/.*',
                r'.*\.gitlab-ci\.yml$',
                r'.*Jenkinsfile.*',
                r'.*terraform/.*',
                r'.*\.tf$',
                r'.*ansible/.*'
            ],
            AgentType.DATA_ARCHITECT: [
                r'.*\.sql$',
                r'.*migrations/.*',
                r'.*schema.*',
                r'.*models/.*',
                r'.*\.json$'
            ],
            AgentType.DATA_FLOW_ARCHITECT: [
                r'.*api/.*',
                r'.*backend/.*',
                r'.*server/.*',
                r'.*\.go$',
                r'.*\.rs$',
                r'.*\.py$'
            ],
            AgentType.DEVEX_CURATOR: [
                r'.*package\.json$',
                r'.*Makefile$',
                r'.*\.vscode/.*',
                r'.*\.editorconfig$',
                r'.*dev-scripts/.*'
            ],
            AgentType.QUALITY_ASSURANCE_ENGINEER: [
                r'.*test.*',
                r'.*spec.*',
                r'.*\.test\.(js|ts|py|go|rs)$',
                r'.*\.spec\.(js|ts|py|go|rs)$',
                r'.*cypress/.*',
                r'.*jest\.config.*'
            ]
        }
    
    def _initialize_keyword_patterns(self) -> Dict[AgentType, List[str]]:
        """Define keyword patterns that indicate specific agent expertise"""
        return {
            AgentType.RELIABILITY_ENGINEER: [
                'architecture', 'reliability', 'scalability', 'system design',
                'technical debt', 'refactor', 'maintainability', 'documentation'
            ],
            AgentType.CODE_SOVEREIGN: [
                'code quality', 'best practices', 'clean code', 'design patterns',
                'code review', 'architecture review', 'refactoring'
            ],
            AgentType.PERFORMANCE_VIRTUOSO: [
                'performance', 'optimization', 'speed', 'latency', 'throughput',
                'bottleneck', 'profiling', 'caching', 'memory', 'cpu'
            ],
            AgentType.CLOUD_NAVIGATOR: [
                'kubernetes', 'k8s', 'docker', 'cloud', 'aws', 'gcp', 'azure',
                'microservices', 'containerization', 'orchestration'
            ],
            AgentType.DEPLOYMENT_COMMANDER: [
                'ci/cd', 'pipeline', 'deployment', 'devops', 'infrastructure',
                'terraform', 'ansible', 'github actions', 'jenkins'
            ],
            AgentType.FORTRESS_GUARDIAN: [
                'security', 'authentication', 'authorization', 'oauth', 'jwt',
                'vulnerability', 'encryption', 'compliance', 'gdpr', 'soc2'
            ],
            AgentType.INTERFACE_ARTISAN: [
                'ui', 'ux', 'frontend', 'responsive', 'accessibility', 'wcag',
                'design system', 'components', 'css', 'styling'
            ],
            AgentType.MOBILE_PLATFORM_SPECIALIST: [
                'mobile', 'ios', 'android', 'react native', 'flutter',
                'cross-platform', 'native', 'app store', 'play store'
            ],
            AgentType.INTEGRATION_MAESTRO: [
                'api', 'integration', 'webhook', 'third-party', 'rate limiting',
                'circuit breaker', 'resilience', 'retry', 'backoff'
            ],
            AgentType.DATA_ARCHITECT: [
                'database', 'schema', 'migration', 'data model', 'validation',
                'json schema', 'integrity', 'consistency'
            ],
            AgentType.DATA_FLOW_ARCHITECT: [
                'backend', 'api', 'database', 'query optimization', 'caching',
                'data pipeline', 'stream processing', 'event-driven'
            ],
            AgentType.QUALITY_ASSURANCE_ENGINEER: [
                'testing', 'test automation', 'quality', 'coverage', 'unit test',
                'integration test', 'e2e test', 'qa', 'cypress', 'jest'
            ],
            AgentType.DEVEX_CURATOR: [
                'developer experience', 'workflow', 'tooling', 'productivity',
                'automation', 'onboarding', 'documentation'
            ],
            AgentType.KNOWLEDGE_CURATOR: [
                'documentation', 'readme', 'api docs', 'guide', 'tutorial',
                'knowledge base', 'wiki', 'help'
            ],
            AgentType.INTELLIGENCE_ORCHESTRATOR: [
                'ai', 'ml', 'machine learning', 'artificial intelligence',
                'model', 'neural network', 'prompt engineering'
            ],
            AgentType.REPOSITORY_SURGEON: [
                'repository', 'project structure', 'organization', 'cleanup',
                'maintenance', 'issue grooming', 'technical debt'
            ]
        }
    
    def _initialize_agent_descriptions(self) -> Dict[AgentType, str]:
        """Agent descriptions for reasoning output"""
        return {
            AgentType.RELIABILITY_ENGINEER: "Elite systems architect with 10x programmer precision",
            AgentType.CODE_SOVEREIGN: "Regal code quality specialist focused on architectural excellence",
            AgentType.PERFORMANCE_VIRTUOSO: "Elite performance engineering specialist focused on optimization",
            AgentType.CLOUD_NAVIGATOR: "Elite cloud architecture specialist with atmospheric computing vision",
            AgentType.DEPLOYMENT_COMMANDER: "Elite infrastructure specialist with military-grade deployment precision",
            AgentType.FORTRESS_GUARDIAN: "Elite security specialist operating under zero-trust principles",
            AgentType.INTERFACE_ARTISAN: "Master frontend developer with pixel-perfect perception",
            AgentType.MOBILE_PLATFORM_SPECIALIST: "Cross-platform mobile expert with omnipresent mastery",
            AgentType.INTEGRATION_MAESTRO: "API integration specialist building resilient architectures",
            AgentType.DATA_ARCHITECT: "Elite data specialist creating self-validating architectures",
            AgentType.DATA_FLOW_ARCHITECT: "Backend systems engineer with omniscient data vision",
            AgentType.QUALITY_ASSURANCE_ENGINEER: "Elite testing strategist with prophetic failure detection",
            AgentType.DEVEX_CURATOR: "Flow state specialist eliminating development friction",
            AgentType.KNOWLEDGE_CURATOR: "Elite documentation specialist transforming technical complexity",
            AgentType.INTELLIGENCE_ORCHESTRATOR: "AI/ML systems architect with neural network synthesis",
            AgentType.REPOSITORY_SURGEON: "Elite repository specialist transforming chaotic systems"
        }
    
    def analyze_files(self, files: List[str]) -> Dict[AgentType, float]:
        """Analyze file patterns to determine agent relevance"""
        scores = {}
        
        for agent, patterns in self.file_patterns.items():
            score = 0.0
            matches = 0
            
            for file in files:
                for pattern in patterns:
                    if re.match(pattern, file, re.IGNORECASE):
                        matches += 1
                        score += 1.0
                        break
            
            if matches > 0:
                # Normalize score based on total files
                scores[agent] = min(score / len(files), 1.0)
                
        return scores
    
    def analyze_content(self, content: str) -> Dict[AgentType, float]:
        """Analyze text content to determine agent relevance"""
        scores = {}
        content_lower = content.lower()
        
        for agent, keywords in self.keyword_patterns.items():
            score = 0.0
            matches = 0
            
            for keyword in keywords:
                if keyword.lower() in content_lower:
                    matches += 1
                    # Weight longer keywords more heavily
                    score += len(keyword.split()) * 0.1
            
            if matches > 0:
                scores[agent] = min(score, 1.0)
                
        return scores
    
    def analyze_labels(self, labels: List[str]) -> Dict[AgentType, float]:
        """Analyze issue/PR labels to determine agent relevance"""
        scores = {}
        label_text = ' '.join(labels).lower()
        
        # Direct label mappings
        label_mappings = {
            'bug': [AgentType.QUALITY_ASSURANCE_ENGINEER, AgentType.RELIABILITY_ENGINEER],
            'performance': [AgentType.PERFORMANCE_VIRTUOSO],
            'security': [AgentType.FORTRESS_GUARDIAN],
            'frontend': [AgentType.INTERFACE_ARTISAN],
            'backend': [AgentType.DATA_FLOW_ARCHITECT],
            'mobile': [AgentType.MOBILE_PLATFORM_SPECIALIST],
            'deployment': [AgentType.DEPLOYMENT_COMMANDER],
            'infrastructure': [AgentType.CLOUD_NAVIGATOR],
            'documentation': [AgentType.KNOWLEDGE_CURATOR],
            'testing': [AgentType.QUALITY_ASSURANCE_ENGINEER],
            'api': [AgentType.INTEGRATION_MAESTRO],
            'database': [AgentType.DATA_ARCHITECT],
            'devex': [AgentType.DEVEX_CURATOR]
        }
        
        for label in labels:
            label_lower = label.lower()
            for pattern, agents in label_mappings.items():
                if pattern in label_lower:
                    for agent in agents:
                        scores[agent] = scores.get(agent, 0) + 0.5
        
        return scores
    
    def get_collaboration_patterns(self, primary_agents: List[AgentType]) -> Dict[str, List[AgentType]]:
        """Define intelligent collaboration patterns"""
        patterns = {
            'security_review': [AgentType.FORTRESS_GUARDIAN, AgentType.CODE_SOVEREIGN],
            'performance_optimization': [AgentType.PERFORMANCE_VIRTUOSO, AgentType.DATA_FLOW_ARCHITECT],
            'frontend_excellence': [AgentType.INTERFACE_ARTISAN, AgentType.QUALITY_ASSURANCE_ENGINEER],
            'backend_optimization': [AgentType.DATA_FLOW_ARCHITECT, AgentType.PERFORMANCE_VIRTUOSO],
            'infrastructure_review': [AgentType.CLOUD_NAVIGATOR, AgentType.DEPLOYMENT_COMMANDER],
            'quality_gates': [AgentType.QUALITY_ASSURANCE_ENGINEER, AgentType.CODE_SOVEREIGN],
            'mobile_excellence': [AgentType.MOBILE_PLATFORM_SPECIALIST, AgentType.PERFORMANCE_VIRTUOSO],
            'api_resilience': [AgentType.INTEGRATION_MAESTRO, AgentType.FORTRESS_GUARDIAN],
            'comprehensive_review': [AgentType.RELIABILITY_ENGINEER, AgentType.CODE_SOVEREIGN, AgentType.QUALITY_ASSURANCE_ENGINEER]
        }
        
        active_patterns = {}
        for pattern_name, pattern_agents in patterns.items():
            if any(agent in primary_agents for agent in pattern_agents):
                active_patterns[pattern_name] = pattern_agents
        
        return active_patterns
    
    def select_agents(self, 
                     title: str = "", 
                     body: str = "", 
                     files: List[str] = None, 
                     labels: List[str] = None,
                     pr_type: str = "pull_request") -> List[AgentMatch]:
        """
        Select the most appropriate agents based on context analysis
        """
        if files is None:
            files = []
        if labels is None:
            labels = []
        
        logger.info(f"Analyzing {pr_type} with {len(files)} files and {len(labels)} labels")
        
        # Combine all text content
        content = f"{title} {body}".strip()
        
        # Get scores from different analysis methods
        file_scores = self.analyze_files(files) if files else {}
        content_scores = self.analyze_content(content) if content else {}
        label_scores = self.analyze_labels(labels) if labels else {}
        
        # Combine scores with weights
        combined_scores = {}
        all_agents = set(file_scores.keys()) | set(content_scores.keys()) | set(label_scores.keys())
        
        for agent in all_agents:
            file_weight = 0.4
            content_weight = 0.4
            label_weight = 0.2
            
            score = (
                file_scores.get(agent, 0) * file_weight +
                content_scores.get(agent, 0) * content_weight +
                label_scores.get(agent, 0) * label_weight
            )
            combined_scores[agent] = score
        
        # Create agent matches with reasoning
        matches = []
        for agent, score in combined_scores.items():
            if score > 0.1:  # Minimum threshold
                reasoning = []
                
                if agent in file_scores:
                    reasoning.append(f"File pattern match (score: {file_scores[agent]:.2f})")
                if agent in content_scores:
                    reasoning.append(f"Content analysis match (score: {content_scores[agent]:.2f})")
                if agent in label_scores:
                    reasoning.append(f"Label match (score: {label_scores[agent]:.2f})")
                
                # Determine priority based on score
                if score >= 0.7:
                    priority = 1  # High
                elif score >= 0.4:
                    priority = 2  # Medium
                else:
                    priority = 3  # Low
                
                matches.append(AgentMatch(
                    agent=agent,
                    confidence=score,
                    reasoning=reasoning,
                    priority=priority
                ))
        
        # Sort by confidence score (descending)
        matches.sort(key=lambda x: x.confidence, reverse=True)
        
        # Always include Repository Surgeon for comprehensive analysis if no specific matches
        if not matches:
            logger.info("No specific agent matches found, defaulting to Repository Surgeon")
            matches.append(AgentMatch(
                agent=AgentType.REPOSITORY_SURGEON,
                confidence=0.5,
                reasoning=["Default comprehensive analysis agent"],
                priority=2
            ))
        
        logger.info(f"Selected {len(matches)} agents with confidence scores: {[f'{m.agent.value}({m.confidence:.2f})' for m in matches[:3]]}")
        
        return matches[:5]  # Limit to top 5 matches

    def format_output(self, matches: List[AgentMatch], include_collaboration: bool = True) -> Dict:
        """Format the agent selection results for GitHub Actions output"""
        primary_agents = [match.agent for match in matches if match.priority <= 2]
        
        output = {
            'primary_agent': matches[0].agent.value if matches else 'repository-surgeon',
            'all_agents': [match.agent.value for match in matches],
            'agent_details': [
                {
                    'agent': match.agent.value,
                    'confidence': match.confidence,
                    'priority': match.priority,
                    'reasoning': match.reasoning,
                    'description': self.agent_descriptions.get(match.agent, "Specialized agent")
                }
                for match in matches
            ]
        }
        
        if include_collaboration:
            collaboration_patterns = self.get_collaboration_patterns(primary_agents)
            output['collaboration_patterns'] = {
                pattern: [agent.value for agent in agents]
                for pattern, agents in collaboration_patterns.items()
            }
        
        return output

def main():
    """Main CLI interface for agent selection"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Select appropriate Claude Nexus agents')
    parser.add_argument('--title', default='', help='PR/Issue title')
    parser.add_argument('--body', default='', help='PR/Issue body')
    parser.add_argument('--files', nargs='*', default=[], help='Changed files')
    parser.add_argument('--labels', nargs='*', default=[], help='Issue/PR labels')
    parser.add_argument('--type', choices=['pull_request', 'issue'], default='pull_request')
    parser.add_argument('--output-format', choices=['json', 'github-actions'], default='json')
    
    args = parser.parse_args()
    
    selector = AgentSelector()
    matches = selector.select_agents(
        title=args.title,
        body=args.body,
        files=args.files,
        labels=args.labels,
        pr_type=args.type
    )
    
    result = selector.format_output(matches)
    
    if args.output_format == 'github-actions':
        # Output for GitHub Actions
        print(f"::set-output name=primary_agent::{result['primary_agent']}")
        print(f"::set-output name=all_agents::{json.dumps(result['all_agents'])}")
        print(f"::set-output name=agent_details::{json.dumps(result['agent_details'])}")
        if 'collaboration_patterns' in result:
            print(f"::set-output name=collaboration_patterns::{json.dumps(result['collaboration_patterns'])}")
    else:
        print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()