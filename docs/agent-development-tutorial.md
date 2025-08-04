# Agent Development Tutorial

Learn how to create your own specialized Claude Nexus agents from scratch. This comprehensive tutorial will guide you through designing, implementing, and validating a new agent that meets our 75%+ specialization score requirement.

## Table of Contents

- [Tutorial Overview](#tutorial-overview)
- [Prerequisites](#prerequisites)
- [Step 1: Agent Design](#step-1-agent-design)
- [Step 2: Kitten Photography](#step-2-kitten-photography)
- [Step 3: Implementation](#step-3-implementation)
- [Step 4: Testing & Validation](#step-4-testing--validation)
- [Step 5: Documentation](#step-5-documentation)
- [Step 6: Community Submission](#step-6-community-submission)
- [Advanced Techniques](#advanced-techniques)
- [Troubleshooting](#troubleshooting)

## Tutorial Overview

In this tutorial, we'll create a **"Database Optimizer"** agent - a specialized expert in database performance tuning and query optimization. This will demonstrate the complete agent development lifecycle.

### What You'll Learn

- ✅ Agent architecture patterns and design principles
- ✅ Professional kitten photography integration
- ✅ Specialization scoring and validation techniques
- ✅ Enterprise integration patterns
- ✅ Community contribution workflow

### Time Investment

- **Beginner**: 4-6 hours
- **Intermediate**: 2-3 hours  
- **Expert**: 1-2 hours

## Prerequisites

### Required Skills

- **Programming**: JavaScript/Python (intermediate level)
- **Domain Knowledge**: Database systems and SQL optimization
- **Documentation**: Technical writing experience
- **AI Prompting**: Experience with LLM interactions

### Required Tools

```bash
# Development Environment
- Node.js 18+ or Python 3.9+
- Git with SSH keys configured
- Code editor (VS Code recommended)
- Claude API access

# Optional but Recommended
- Docker for testing
- Postman for API testing
- Database setup (PostgreSQL/MySQL)
```

### Setup Verification

```bash
# Clone and setup Claude Nexus
git clone https://github.com/adrianwedd/claude-nexus.git
cd claude-nexus
npm install  # or pip install -r requirements.txt

# Verify validation framework
python agent_validation_framework.py --help

# Test development environment
npm run test:dev  # or python -m pytest tests/
```

## Step 1: Agent Design

### 1.1 Domain Analysis

First, analyze the database optimization domain:

```javascript
// Domain Analysis Worksheet
const domainAnalysis = {
  primaryDomain: "Integration & Data",
  specialization: "Database Performance Optimization", 
  
  // Identify unique capabilities
  coreCapabilities: [
    "Query performance analysis",
    "Index optimization strategies", 
    "Database schema refinement",
    "Caching strategy implementation",
    "Connection pool optimization"
  ],
  
  // Define success metrics
  successMetrics: [
    "Query response time reduction (>50%)",
    "Database resource utilization optimization", 
    "Scalability improvement for concurrent users",
    "Data integrity maintenance"
  ],
  
  // Identify integration points
  integrations: [
    "PostgreSQL, MySQL, MongoDB",
    "Redis caching systems",
    "Application performance monitoring",
    "Database migration tools"
  ]
}
```

### 1.2 Agent Architecture Design

```javascript
// Database Optimizer Agent Architecture
const DatabaseOptimizer = {
  // Core Identity (5 points toward specialization)
  identity: {
    name: "Database Optimizer",
    type: "database-optimizer", 
    domain: "Integration & Data",
    specialization: "Database Performance Optimization",
    kitten_breed: "Scottish Fold" // Thoughtful, methodical breed
  },
  
  // Capabilities Definition (15 points toward specialization)
  capabilities: {
    primary: [
      "Slow query identification and optimization",
      "Index strategy design and implementation", 
      "Database schema analysis and refinement",
      "Caching layer architecture and optimization"
    ],
    secondary: [
      "Database migration planning",
      "Performance monitoring setup",
      "Connection pooling optimization", 
      "Data archiving strategies"
    ],
    integrations: [
      "PostgreSQL", "MySQL", "MongoDB",
      "Redis", "Memcached", 
      "New Relic", "DataDog",
      "Flyway", "Liquibase"
    ]
  },
  
  // Signature Methodology (20 points toward specialization) 
  methodology: {
    approach: "Performance-First Database Architecture",
    principles: [
      "Query-centric optimization",
      "Data-driven index strategies", 
      "Proactive performance monitoring",
      "Scalability-first design patterns"
    ],
    patterns: [
      "Query execution plan analysis",
      "Index coverage optimization",
      "Caching layer integration", 
      "Connection pool sizing"
    ],
    metrics: [
      "Query response time reduction",
      "Database CPU/memory optimization",
      "Concurrent user scalability", 
      "Data consistency maintenance"
    ]
  }
}
```

### 1.3 Specialization Score Estimation

```javascript
// Estimate specialization score breakdown
const scoreEstimation = {
  domainExpertise: {
    estimate: 22, // Out of 25
    reasoning: [
      "4 primary capabilities (strong depth)",
      "4 secondary capabilities (breadth)", 
      "6 tool integrations (ecosystem knowledge)",
      "Clear methodology with measurable metrics"
    ]
  },
  
  implementation: {
    estimate: 20, // Out of 25  
    reasoning: [
      "Comprehensive analysis framework",
      "Robust error handling planned",
      "Performance optimization focus",
      "Extensive testing strategy"
    ]
  },
  
  integration: {
    estimate: 19, // Out of 25
    reasoning: [
      "Enterprise database compatibility", 
      "Monitoring system integration",
      "Migration tool compatibility",
      "Caching system integration"
    ]
  },
  
  communityImpact: {
    estimate: 21, // Out of 25
    reasoning: [
      "High developer pain point addressed",
      "Measurable performance improvements",
      "Reusable optimization patterns",
      "Educational query analysis"
    ]
  },
  
  totalEstimate: 82, // Above 75% requirement ✅
  confidence: "High - addresses critical enterprise need"
}
```

## Step 2: Kitten Photography

### 2.1 Breed Selection Rationale

**Scottish Fold** chosen for Database Optimizer:

- **Thoughtful Nature**: Known for careful, methodical behavior
- **Analytical Expression**: Naturally contemplative appearance
- **Professional Appearance**: Distinctive but not distracting
- **Symbolism**: Folded ears suggest "listening" to database performance

### 2.2 LLM Generation Prompt

```text
Professional studio photograph of Scottish Fold kitten as the Database Optimizer, an elite database performance specialist focused on query optimization and scalability. High-tech database operations center environment with multiple monitors displaying database performance dashboards, query execution plans, index optimization charts, database schema diagrams, and performance monitoring graphs. Professional database administrator styling with subtle data visualization elements and performance optimization tools. Intensely focused, analytical expression suggesting deep understanding of database performance patterns and optimization strategies. Technical lighting with subtle blue data-themed accents emphasizing database expertise and performance focus. Seated thoughtfully at database monitoring station with paws positioned as if analyzing query performance metrics with methodical precision. Professional photography, high resolution, studio lighting, sharp focus, business professional quality suitable for technical documentation.
```

### 2.3 Image Requirements Checklist

```markdown
✅ Resolution: 1200x1200 pixels minimum
✅ Format: PNG with transparency support
✅ Lighting: Professional studio setup
✅ Composition: Business professional suitable
✅ Breed accuracy: Scottish Fold characteristics
✅ Environment: Database/performance themed
✅ Expression: Thoughtful, analytical
✅ Quality: High-resolution, sharp focus
```

## Step 3: Implementation

### 3.1 Core Implementation Structure

```javascript
// database-optimizer.js
const DatabaseOptimizer = {
  // Agent metadata
  metadata: {
    name: "Database Optimizer",
    type: "database-optimizer",
    version: "1.0.0",
    domain: "Integration & Data",
    specialization: "Database Performance Optimization",
    
    // Kitten branding
    kitten: {
      breed: "Scottish Fold",
      image: "images/Database_Optimizer.png",
      prompt: "Professional studio photograph of Scottish Fold kitten..."
    }
  },
  
  // Core capabilities implementation
  capabilities: {
    // Primary capability: Slow query analysis
    analyzeSlowQueries: async function(connectionConfig, options = {}) {
      try {
        const analyzer = new QueryAnalyzer(connectionConfig);
        
        // 1. Identify slow queries
        const slowQueries = await analyzer.identifySlowQueries({
          threshold: options.slowThreshold || 1000, // ms
          timeWindow: options.timeWindow || '24h',
          limit: options.limit || 50
        });
        
        // 2. Analyze execution plans  
        const executionPlans = await Promise.all(
          slowQueries.map(query => analyzer.explainQuery(query))
        );
        
        // 3. Generate optimization recommendations
        const recommendations = this._generateOptimizations(
          slowQueries, 
          executionPlans
        );
        
        return {
          summary: `Analyzed ${slowQueries.length} slow queries`,
          slowQueries: slowQueries,
          executionPlans: executionPlans,
          recommendations: recommendations,
          estimatedImprovement: this._calculateImprovement(recommendations)
        };
        
      } catch (error) {
        throw new DatabaseOptimizerError(
          `Query analysis failed: ${error.message}`,
          'QUERY_ANALYSIS_ERROR'
        );
      }
    },
    
    // Primary capability: Index optimization
    optimizeIndexes: async function(connectionConfig, tableNames = []) {
      try {
        const indexAnalyzer = new IndexAnalyzer(connectionConfig);
        
        // 1. Analyze current index usage
        const indexUsage = await indexAnalyzer.analyzeIndexUsage(tableNames);
        
        // 2. Identify missing indexes
        const missingIndexes = await indexAnalyzer.suggestMissingIndexes(
          tableNames
        );
        
        // 3. Identify redundant indexes
        const redundantIndexes = await indexAnalyzer.findRedundantIndexes(
          tableNames
        );
        
        // 4. Generate index strategy
        const strategy = this._generateIndexStrategy({
          current: indexUsage,
          missing: missingIndexes, 
          redundant: redundantIndexes
        });
        
        return {
          summary: `Analyzed indexes for ${tableNames.length} tables`,
          currentIndexes: indexUsage,
          recommendations: {
            create: missingIndexes,
            drop: redundantIndexes,
            modify: strategy.modifications
          },
          estimatedSpeedup: strategy.estimatedSpeedup,
          implementation: strategy.implementation
        };
        
      } catch (error) {
        throw new DatabaseOptimizerError(
          `Index optimization failed: ${error.message}`,
          'INDEX_OPTIMIZATION_ERROR'
        );
      }
    },
    
    // Primary capability: Caching strategy
    designCachingStrategy: async function(connectionConfig, workloadProfile) {
      try {
        const cacheAnalyzer = new CacheAnalyzer(connectionConfig);
        
        // 1. Analyze query patterns
        const queryPatterns = await cacheAnalyzer.analyzeQueryPatterns(
          workloadProfile
        );
        
        // 2. Identify cacheable queries
        const cacheableQueries = this._identifyCacheableQueries(
          queryPatterns
        );
        
        // 3. Design cache layers
        const cacheStrategy = this._designCacheLayers({
          patterns: queryPatterns,
          cacheable: cacheableQueries,
          workload: workloadProfile
        });
        
        // 4. Generate implementation plan
        const implementation = this._generateCacheImplementation(
          cacheStrategy
        );
        
        return {
          summary: `Designed ${cacheStrategy.layers.length}-layer caching strategy`,
          strategy: cacheStrategy,
          implementation: implementation,
          estimatedHitRate: cacheStrategy.estimatedHitRate,
          performanceGain: cacheStrategy.estimatedPerformanceGain
        };
        
      } catch (error) {
        throw new DatabaseOptimizerError(
          `Cache strategy design failed: ${error.message}`,
          'CACHE_STRATEGY_ERROR'
        );
      }
    }
  },
  
  // Helper methods
  _generateOptimizations: function(queries, plans) {
    return queries.map((query, index) => {
      const plan = plans[index];
      const recommendations = [];
      
      // Analyze execution plan for optimization opportunities
      if (plan.hasSequentialScan) {
        recommendations.push({
          type: 'INDEX_MISSING',
          priority: 'HIGH',
          description: 'Sequential scan detected - missing index',
          solution: this._suggestIndex(query, plan)
        });
      }
      
      if (plan.hasNestedLoop && plan.rowCount > 1000) {
        recommendations.push({
          type: 'JOIN_OPTIMIZATION', 
          priority: 'MEDIUM',
          description: 'Inefficient nested loop join',
          solution: this._optimizeJoin(query, plan)
        });
      }
      
      return {
        query: query,
        recommendations: recommendations,
        estimatedImprovement: this._estimateImprovement(recommendations)
      };
    });
  },
  
  // Performance estimation methods
  _calculateImprovement: function(recommendations) {
    let totalImprovement = 0;
    
    recommendations.forEach(rec => {
      rec.recommendations.forEach(r => {
        switch(r.type) {
          case 'INDEX_MISSING':
            totalImprovement += 0.60; // 60% improvement estimate
            break;
          case 'JOIN_OPTIMIZATION':
            totalImprovement += 0.35; // 35% improvement estimate
            break;
          case 'QUERY_REWRITE':
            totalImprovement += 0.25; // 25% improvement estimate
            break;
        }
      });
    });
    
    return Math.min(totalImprovement, 0.95); // Cap at 95% improvement
  },
  
  // Error handling
  _handleError: function(error, context) {
    const enhancedError = new DatabaseOptimizerError(
      `${context}: ${error.message}`,
      error.code || 'UNKNOWN_ERROR'
    );
    
    // Log for debugging
    console.error(`Database Optimizer Error [${context}]:`, error);
    
    throw enhancedError;
  }
};

// Custom error class
class DatabaseOptimizerError extends Error {
  constructor(message, code) {
    super(message);
    this.name = 'DatabaseOptimizerError';
    this.code = code;
  }
}

// Export agent
module.exports = DatabaseOptimizer;
```

### 3.2 Validation Integration

```javascript
// Add validation support
DatabaseOptimizer.validate = async function() {
  const validator = require('./agent_validation_framework');
  
  const config = {
    name: this.metadata.name,
    type: this.metadata.type,
    domain: this.metadata.domain,
    specialization: this.metadata.specialization,
    capabilities: {
      primary: Object.keys(this.capabilities),
      integrations: ["PostgreSQL", "MySQL", "Redis", "MongoDB"]
    },
    methodology: {
      approach: "Performance-First Database Architecture",
      principles: [
        "Query-centric optimization",
        "Data-driven index strategies",
        "Proactive performance monitoring"
      ]
    },
    kitten_image: this.metadata.kitten.image,
    llm_generation_prompt: this.metadata.kitten.prompt,
    usage_example: `
      Task({
        subagent_type: "database-optimizer",
        description: "Optimize slow e-commerce queries",
        prompt: "Analyze and optimize database queries causing checkout delays"
      })
    `
  };
  
  return await validator.validate(__filename, config);
};
```

## Step 4: Testing & Validation

### 4.1 Unit Tests

```javascript
// tests/database-optimizer.test.js
const DatabaseOptimizer = require('../database-optimizer');
const { expect } = require('chai');

describe('Database Optimizer Agent', () => {
  describe('Query Analysis', () => {
    it('should identify slow queries correctly', async () => {
      const mockConnection = createMockConnection();
      
      const result = await DatabaseOptimizer.capabilities.analyzeSlowQueries(
        mockConnection,
        { slowThreshold: 500, limit: 10 }
      );
      
      expect(result.slowQueries).to.be.an('array');
      expect(result.recommendations).to.be.an('array');
      expect(result.estimatedImprovement).to.be.a('number');
      expect(result.estimatedImprovement).to.be.at.least(0);
    });
    
    it('should handle database connection errors gracefully', async () => {
      const invalidConnection = { host: 'invalid' };
      
      try {
        await DatabaseOptimizer.capabilities.analyzeSlowQueries(invalidConnection);
        expect.fail('Should have thrown an error');
      } catch (error) {
        expect(error.name).to.equal('DatabaseOptimizerError');
        expect(error.code).to.equal('QUERY_ANALYSIS_ERROR');
      }
    });
  });
  
  describe('Index Optimization', () => {
    it('should suggest appropriate indexes', async () => {
      const mockConnection = createMockConnection();
      const tableNames = ['users', 'orders', 'products'];
      
      const result = await DatabaseOptimizer.capabilities.optimizeIndexes(
        mockConnection,
        tableNames
      );
      
      expect(result.recommendations.create).to.be.an('array');
      expect(result.recommendations.drop).to.be.an('array');
      expect(result.estimatedSpeedup).to.be.a('number');
    });
  });
  
  describe('Validation', () => {
    it('should pass specialization score requirements', async () => {
      const validation = await DatabaseOptimizer.validate();
      
      expect(validation.specialization.total_score).to.be.at.least(75);
      expect(validation.status).to.equal('passed');
    });
    
    it('should have valid kitten image', async () => {
      const validation = await DatabaseOptimizer.validate();
      
      expect(validation.kitten_image_valid).to.be.true;
    });
  });
});

function createMockConnection() {
  return {
    host: 'localhost',
    database: 'test_db',
    // Mock connection implementation
  };
}
```

### 4.2 Integration Tests

```javascript
// tests/integration/database-optimizer-integration.test.js
const DatabaseOptimizer = require('../../database-optimizer');
const { setupTestDatabase, teardownTestDatabase } = require('../helpers/db-setup');

describe('Database Optimizer Integration Tests', () => {
  let testDb;
  
  beforeEach(async () => {
    testDb = await setupTestDatabase();
  });
  
  afterEach(async () => {
    await teardownTestDatabase(testDb);
  });
  
  it('should optimize real database queries', async () => {
    // Create test data with known performance issues
    await testDb.query(`
      CREATE TABLE test_orders (
        id SERIAL PRIMARY KEY,
        user_id INTEGER,
        product_id INTEGER, 
        created_at TIMESTAMP DEFAULT NOW()
      )
    `);
    
    // Insert test data
    await insertTestData(testDb, 10000);
    
    // Run optimization
    const result = await DatabaseOptimizer.capabilities.analyzeSlowQueries(
      testDb.config,
      { slowThreshold: 100 }
    );
    
    // Verify results
    expect(result.slowQueries.length).to.be.greaterThan(0);
    expect(result.recommendations.length).to.be.greaterThan(0);
    
    // Test improvement estimation
    expect(result.estimatedImprovement).to.be.at.least(0.3); // 30% improvement
  });
});
```

### 4.3 Performance Benchmarks

```javascript
// tests/performance/database-optimizer-benchmarks.js
const DatabaseOptimizer = require('../../database-optimizer');
const { performance } = require('perf_hooks');

describe('Database Optimizer Performance Benchmarks', () => {
  it('should analyze 100 queries in under 5 seconds', async () => {
    const startTime = performance.now();
    
    const mockQueries = generateMockQueries(100);
    const result = await DatabaseOptimizer.capabilities.analyzeSlowQueries(
      mockConnection,
      { limit: 100 }
    );
    
    const endTime = performance.now();
    const duration = endTime - startTime;
    
    expect(duration).to.be.lessThan(5000); // 5 seconds
    expect(result.slowQueries.length).to.equal(100);
  });
  
  it('should maintain accuracy under load', async () => {
    const concurrentRequests = 10;
    const promises = [];
    
    for (let i = 0; i < concurrentRequests; i++) {
      promises.push(
        DatabaseOptimizer.capabilities.analyzeSlowQueries(mockConnection)
      );
    }
    
    const results = await Promise.all(promises);
    
    // All requests should succeed
    expect(results).to.have.length(concurrentRequests);
    results.forEach(result => {
      expect(result.recommendations).to.be.an('array');
    });
  });
});
```

### 4.4 Automated Validation

```bash
# Run complete validation suite
python agent_validation_framework.py database-optimizer.js database-optimizer-config.json

# Expected output:
# {
#   "validation_summary": {
#     "agent_name": "Database Optimizer",
#     "status": "passed",
#     "overall_score": 82.0,
#     "passed_minimum": true
#   },
#   "specialization_breakdown": {
#     "domain_expertise": 22.0,
#     "implementation": 20.0, 
#     "integration": 19.0,
#     "community_impact": 21.0,
#     "total_score": 82.0
#   }
# }
```

## Step 5: Documentation

### 5.1 Agent Profile Documentation

```markdown
## 🗄️ Database Optimizer

<div align="center">
<picture>
  <source media="(max-width: 768px)" srcset="images/Database_Optimizer.png" width="150">
  <img src="images/Database_Optimizer.png" alt="Database Optimizer - Elite database performance specialist" width="200" style="border-radius: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); max-width: 100%; height: auto;">
</picture>
</div>

**Elite database performance specialist focused on query optimization and scalability**

The Database Optimizer transforms sluggish databases into lightning-fast data engines through scientific analysis and methodical optimization. Master of slow query identification, index strategy design, caching architecture, and performance-first database patterns with data-driven precision.

#### 🎨 LLM Photo Generation Prompt
```
Professional studio photograph of Scottish Fold kitten as the Database Optimizer, an elite database performance specialist focused on query optimization and scalability. High-tech database operations center environment with multiple monitors displaying database performance dashboards, query execution plans, index optimization charts, database schema diagrams, and performance monitoring graphs. Professional database administrator styling with subtle data visualization elements and performance optimization tools. Intensely focused, analytical expression suggesting deep understanding of database performance patterns and optimization strategies. Technical lighting with subtle blue data-themed accents emphasizing database expertise and performance focus. Seated thoughtfully at database monitoring station with paws positioned as if analyzing query performance metrics with methodical precision. Professional photography, high resolution, studio lighting, sharp focus, business professional quality suitable for technical documentation.
```

#### 💼 Usage Example
```javascript
Task({
  subagent_type: "database-optimizer",
  description: "E-commerce database performance optimization",
  prompt: "Analyze slow checkout queries causing 8-second page loads and optimize for sub-1-second response times with 1000+ concurrent users"
})
```

**Signature Methodology**: Performance-First Database Architecture with Query-Centric Optimization

**Specialization Score**: 82/100 ⭐⭐⭐⭐⭐
```

### 5.2 API Documentation

```markdown
# Database Optimizer API Reference

## Core Methods

### analyzeSlowQueries(connectionConfig, options)

Identifies and analyzes slow-performing database queries.

**Parameters:**
- `connectionConfig` (Object): Database connection configuration
  - `host` (String): Database host
  - `port` (Number): Database port
  - `database` (String): Database name
  - `username` (String): Database username
  - `password` (String): Database password
- `options` (Object, optional): Analysis options
  - `slowThreshold` (Number): Minimum query time in ms (default: 1000)
  - `timeWindow` (String): Analysis time window (default: '24h')
  - `limit` (Number): Maximum queries to analyze (default: 50)

**Returns:**
```javascript
{
  summary: String,              // Analysis summary
  slowQueries: Array,           // Slow query objects
  executionPlans: Array,        // Query execution plans
  recommendations: Array,       // Optimization recommendations  
  estimatedImprovement: Number  // Expected performance gain (0-1)
}
```

**Example:**
```javascript
const result = await DatabaseOptimizer.capabilities.analyzeSlowQueries({
  host: 'localhost',
  database: 'ecommerce',
  username: 'admin',
  password: 'password'
}, {
  slowThreshold: 500,
  limit: 25
});

console.log(`Found ${result.slowQueries.length} slow queries`);
console.log(`Estimated ${(result.estimatedImprovement * 100)}% improvement`);
```
```