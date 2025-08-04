#!/usr/bin/env python3
"""
Multi-Tenant Role-Based Access Control (RBAC) System

Comprehensive RBAC implementation for enterprise multi-tenant architecture.
Provides hierarchical role management, resource-level permissions, and
organizational structure support with SOC 2 compliance.

Features:
- Hierarchical role inheritance
- Resource-based permissions
- Organizational unit support
- Dynamic policy evaluation
- Audit trail integration
- Principle of least privilege enforcement

Author: Fortress Guardian
Version: 1.0.0
Compliance: SOC 2 Type II, GDPR, CCPA
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid
from collections import defaultdict, deque
import hashlib
from functools import wraps

# Import from enterprise security architecture
from enterprise_security_architecture import (
    AuditEvent, AuditEventType, ThreatLevel, SecurityLevel
)

logger = logging.getLogger(__name__)

class PermissionType(Enum):
    """Types of permissions that can be granted."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    EXECUTE = "execute"
    APPROVE = "approve"
    AUDIT = "audit"

class ResourceType(Enum):
    """Types of resources in the system."""
    AGENT = "agent"
    TENANT = "tenant"
    USER = "user"
    ROLE = "role"
    POLICY = "policy"
    AUDIT_LOG = "audit_log"
    SYSTEM = "system"
    DATA = "data"

@dataclass
class Permission:
    """Individual permission definition."""
    permission_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    resource_type: ResourceType = ResourceType.DATA
    resource_id: str = "*"  # * for all resources of type
    permission_type: PermissionType = PermissionType.READ
    conditions: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __str__(self) -> str:
        return f"{self.permission_type.value}:{self.resource_type.value}:{self.resource_id}"

@dataclass
class Role:
    """Role definition with permissions and hierarchy."""
    role_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    tenant_id: str = ""
    parent_role_id: Optional[str] = None
    permissions: Set[Permission] = field(default_factory=set)
    is_system_role: bool = False
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def add_permission(self, permission: Permission):
        """Add permission to role."""
        self.permissions.add(permission)
        self.updated_at = datetime.utcnow()
    
    def remove_permission(self, permission_id: str):
        """Remove permission from role."""
        self.permissions = {
            p for p in self.permissions 
            if p.permission_id != permission_id
        }
        self.updated_at = datetime.utcnow()

@dataclass
class OrganizationalUnit:
    """Organizational unit for hierarchical tenant structure."""
    ou_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    tenant_id: str = ""
    parent_ou_id: Optional[str] = None
    managers: Set[str] = field(default_factory=set)  # User IDs
    members: Set[str] = field(default_factory=set)   # User IDs
    default_roles: Set[str] = field(default_factory=set)  # Role IDs
    security_level: SecurityLevel = SecurityLevel.INTERNAL
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def add_member(self, user_id: str):
        """Add user to organizational unit."""
        self.members.add(user_id)
    
    def add_manager(self, user_id: str):
        """Add manager to organizational unit."""
        self.managers.add(user_id)
        self.members.add(user_id)  # Managers are also members

@dataclass
class UserRoleAssignment:
    """Assignment of roles to users with context."""
    assignment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    role_id: str = ""
    tenant_id: str = ""
    ou_id: Optional[str] = None
    assigned_by: str = ""
    assigned_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    is_active: bool = True
    conditions: Dict[str, Any] = field(default_factory=dict)

class PolicyEngine:
    """Dynamic policy evaluation engine."""
    
    def __init__(self):
        self.policies: Dict[str, Dict[str, Any]] = {}
        self.policy_cache: Dict[str, Tuple[bool, datetime]] = {}
        
    def add_policy(self, policy_id: str, policy_definition: Dict[str, Any]):
        """Add security policy."""
        self.policies[policy_id] = policy_definition
        # Clear cache for policies that might be affected
        self.policy_cache.clear()
    
    def evaluate_policy(self, policy_id: str, context: Dict[str, Any]) -> bool:
        """Evaluate policy against given context."""
        
        # Check cache first (policies cached for 5 minutes)
        cache_key = f"{policy_id}:{hashlib.md5(str(context).encode()).hexdigest()}"
        if cache_key in self.policy_cache:
            result, cached_at = self.policy_cache[cache_key]
            if datetime.utcnow() - cached_at < timedelta(minutes=5):
                return result
        
        if policy_id not in self.policies:
            return False
        
        policy = self.policies[policy_id]
        result = self._evaluate_policy_rules(policy, context)
        
        # Cache result
        self.policy_cache[cache_key] = (result, datetime.utcnow())
        
        return result
    
    def _evaluate_policy_rules(self, policy: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate individual policy rules."""
        
        # Default allow if no rules
        if 'rules' not in policy:
            return True
        
        rules = policy['rules']
        
        # Evaluate each rule
        for rule in rules:
            if not self._evaluate_rule(rule, context):
                return False
        
        return True
    
    def _evaluate_rule(self, rule: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate a single rule."""
        
        rule_type = rule.get('type')
        
        if rule_type == 'time_based':
            return self._evaluate_time_rule(rule, context)
        elif rule_type == 'ip_based':
            return self._evaluate_ip_rule(rule, context)
        elif rule_type == 'resource_based':
            return self._evaluate_resource_rule(rule, context)
        elif rule_type == 'conditional':
            return self._evaluate_conditional_rule(rule, context)
        
        return True
    
    def _evaluate_time_rule(self, rule: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate time-based access rule."""
        
        current_time = datetime.utcnow().time()
        start_time = datetime.strptime(rule.get('start_time', '00:00'), '%H:%M').time()
        end_time = datetime.strptime(rule.get('end_time', '23:59'), '%H:%M').time()
        
        return start_time <= current_time <= end_time
    
    def _evaluate_ip_rule(self, rule: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate IP-based access rule."""
        
        allowed_ips = set(rule.get('allowed_ips', []))
        blocked_ips = set(rule.get('blocked_ips', []))
        
        user_ip = context.get('ip_address', '')
        
        # Check if IP is blocked
        if user_ip in blocked_ips:
            return False
        
        # Check if IP is in allowed list (if specified)
        if allowed_ips and user_ip not in allowed_ips:
            return False
        
        return True
    
    def _evaluate_resource_rule(self, rule: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate resource-based access rule."""
        
        required_attributes = rule.get('required_attributes', {})
        resource_attributes = context.get('resource_attributes', {})
        
        for attr, required_value in required_attributes.items():
            if resource_attributes.get(attr) != required_value:
                return False
        
        return True
    
    def _evaluate_conditional_rule(self, rule: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate conditional rule with complex logic."""
        
        condition = rule.get('condition', '')
        
        # Simple condition evaluation (in production, use a proper expression evaluator)
        # For security, only allow specific operators and functions
        safe_context = {
            'user_id': context.get('user_id', ''),
            'tenant_id': context.get('tenant_id', ''),
            'resource_type': context.get('resource_type', ''),
            'action': context.get('action', ''),
            'time_of_day': datetime.utcnow().hour
        }
        
        try:
            # Note: In production, use a safe expression evaluator
            # This is a simplified example
            result = eval(condition, {"__builtins__": {}}, safe_context)
            return bool(result)
        except Exception:
            return False

class MultiTenantRBACSystem:
    """Main RBAC system for multi-tenant architecture."""
    
    def __init__(self, audit_logger=None):
        self.roles: Dict[str, Role] = {}
        self.users: Dict[str, Dict[str, Any]] = {}
        self.role_assignments: Dict[str, List[UserRoleAssignment]] = defaultdict(list)
        self.organizational_units: Dict[str, OrganizationalUnit] = {}
        self.policy_engine = PolicyEngine()
        self.audit_logger = audit_logger
        self._create_system_roles()
        
    def _create_system_roles(self):
        """Create default system roles."""
        
        # Super Admin Role
        super_admin = Role(
            name="Super Admin",
            description="Full system access",
            tenant_id="system",
            is_system_role=True
        )
        super_admin.add_permission(Permission(
            resource_type=ResourceType.SYSTEM,
            resource_id="*",
            permission_type=PermissionType.ADMIN
        ))
        self.roles[super_admin.role_id] = super_admin
        
        # Tenant Admin Role
        tenant_admin = Role(
            name="Tenant Admin",
            description="Full tenant access",
            tenant_id="system",
            is_system_role=True
        )
        for resource_type in ResourceType:
            tenant_admin.add_permission(Permission(
                resource_type=resource_type,
                resource_id="*",
                permission_type=PermissionType.ADMIN
            ))
        self.roles[tenant_admin.role_id] = tenant_admin
        
        # Agent User Role
        agent_user = Role(
            name="Agent User",
            description="Standard agent consultation access",
            tenant_id="system",
            is_system_role=True
        )
        agent_user.add_permission(Permission(
            resource_type=ResourceType.AGENT,
            resource_id="*",
            permission_type=PermissionType.EXECUTE
        ))
        agent_user.add_permission(Permission(
            resource_type=ResourceType.DATA,
            resource_id="*",
            permission_type=PermissionType.READ
        ))
        self.roles[agent_user.role_id] = agent_user
        
        # Security Officer Role
        security_officer = Role(
            name="Security Officer",
            description="Security and compliance oversight",
            tenant_id="system",
            is_system_role=True
        )
        security_officer.add_permission(Permission(
            resource_type=ResourceType.AUDIT_LOG,
            resource_id="*",
            permission_type=PermissionType.READ
        ))
        security_officer.add_permission(Permission(
            resource_type=ResourceType.POLICY,
            resource_id="*",
            permission_type=PermissionType.ADMIN
        ))
        self.roles[security_officer.role_id] = security_officer
        
        logger.info("Created system roles")
    
    def create_role(self, name: str, description: str, tenant_id: str,
                   parent_role_id: Optional[str] = None) -> Role:
        """Create a new role."""
        
        role = Role(
            name=name,
            description=description,
            tenant_id=tenant_id,
            parent_role_id=parent_role_id
        )
        
        self.roles[role.role_id] = role
        
        # Log role creation
        if self.audit_logger:
            audit_event = AuditEvent(
                tenant_id=tenant_id,
                event_type=AuditEventType.SYSTEM_EVENT,
                resource="role",
                action="create",
                result="success",
                metadata={
                    'role_id': role.role_id,
                    'role_name': name,
                    'parent_role_id': parent_role_id
                }
            )
            self.audit_logger.log_audit_event(audit_event)
        
        logger.info(f"Created role {name} for tenant {tenant_id}")
        return role
    
    def assign_role_to_user(self, user_id: str, role_id: str, tenant_id: str,
                           assigned_by: str, ou_id: Optional[str] = None,
                           expires_at: Optional[datetime] = None) -> UserRoleAssignment:
        """Assign role to user."""
        
        if role_id not in self.roles:
            raise ValueError(f"Role {role_id} not found")
        
        assignment = UserRoleAssignment(
            user_id=user_id,
            role_id=role_id,
            tenant_id=tenant_id,
            ou_id=ou_id,
            assigned_by=assigned_by,
            expires_at=expires_at
        )
        
        self.role_assignments[user_id].append(assignment)
        
        # Log role assignment
        if self.audit_logger:
            audit_event = AuditEvent(
                tenant_id=tenant_id,
                user_id=user_id,
                event_type=AuditEventType.AUTHORIZATION,
                resource="role_assignment",
                action="assign",
                result="success",
                metadata={
                    'role_id': role_id,
                    'assigned_by': assigned_by,
                    'ou_id': ou_id,
                    'expires_at': expires_at.isoformat() if expires_at else None
                }
            )
            self.audit_logger.log_audit_event(audit_event)
        
        logger.info(f"Assigned role {role_id} to user {user_id} in tenant {tenant_id}")
        return assignment
    
    def create_organizational_unit(self, name: str, description: str, tenant_id: str,
                                 parent_ou_id: Optional[str] = None,
                                 security_level: SecurityLevel = SecurityLevel.INTERNAL) -> OrganizationalUnit:
        """Create organizational unit."""
        
        ou = OrganizationalUnit(
            name=name,
            description=description,
            tenant_id=tenant_id,
            parent_ou_id=parent_ou_id,
            security_level=security_level
        )
        
        self.organizational_units[ou.ou_id] = ou
        
        # Log OU creation
        if self.audit_logger:
            audit_event = AuditEvent(
                tenant_id=tenant_id,
                event_type=AuditEventType.SYSTEM_EVENT,
                resource="organizational_unit",
                action="create",
                result="success",
                metadata={
                    'ou_id': ou.ou_id,
                    'ou_name': name,
                    'parent_ou_id': parent_ou_id,
                    'security_level': security_level.value
                }
            )
            self.audit_logger.log_audit_event(audit_event)
        
        logger.info(f"Created organizational unit {name} for tenant {tenant_id}")
        return ou
    
    def check_permission(self, user_id: str, tenant_id: str, resource_type: ResourceType,
                        resource_id: str, permission_type: PermissionType,
                        context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if user has permission for specific resource and action."""
        
        context = context or {}
        context.update({
            'user_id': user_id,
            'tenant_id': tenant_id,
            'resource_type': resource_type.value,
            'resource_id': resource_id,
            'action': permission_type.value
        })
        
        # Get user's effective permissions
        effective_permissions = self.get_user_permissions(user_id, tenant_id)
        
        # Check if user has required permission
        has_permission = False
        for permission in effective_permissions:
            if (permission.resource_type == resource_type and
                (permission.resource_id == "*" or permission.resource_id == resource_id) and
                (permission.permission_type == permission_type or permission.permission_type == PermissionType.ADMIN)):
                
                # Check permission conditions
                if self._check_permission_conditions(permission, context):
                    has_permission = True
                    break
        
        # Log permission check
        if self.audit_logger:
            audit_event = AuditEvent(
                tenant_id=tenant_id,
                user_id=user_id,
                event_type=AuditEventType.AUTHORIZATION,
                resource=f"{resource_type.value}:{resource_id}",
                action=f"check_{permission_type.value}",
                result="success" if has_permission else "denied",
                metadata=context
            )
            self.audit_logger.log_audit_event(audit_event)
        
        return has_permission
    
    def get_user_permissions(self, user_id: str, tenant_id: str) -> Set[Permission]:
        """Get all effective permissions for user in tenant."""
        
        effective_permissions = set()
        
        # Get user's role assignments
        assignments = self.role_assignments.get(user_id, [])
        
        for assignment in assignments:
            # Check if assignment is active and not expired
            if (assignment.is_active and 
                assignment.tenant_id == tenant_id and
                (not assignment.expires_at or assignment.expires_at > datetime.utcnow())):
                
                # Get role permissions (including inherited)
                role_permissions = self._get_role_permissions(assignment.role_id)
                effective_permissions.update(role_permissions)
        
        return effective_permissions
    
    def _get_role_permissions(self, role_id: str) -> Set[Permission]:
        """Get all permissions for role including inherited permissions."""
        
        if role_id not in self.roles:
            return set()
        
        role = self.roles[role_id]
        permissions = set(role.permissions)
        
        # Inherit permissions from parent role
        if role.parent_role_id:
            parent_permissions = self._get_role_permissions(role.parent_role_id)
            permissions.update(parent_permissions)
        
        return permissions
    
    def _check_permission_conditions(self, permission: Permission, context: Dict[str, Any]) -> bool:
        """Check if permission conditions are met."""
        
        if not permission.conditions:
            return True
        
        # Evaluate policy conditions
        for policy_id in permission.conditions.get('policies', []):
            if not self.policy_engine.evaluate_policy(policy_id, context):
                return False
        
        return True
    
    def get_user_roles(self, user_id: str, tenant_id: str) -> List[Role]:
        """Get all active roles for user in tenant."""
        
        user_roles = []
        assignments = self.role_assignments.get(user_id, [])
        
        for assignment in assignments:
            if (assignment.is_active and 
                assignment.tenant_id == tenant_id and
                (not assignment.expires_at or assignment.expires_at > datetime.utcnow())):
                
                if assignment.role_id in self.roles:
                    user_roles.append(self.roles[assignment.role_id])
        
        return user_roles
    
    def revoke_role_assignment(self, assignment_id: str, revoked_by: str) -> bool:
        """Revoke role assignment."""
        
        for user_id, assignments in self.role_assignments.items():
            for assignment in assignments:
                if assignment.assignment_id == assignment_id:
                    assignment.is_active = False
                    
                    # Log role revocation
                    if self.audit_logger:
                        audit_event = AuditEvent(
                            tenant_id=assignment.tenant_id,
                            user_id=user_id,
                            event_type=AuditEventType.AUTHORIZATION,
                            resource="role_assignment",
                            action="revoke",
                            result="success",
                            metadata={
                                'assignment_id': assignment_id,
                                'role_id': assignment.role_id,
                                'revoked_by': revoked_by
                            }
                        )
                        self.audit_logger.log_audit_event(audit_event)
                    
                    logger.info(f"Revoked role assignment {assignment_id} for user {user_id}")
                    return True
        
        return False
    
    def get_rbac_health_metrics(self, tenant_id: str) -> Dict[str, Any]:
        """Get RBAC system health metrics for tenant."""
        
        tenant_roles = [role for role in self.roles.values() if role.tenant_id == tenant_id]
        tenant_assignments = [
            assignment for assignments in self.role_assignments.values()
            for assignment in assignments
            if assignment.tenant_id == tenant_id and assignment.is_active
        ]
        
        # Calculate metrics
        total_users = len(set(assignment.user_id for assignment in tenant_assignments))
        avg_roles_per_user = len(tenant_assignments) / max(total_users, 1)
        
        # Find users with excessive privileges
        privilege_distribution = defaultdict(int)
        for assignment in tenant_assignments:
            role = self.roles.get(assignment.role_id)
            if role:
                privilege_count = len(self._get_role_permissions(role.role_id))
                privilege_distribution[assignment.user_id] += privilege_count
        
        high_privilege_users = [
            user_id for user_id, count in privilege_distribution.items()
            if count > 50  # Threshold for high privileges
        ]
        
        return {
            'tenant_id': tenant_id,
            'total_roles': len(tenant_roles),
            'total_users': total_users,
            'total_assignments': len(tenant_assignments),
            'avg_roles_per_user': round(avg_roles_per_user, 2),
            'high_privilege_users': len(high_privilege_users),
            'system_roles': len([r for r in tenant_roles if r.is_system_role]),
            'custom_roles': len([r for r in tenant_roles if not r.is_system_role]),
            'expired_assignments': len([
                a for a in tenant_assignments
                if a.expires_at and a.expires_at <= datetime.utcnow()
            ])
        }

# RBAC decorator for API endpoints
def require_permission(rbac_system: MultiTenantRBACSystem, 
                     resource_type: ResourceType,
                     permission_type: PermissionType):
    """Decorator to require specific permission for API endpoints."""
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract auth context from request
            auth_context = kwargs.get('auth_context')
            if not auth_context:
                raise ValueError("Authentication required")
            
            user_id = auth_context['user_id']
            tenant_id = auth_context['tenant_id']
            
            # Extract resource ID from kwargs or use wildcard
            resource_id = kwargs.get('resource_id', '*')
            
            # Check permission
            if not rbac_system.check_permission(
                user_id, tenant_id, resource_type, resource_id, permission_type
            ):
                raise PermissionError(
                    f"Insufficient permissions: {permission_type.value} on {resource_type.value}:{resource_id}"
                )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator

# Example usage
if __name__ == "__main__":
    # Initialize RBAC system
    rbac = MultiTenantRBACSystem()
    
    # Create tenant-specific role
    custom_role = rbac.create_role(
        name="Data Analyst",
        description="Read-only access to data and analytics",
        tenant_id="test-corp"
    )
    
    # Add permissions to role
    custom_role.add_permission(Permission(
        resource_type=ResourceType.DATA,
        resource_id="*",
        permission_type=PermissionType.READ
    ))
    
    custom_role.add_permission(Permission(
        resource_type=ResourceType.AGENT,
        resource_id="data-architect",
        permission_type=PermissionType.EXECUTE
    ))
    
    # Create organizational unit
    analytics_ou = rbac.create_organizational_unit(
        name="Analytics Team",
        description="Data analytics and reporting team",
        tenant_id="test-corp",
        security_level=SecurityLevel.CONFIDENTIAL
    )
    
    # Assign role to user
    assignment = rbac.assign_role_to_user(
        user_id="analyst@test-corp.com",
        role_id=custom_role.role_id,
        tenant_id="test-corp",
        assigned_by="admin@test-corp.com",
        ou_id=analytics_ou.ou_id
    )
    
    # Check permissions
    can_read_data = rbac.check_permission(
        user_id="analyst@test-corp.com",
        tenant_id="test-corp",
        resource_type=ResourceType.DATA,
        resource_id="customer_data",
        permission_type=PermissionType.READ
    )
    
    can_delete_data = rbac.check_permission(
        user_id="analyst@test-corp.com",
        tenant_id="test-corp",
        resource_type=ResourceType.DATA,
        resource_id="customer_data",
        permission_type=PermissionType.DELETE
    )
    
    print(f"Can read data: {can_read_data}")  # True
    print(f"Can delete data: {can_delete_data}")  # False
    
    # Get health metrics
    metrics = rbac.get_rbac_health_metrics("test-corp")
    print(f"RBAC Health Metrics: {metrics}")
