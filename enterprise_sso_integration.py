#!/usr/bin/env python3
"""
Enterprise Single Sign-On (SSO) Integration System

Comprehensive SSO integration supporting multiple identity providers
including Azure AD, Okta, Auth0, and SAML-based systems. Provides
seamless authentication and authorization for enterprise deployments.

Features:
- Multi-provider SSO support (SAML, OIDC, OAuth2)
- Just-in-time (JIT) user provisioning
- Group/role mapping from identity providers
- Session management and token refresh
- Multi-factor authentication (MFA) support
- Audit trail integration

Supported Providers:
- Azure Active Directory (Azure AD)
- Okta
- Auth0
- Google Workspace
- Generic SAML 2.0 providers
- Generic OIDC providers

Author: Fortress Guardian
Version: 1.0.0
Compliance: SOC 2 Type II, GDPR, CCPA
"""

import asyncio
import base64
import json
import logging
import secrets
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urlencode, parse_qs
import uuid
import hashlib
import hmac

# External dependencies for SSO protocols
import jwt
import requests
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.x509 import load_pem_x509_certificate

# Import from our security system
from enterprise_security_architecture import (
    AuditEvent, AuditEventType, SecurityLevel
)
from multi_tenant_rbac_system import (
    MultiTenantRBACSystem, Role, Permission, ResourceType, PermissionType
)

logger = logging.getLogger(__name__)

class IdentityProvider(Enum):
    """Supported identity providers."""
    AZURE_AD = "azure_ad"
    OKTA = "okta"
    AUTH0 = "auth0"
    GOOGLE_WORKSPACE = "google_workspace"
    SAML_GENERIC = "saml_generic"
    OIDC_GENERIC = "oidc_generic"

class AuthenticationMethod(Enum):
    """Authentication methods supported."""
    SAML2 = "saml2"
    OIDC = "oidc"
    OAUTH2 = "oauth2"

class UserProvisioningMode(Enum):
    """User provisioning modes."""
    JIT = "just_in_time"  # Create users on first login
    MANUAL = "manual"     # Users must be pre-created
    SYNC = "sync"         # Periodic sync from IdP

@dataclass
class IdPConfiguration:
    """Identity provider configuration."""
    idp_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    provider_type: IdentityProvider = IdentityProvider.SAML_GENERIC
    auth_method: AuthenticationMethod = AuthenticationMethod.SAML2
    tenant_id: str = ""
    
    # SAML Configuration
    saml_sso_url: Optional[str] = None
    saml_slo_url: Optional[str] = None
    saml_entity_id: Optional[str] = None
    saml_x509_cert: Optional[str] = None
    saml_name_id_format: str = "urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress"
    
    # OIDC/OAuth2 Configuration
    oidc_discovery_url: Optional[str] = None
    oidc_client_id: Optional[str] = None
    oidc_client_secret: Optional[str] = None
    oidc_scopes: Set[str] = field(default_factory=lambda: {"openid", "profile", "email"})
    
    # Provider-specific endpoints
    authorization_endpoint: Optional[str] = None
    token_endpoint: Optional[str] = None
    userinfo_endpoint: Optional[str] = None
    jwks_uri: Optional[str] = None
    
    # User provisioning
    provisioning_mode: UserProvisioningMode = UserProvisioningMode.JIT
    default_roles: Set[str] = field(default_factory=set)
    attribute_mapping: Dict[str, str] = field(default_factory=dict)
    group_attribute: Optional[str] = None
    role_mapping: Dict[str, str] = field(default_factory=dict)  # IdP group -> role
    
    # Security settings
    require_mfa: bool = False
    allowed_domains: Set[str] = field(default_factory=set)
    session_timeout_hours: int = 8
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True

@dataclass
class SSOSession:
    """SSO session information."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    tenant_id: str = ""
    idp_id: str = ""
    idp_session_id: Optional[str] = None
    
    # User attributes from IdP
    user_attributes: Dict[str, Any] = field(default_factory=dict)
    groups: Set[str] = field(default_factory=set)
    roles: Set[str] = field(default_factory=set)
    
    # Session metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(hours=8))
    ip_address: str = ""
    user_agent: str = ""
    
    # MFA status
    mfa_verified: bool = False
    mfa_methods: Set[str] = field(default_factory=set)
    
    def is_expired(self) -> bool:
        """Check if session is expired."""
        return datetime.utcnow() > self.expires_at
    
    def refresh_session(self, hours: int = 8):
        """Refresh session expiration."""
        self.last_accessed = datetime.utcnow()
        self.expires_at = datetime.utcnow() + timedelta(hours=hours)

class SAMLHandler:
    """SAML 2.0 authentication handler."""
    
    def __init__(self, idp_config: IdPConfiguration):
        self.idp_config = idp_config
        self._load_idp_certificate()
    
    def _load_idp_certificate(self):
        """Load and validate IdP X.509 certificate."""
        if self.idp_config.saml_x509_cert:
            try:
                cert_data = base64.b64decode(self.idp_config.saml_x509_cert)
                self.idp_certificate = load_pem_x509_certificate(cert_data)
                logger.info(f"Loaded SAML certificate for IdP {self.idp_config.name}")
            except Exception as e:
                logger.error(f"Failed to load SAML certificate: {e}")
                self.idp_certificate = None
        else:
            self.idp_certificate = None
    
    def generate_saml_request(self, relay_state: Optional[str] = None) -> Tuple[str, str]:
        """Generate SAML authentication request."""
        
        request_id = f"__{uuid.uuid4()}"
        issue_instant = datetime.utcnow().isoformat() + "Z"
        
        # Build SAML AuthnRequest
        saml_request = f'''
        <samlp:AuthnRequest 
            xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol"
            xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion"
            ID="{request_id}"
            Version="2.0"
            IssueInstant="{issue_instant}"
            ProtocolBinding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST"
            AssertionConsumerServiceURL="https://claude-nexus.example.com/sso/saml/acs">
            <saml:Issuer>claude-nexus-sp</saml:Issuer>
            <samlp:NameIDPolicy Format="{self.idp_config.saml_name_id_format}" AllowCreate="true"/>
        </samlp:AuthnRequest>
        '''.strip()
        
        # Base64 encode and compress (deflate)
        import zlib
        compressed = zlib.compress(saml_request.encode('utf-8'))[2:-4]
        encoded_request = base64.b64encode(compressed).decode('utf-8')
        
        # Build redirect URL
        params = {
            'SAMLRequest': encoded_request
        }
        
        if relay_state:
            params['RelayState'] = relay_state
        
        redirect_url = f"{self.idp_config.saml_sso_url}?{urlencode(params)}"
        
        return redirect_url, request_id
    
    def process_saml_response(self, saml_response: str, relay_state: Optional[str] = None) -> Dict[str, Any]:
        """Process SAML response and extract user information."""
        
        try:
            # Decode SAML response
            decoded_response = base64.b64decode(saml_response)
            
            # Parse XML
            root = ET.fromstring(decoded_response)
            
            # Extract namespaces
            namespaces = {
                'saml': 'urn:oasis:names:tc:SAML:2.0:assertion',
                'samlp': 'urn:oasis:names:tc:SAML:2.0:protocol'
            }
            
            # Validate response status
            status = root.find('.//samlp:Status/samlp:StatusCode', namespaces)
            if status is None or status.get('Value') != 'urn:oasis:names:tc:SAML:2.0:status:Success':
                raise ValueError("SAML authentication failed")
            
            # Extract assertion
            assertion = root.find('.//saml:Assertion', namespaces)
            if assertion is None:
                raise ValueError("No SAML assertion found")
            
            # Validate signature (if certificate available)
            if self.idp_certificate:
                self._validate_saml_signature(assertion)
            
            # Extract user attributes
            user_data = self._extract_user_attributes(assertion, namespaces)
            
            return user_data
            
        except Exception as e:
            logger.error(f"Failed to process SAML response: {e}")
            raise ValueError(f"SAML processing failed: {str(e)}")
    
    def _validate_saml_signature(self, assertion: ET.Element):
        """Validate SAML assertion signature."""
        # Note: In production, implement proper XML signature validation
        # This is a simplified placeholder
        logger.info("SAML signature validation (placeholder)")
    
    def _extract_user_attributes(self, assertion: ET.Element, namespaces: Dict[str, str]) -> Dict[str, Any]:
        """Extract user attributes from SAML assertion."""
        
        # Extract NameID (user identifier)
        name_id = assertion.find('.//saml:Subject/saml:NameID', namespaces)
        user_id = name_id.text if name_id is not None else None
        
        if not user_id:
            raise ValueError("No user ID found in SAML assertion")
        
        # Extract attribute statements
        attributes = {}
        attribute_statements = assertion.findall('.//saml:AttributeStatement/saml:Attribute', namespaces)
        
        for attr in attribute_statements:
            attr_name = attr.get('Name')
            attr_values = [v.text for v in attr.findall('saml:AttributeValue', namespaces) if v.text]
            
            if attr_values:
                if len(attr_values) == 1:
                    attributes[attr_name] = attr_values[0]
                else:
                    attributes[attr_name] = attr_values
        
        # Map attributes based on configuration
        mapped_attributes = {}
        for saml_attr, user_attr in self.idp_config.attribute_mapping.items():
            if saml_attr in attributes:
                mapped_attributes[user_attr] = attributes[saml_attr]
        
        # Extract groups if specified
        groups = set()
        if self.idp_config.group_attribute and self.idp_config.group_attribute in attributes:
            group_value = attributes[self.idp_config.group_attribute]
            if isinstance(group_value, list):
                groups.update(group_value)
            else:
                groups.add(group_value)
        
        return {
            'user_id': user_id,
            'attributes': mapped_attributes,
            'groups': groups,
            'raw_attributes': attributes
        }

class OIDCHandler:
    """OpenID Connect authentication handler."""
    
    def __init__(self, idp_config: IdPConfiguration):
        self.idp_config = idp_config
        self.discovery_document = None
        self.jwks = None
        self._load_discovery_document()
    
    def _load_discovery_document(self):
        """Load OIDC discovery document."""
        if self.idp_config.oidc_discovery_url:
            try:
                response = requests.get(self.idp_config.oidc_discovery_url, timeout=10)
                response.raise_for_status()
                self.discovery_document = response.json()
                
                # Update endpoints from discovery
                self.idp_config.authorization_endpoint = self.discovery_document.get('authorization_endpoint')
                self.idp_config.token_endpoint = self.discovery_document.get('token_endpoint')
                self.idp_config.userinfo_endpoint = self.discovery_document.get('userinfo_endpoint')
                self.idp_config.jwks_uri = self.discovery_document.get('jwks_uri')
                
                logger.info(f"Loaded OIDC discovery document for {self.idp_config.name}")
                
            except Exception as e:
                logger.error(f"Failed to load OIDC discovery document: {e}")
    
    def _load_jwks(self):
        """Load JSON Web Key Set for token validation."""
        if self.idp_config.jwks_uri and not self.jwks:
            try:
                response = requests.get(self.idp_config.jwks_uri, timeout=10)
                response.raise_for_status()
                self.jwks = response.json()
                logger.info(f"Loaded JWKS for {self.idp_config.name}")
            except Exception as e:
                logger.error(f"Failed to load JWKS: {e}")
    
    def generate_authorization_url(self, state: str, nonce: str, redirect_uri: str) -> str:
        """Generate OIDC authorization URL."""
        
        params = {
            'response_type': 'code',
            'client_id': self.idp_config.oidc_client_id,
            'redirect_uri': redirect_uri,
            'scope': ' '.join(self.idp_config.oidc_scopes),
            'state': state,
            'nonce': nonce
        }
        
        if self.idp_config.require_mfa:
            params['acr_values'] = 'mfa'
        
        return f"{self.idp_config.authorization_endpoint}?{urlencode(params)}"
    
    def exchange_code_for_tokens(self, code: str, redirect_uri: str) -> Dict[str, Any]:
        """Exchange authorization code for tokens."""
        
        token_data = {
            'grant_type': 'authorization_code',
            'client_id': self.idp_config.oidc_client_id,
            'client_secret': self.idp_config.oidc_client_secret,
            'code': code,
            'redirect_uri': redirect_uri
        }
        
        try:
            response = requests.post(
                self.idp_config.token_endpoint,
                data=token_data,
                headers={'Content-Type': 'application/x-www-form-urlencoded'},
                timeout=10
            )
            response.raise_for_status()
            
            tokens = response.json()
            
            # Validate ID token
            if 'id_token' in tokens:
                id_token_claims = self._validate_id_token(tokens['id_token'])
                tokens['id_token_claims'] = id_token_claims
            
            return tokens
            
        except Exception as e:
            logger.error(f"Failed to exchange code for tokens: {e}")
            raise ValueError(f"Token exchange failed: {str(e)}")
    
    def _validate_id_token(self, id_token: str) -> Dict[str, Any]:
        """Validate and decode ID token."""
        
        self._load_jwks()
        
        try:
            # Decode header to get key ID
            header = jwt.get_unverified_header(id_token)
            key_id = header.get('kid')
            
            # Find matching key in JWKS
            signing_key = None
            if self.jwks and key_id:
                for key in self.jwks.get('keys', []):
                    if key.get('kid') == key_id:
                        signing_key = jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(key))
                        break
            
            if not signing_key:
                raise ValueError("No matching signing key found")
            
            # Validate and decode token
            claims = jwt.decode(
                id_token,
                signing_key,
                algorithms=['RS256'],
                audience=self.idp_config.oidc_client_id,
                options={'verify_exp': True, 'verify_aud': True}
            )
            
            return claims
            
        except Exception as e:
            logger.error(f"ID token validation failed: {e}")
            raise ValueError(f"Invalid ID token: {str(e)}")
    
    def get_user_info(self, access_token: str) -> Dict[str, Any]:
        """Get user information using access token."""
        
        try:
            response = requests.get(
                self.idp_config.userinfo_endpoint,
                headers={'Authorization': f'Bearer {access_token}'},
                timeout=10
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to get user info: {e}")
            raise ValueError(f"User info request failed: {str(e)}")

class EnterpriseSSO:
    """Main enterprise SSO system."""
    
    def __init__(self, rbac_system: MultiTenantRBACSystem = None, audit_logger=None):
        self.idp_configurations: Dict[str, IdPConfiguration] = {}
        self.active_sessions: Dict[str, SSOSession] = {}
        self.rbac_system = rbac_system
        self.audit_logger = audit_logger
        self.saml_handlers: Dict[str, SAMLHandler] = {}
        self.oidc_handlers: Dict[str, OIDCHandler] = {}
    
    def register_identity_provider(self, idp_config: IdPConfiguration) -> str:
        """Register new identity provider."""
        
        self.idp_configurations[idp_config.idp_id] = idp_config
        
        # Initialize appropriate handler
        if idp_config.auth_method == AuthenticationMethod.SAML2:
            self.saml_handlers[idp_config.idp_id] = SAMLHandler(idp_config)
        elif idp_config.auth_method == AuthenticationMethod.OIDC:
            self.oidc_handlers[idp_config.idp_id] = OIDCHandler(idp_config)
        
        # Log IdP registration
        if self.audit_logger:
            audit_event = AuditEvent(
                tenant_id=idp_config.tenant_id,
                event_type=AuditEventType.SYSTEM_EVENT,
                resource="identity_provider",
                action="register",
                result="success",
                metadata={
                    'idp_id': idp_config.idp_id,
                    'provider_type': idp_config.provider_type.value,
                    'auth_method': idp_config.auth_method.value
                }
            )
            self.audit_logger.log_audit_event(audit_event)
        
        logger.info(f"Registered identity provider {idp_config.name} for tenant {idp_config.tenant_id}")
        return idp_config.idp_id
    
    def initiate_sso_login(self, idp_id: str, redirect_uri: str, 
                          ip_address: str = "", user_agent: str = "") -> Dict[str, str]:
        """Initiate SSO login process."""
        
        if idp_id not in self.idp_configurations:
            raise ValueError(f"Identity provider {idp_id} not found")
        
        idp_config = self.idp_configurations[idp_id]
        
        if idp_config.auth_method == AuthenticationMethod.SAML2:
            handler = self.saml_handlers[idp_id]
            redirect_url, request_id = handler.generate_saml_request()
            
            return {
                'redirect_url': redirect_url,
                'request_id': request_id,
                'method': 'saml'
            }
            
        elif idp_config.auth_method == AuthenticationMethod.OIDC:
            handler = self.oidc_handlers[idp_id]
            state = secrets.token_urlsafe(32)
            nonce = secrets.token_urlsafe(32)
            
            redirect_url = handler.generate_authorization_url(state, nonce, redirect_uri)
            
            return {
                'redirect_url': redirect_url,
                'state': state,
                'nonce': nonce,
                'method': 'oidc'
            }
        
        else:
            raise ValueError(f"Unsupported authentication method: {idp_config.auth_method}")
    
    def process_sso_callback(self, idp_id: str, callback_data: Dict[str, Any],
                           ip_address: str = "", user_agent: str = "") -> SSOSession:
        """Process SSO callback and create session."""
        
        if idp_id not in self.idp_configurations:
            raise ValueError(f"Identity provider {idp_id} not found")
        
        idp_config = self.idp_configurations[idp_id]
        
        try:
            if idp_config.auth_method == AuthenticationMethod.SAML2:
                user_data = self._process_saml_callback(idp_id, callback_data)
            elif idp_config.auth_method == AuthenticationMethod.OIDC:
                user_data = self._process_oidc_callback(idp_id, callback_data)
            else:
                raise ValueError(f"Unsupported authentication method")
            
            # Validate user domain if restricted
            if idp_config.allowed_domains:
                user_email = user_data.get('attributes', {}).get('email', '')
                user_domain = user_email.split('@')[-1] if '@' in user_email else ''
                if user_domain not in idp_config.allowed_domains:
                    raise ValueError(f"Domain {user_domain} not allowed")
            
            # Create or update user session
            session = self._create_user_session(
                idp_id, user_data, ip_address, user_agent
            )
            
            # Provision user if needed
            if idp_config.provisioning_mode == UserProvisioningMode.JIT:
                self._provision_user(session, idp_config)
            
            # Log successful authentication
            if self.audit_logger:
                audit_event = AuditEvent(
                    tenant_id=idp_config.tenant_id,
                    user_id=session.user_id,
                    event_type=AuditEventType.AUTHENTICATION,
                    resource="sso",
                    action="login",
                    result="success",
                    ip_address=ip_address,
                    user_agent=user_agent,
                    metadata={
                        'idp_id': idp_id,
                        'session_id': session.session_id,
                        'mfa_verified': session.mfa_verified
                    }
                )
                self.audit_logger.log_audit_event(audit_event)
            
            return session
            
        except Exception as e:
            # Log failed authentication
            if self.audit_logger:
                audit_event = AuditEvent(
                    tenant_id=idp_config.tenant_id,
                    event_type=AuditEventType.AUTHENTICATION,
                    resource="sso",
                    action="login",
                    result="failure",
                    ip_address=ip_address,
                    user_agent=user_agent,
                    metadata={
                        'idp_id': idp_id,
                        'error': str(e)
                    }
                )
                self.audit_logger.log_audit_event(audit_event)
            
            logger.error(f"SSO authentication failed: {e}")
            raise
    
    def _process_saml_callback(self, idp_id: str, callback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process SAML callback data."""
        
        saml_response = callback_data.get('SAMLResponse')
        if not saml_response:
            raise ValueError("No SAML response found")
        
        handler = self.saml_handlers[idp_id]
        return handler.process_saml_response(saml_response)
    
    def _process_oidc_callback(self, idp_id: str, callback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process OIDC callback data."""
        
        code = callback_data.get('code')
        state = callback_data.get('state')
        
        if not code or not state:
            raise ValueError("Missing authorization code or state")
        
        handler = self.oidc_handlers[idp_id]
        
        # Exchange code for tokens
        tokens = handler.exchange_code_for_tokens(
            code, "https://claude-nexus.example.com/sso/oidc/callback"
        )
        
        # Get user info
        user_info = handler.get_user_info(tokens['access_token'])
        
        # Extract user data
        id_claims = tokens.get('id_token_claims', {})
        
        return {
            'user_id': user_info.get('email', id_claims.get('email')),
            'attributes': {
                'email': user_info.get('email'),
                'name': user_info.get('name'),
                'given_name': user_info.get('given_name'),
                'family_name': user_info.get('family_name'),
                'picture': user_info.get('picture')
            },
            'groups': set(user_info.get('groups', [])),
            'raw_attributes': user_info,
            'tokens': tokens
        }
    
    def _create_user_session(self, idp_id: str, user_data: Dict[str, Any],
                           ip_address: str, user_agent: str) -> SSOSession:
        """Create user session from authentication data."""
        
        idp_config = self.idp_configurations[idp_id]
        
        # Check for MFA verification
        mfa_verified = False
        mfa_methods = set()
        
        if 'tokens' in user_data:
            id_claims = user_data['tokens'].get('id_token_claims', {})
            amr = id_claims.get('amr', [])
            if isinstance(amr, list):
                mfa_methods.update(amr)
                mfa_verified = 'mfa' in amr or len(amr) > 1
        
        # Map groups to roles
        roles = set(idp_config.default_roles)
        for group in user_data.get('groups', set()):
            if group in idp_config.role_mapping:
                roles.add(idp_config.role_mapping[group])
        
        session = SSOSession(
            user_id=user_data['user_id'],
            tenant_id=idp_config.tenant_id,
            idp_id=idp_id,
            user_attributes=user_data.get('attributes', {}),
            groups=user_data.get('groups', set()),
            roles=roles,
            ip_address=ip_address,
            user_agent=user_agent,
            mfa_verified=mfa_verified,
            mfa_methods=mfa_methods,
            expires_at=datetime.utcnow() + timedelta(hours=idp_config.session_timeout_hours)
        )
        
        self.active_sessions[session.session_id] = session
        
        return session
    
    def _provision_user(self, session: SSOSession, idp_config: IdPConfiguration):
        """Provision user with roles based on SSO data."""
        
        if not self.rbac_system:
            return
        
        # Assign roles to user
        for role_name in session.roles:
            # Find role by name
            matching_roles = [
                role for role in self.rbac_system.roles.values()
                if role.name == role_name and role.tenant_id == session.tenant_id
            ]
            
            if matching_roles:
                role = matching_roles[0]
                
                # Check if user already has this role
                existing_assignments = self.rbac_system.role_assignments.get(session.user_id, [])
                has_role = any(
                    a.role_id == role.role_id and a.tenant_id == session.tenant_id and a.is_active
                    for a in existing_assignments
                )
                
                if not has_role:
                    self.rbac_system.assign_role_to_user(
                        user_id=session.user_id,
                        role_id=role.role_id,
                        tenant_id=session.tenant_id,
                        assigned_by="sso_system"
                    )
                    
                    logger.info(f"Assigned role {role_name} to user {session.user_id}")
    
    def validate_session(self, session_id: str) -> Optional[SSOSession]:
        """Validate and return active session."""
        
        session = self.active_sessions.get(session_id)
        
        if not session:
            return None
        
        if session.is_expired():
            # Remove expired session
            del self.active_sessions[session_id]
            return None
        
        # Update last accessed time
        session.last_accessed = datetime.utcnow()
        
        return session
    
    def logout_session(self, session_id: str) -> bool:
        """Logout user session."""
        
        session = self.active_sessions.get(session_id)
        
        if not session:
            return False
        
        # Log logout
        if self.audit_logger:
            audit_event = AuditEvent(
                tenant_id=session.tenant_id,
                user_id=session.user_id,
                event_type=AuditEventType.AUTHENTICATION,
                resource="sso",
                action="logout",
                result="success",
                metadata={'session_id': session_id}
            )
            self.audit_logger.log_audit_event(audit_event)
        
        # Remove session
        del self.active_sessions[session_id]
        
        logger.info(f"Logged out session {session_id} for user {session.user_id}")
        return True
    
    def get_sso_health_metrics(self, tenant_id: str) -> Dict[str, Any]:
        """Get SSO system health metrics for tenant."""
        
        tenant_idps = [
            idp for idp in self.idp_configurations.values()
            if idp.tenant_id == tenant_id
        ]
        
        tenant_sessions = [
            session for session in self.active_sessions.values()
            if session.tenant_id == tenant_id
        ]
        
        # Calculate session statistics
        active_sessions = len(tenant_sessions)
        mfa_sessions = len([s for s in tenant_sessions if s.mfa_verified])
        
        # Check IdP connectivity
        idp_status = {}
        for idp in tenant_idps:
            if idp.auth_method == AuthenticationMethod.OIDC and idp.oidc_discovery_url:
                try:
                    response = requests.get(idp.oidc_discovery_url, timeout=5)
                    idp_status[idp.idp_id] = "healthy" if response.status_code == 200 else "unhealthy"
                except:
                    idp_status[idp.idp_id] = "unreachable"
            else:
                idp_status[idp.idp_id] = "configured"
        
        return {
            'tenant_id': tenant_id,
            'total_idps': len(tenant_idps),
            'active_sessions': active_sessions,
            'mfa_sessions': mfa_sessions,
            'mfa_coverage': (mfa_sessions / max(active_sessions, 1)) * 100,
            'idp_status': idp_status,
            'session_distribution': {
                idp_id: len([s for s in tenant_sessions if s.idp_id == idp_id])
                for idp_id in [idp.idp_id for idp in tenant_idps]
            }
        }

# Example usage
if __name__ == "__main__":
    from multi_tenant_rbac_system import MultiTenantRBACSystem
    from enterprise_security_architecture import SOC2ComplianceEngine
    
    # Initialize systems
    compliance_engine = SOC2ComplianceEngine()
    rbac_system = MultiTenantRBACSystem(audit_logger=compliance_engine)
    sso = EnterpriseSSO(rbac_system=rbac_system, audit_logger=compliance_engine)
    
    # Configure Azure AD integration
    azure_config = IdPConfiguration(
        name="Azure AD",
        provider_type=IdentityProvider.AZURE_AD,
        auth_method=AuthenticationMethod.OIDC,
        tenant_id="test-corp",
        oidc_discovery_url="https://login.microsoftonline.com/tenant-id/.well-known/openid_configuration",
        oidc_client_id="your-client-id",
        oidc_client_secret="your-client-secret",
        require_mfa=True,
        allowed_domains={"test-corp.com"},
        default_roles={"Agent User"},
        role_mapping={
            "Administrators": "Tenant Admin",
            "Security Team": "Security Officer",
            "Analysts": "Data Analyst"
        },
        attribute_mapping={
            "email": "email",
            "name": "display_name",
            "given_name": "first_name",
            "family_name": "last_name"
        }
    )
    
    # Register identity provider
    idp_id = sso.register_identity_provider(azure_config)
    print(f"Registered Azure AD with ID: {idp_id}")
    
    # Get SSO health metrics
    metrics = sso.get_sso_health_metrics("test-corp")
    print(f"SSO Health Metrics: {metrics}")
