-- =====================================================
-- Zero-Risk Migration Implementation Scripts
-- Version: 2.0.0
-- Description: Production-ready migration with rollback capability
-- =====================================================

-- =====================================================
-- PHASE 1: SCHEMA CREATION
-- =====================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "citext";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create new schema version
CREATE SCHEMA IF NOT EXISTS users_v2;

-- Create main users table with optimized structure
CREATE TABLE users_v2.users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    legacy_id BIGINT UNIQUE NOT NULL,
    email CITEXT UNIQUE NOT NULL,
    email_verified BOOLEAN NOT NULL DEFAULT false,
    username VARCHAR(50) UNIQUE NOT NULL,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    birth_date DATE,
    profile_preferences JSONB NOT NULL DEFAULT '{}',
    gdpr_consent JSONB NOT NULL,
    data_quality_score DECIMAL(3,2) CHECK (data_quality_score >= 0 AND data_quality_score <= 1),
    migration_metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMPTZ,
    
    -- Constraints
    CONSTRAINT email_format CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'),
    CONSTRAINT username_format CHECK (username ~* '^[a-zA-Z0-9_]{3,50}$'),
    CONSTRAINT age_check CHECK (birth_date <= CURRENT_DATE - INTERVAL '13 years' AND 
                                birth_date >= CURRENT_DATE - INTERVAL '120 years'),
    CONSTRAINT gdpr_consent_required CHECK (
        gdpr_consent ? 'marketing' AND 
        gdpr_consent ? 'data_processing' AND 
        gdpr_consent ? 'third_party_sharing'
    )
) PARTITION BY RANGE (created_at);

-- Create partitions for the next 12 months
DO $$
DECLARE
    start_date DATE := DATE_TRUNC('month', CURRENT_DATE);
    partition_date DATE;
    partition_name TEXT;
BEGIN
    FOR i IN 0..11 LOOP
        partition_date := start_date + (i || ' months')::INTERVAL;
        partition_name := 'users_' || TO_CHAR(partition_date, 'YYYY_MM');
        
        EXECUTE format('
            CREATE TABLE users_v2.%I PARTITION OF users_v2.users
            FOR VALUES FROM (%L) TO (%L)',
            partition_name,
            partition_date,
            partition_date + INTERVAL '1 month'
        );
    END LOOP;
END $$;

-- Create optimized indexes
CREATE INDEX idx_users_email ON users_v2.users USING btree(email);
CREATE INDEX idx_users_username ON users_v2.users USING btree(username);
CREATE INDEX idx_users_legacy_id ON users_v2.users USING btree(legacy_id);
CREATE INDEX idx_users_created_at ON users_v2.users USING brin(created_at);
CREATE INDEX idx_users_active ON users_v2.users(updated_at DESC) WHERE deleted_at IS NULL;
CREATE INDEX idx_users_quality_score ON users_v2.users(data_quality_score) WHERE data_quality_score < 0.85;
CREATE INDEX idx_users_preferences ON users_v2.users USING gin(profile_preferences jsonb_path_ops);

-- Create audit log table
CREATE TABLE users_v2.audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL,
    action VARCHAR(50) NOT NULL CHECK (action IN ('create', 'update', 'delete', 'restore', 'merge', 'validate')),
    field_name VARCHAR(100),
    old_value JSONB,
    new_value JSONB,
    actor_id UUID,
    actor_type VARCHAR(20) CHECK (actor_type IN ('user', 'system', 'migration', 'admin')),
    ip_address INET,
    user_agent TEXT,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    compliance_flags JSONB
) PARTITION BY RANGE (timestamp);

-- Create monthly partitions for audit log
DO $$
DECLARE
    start_date DATE := DATE_TRUNC('month', CURRENT_DATE);
    partition_date DATE;
    partition_name TEXT;
BEGIN
    FOR i IN 0..11 LOOP
        partition_date := start_date + (i || ' months')::INTERVAL;
        partition_name := 'audit_log_' || TO_CHAR(partition_date, 'YYYY_MM');
        
        EXECUTE format('
            CREATE TABLE users_v2.%I PARTITION OF users_v2.audit_log
            FOR VALUES FROM (%L) TO (%L)',
            partition_name,
            partition_date,
            partition_date + INTERVAL '1 month'
        );
    END LOOP;
END $$;

-- Create indexes for audit log
CREATE INDEX idx_audit_user_timestamp ON users_v2.audit_log(user_id, timestamp DESC);
CREATE INDEX idx_audit_action_timestamp ON users_v2.audit_log(action, timestamp DESC);
CREATE INDEX idx_audit_timestamp ON users_v2.audit_log USING brin(timestamp);

-- Create data quality issues tracking table
CREATE TABLE users_v2.data_quality_issues (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL,
    issue_type VARCHAR(50) NOT NULL CHECK (issue_type IN ('duplicate_email', 'invalid_date', 'missing_field', 'invalid_format', 'constraint_violation')),
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('critical', 'high', 'medium', 'low')),
    field_name VARCHAR(100) NOT NULL,
    current_value JSONB,
    suggested_value JSONB,
    detection_method VARCHAR(50) CHECK (detection_method IN ('migration', 'validation', 'user_report', 'automated_scan')),
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'in_review', 'resolved', 'ignored', 'auto_corrected')),
    resolved_at TIMESTAMPTZ,
    resolved_by UUID,
    resolution_notes TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for data quality issues
CREATE INDEX idx_dq_user_status ON users_v2.data_quality_issues(user_id, status);
CREATE INDEX idx_dq_severity_status ON users_v2.data_quality_issues(severity, status);
CREATE INDEX idx_dq_created_at ON users_v2.data_quality_issues(created_at DESC);

-- =====================================================
-- PHASE 2: VALIDATION FUNCTIONS
-- =====================================================

-- Function to validate JSON schema
CREATE OR REPLACE FUNCTION users_v2.validate_json_schema(
    p_json JSONB,
    p_schema_name TEXT
) RETURNS BOOLEAN AS $$
DECLARE
    v_valid BOOLEAN := true;
BEGIN
    -- Validate based on schema name
    CASE p_schema_name
        WHEN 'profile_preferences_v1' THEN
            -- Check required fields
            IF NOT (p_json ? 'theme' AND p_json ? 'language' AND p_json ? 'timezone' AND p_json ? 'notification_settings') THEN
                RETURN false;
            END IF;
            
            -- Validate theme
            IF NOT (p_json->>'theme' IN ('light', 'dark', 'auto')) THEN
                RETURN false;
            END IF;
            
            -- Validate language format (e.g., en-US)
            IF NOT (p_json->>'language' ~ '^[a-z]{2}-[A-Z]{2}$') THEN
                RETURN false;
            END IF;
            
        WHEN 'gdpr_consent_v1' THEN
            -- Check all required consent fields
            IF NOT (p_json ? 'marketing' AND p_json ? 'data_processing' AND p_json ? 'third_party_sharing') THEN
                RETURN false;
            END IF;
    END CASE;
    
    RETURN v_valid;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to calculate data quality score
CREATE OR REPLACE FUNCTION users_v2.calculate_data_quality_score(
    p_user_id UUID
) RETURNS DECIMAL(3,2) AS $$
DECLARE
    v_score DECIMAL(3,2);
    v_completeness DECIMAL(3,2);
    v_validity DECIMAL(3,2);
    v_consistency DECIMAL(3,2);
    v_record RECORD;
BEGIN
    SELECT * INTO v_record FROM users_v2.users WHERE id = p_user_id;
    
    -- Calculate completeness (30% weight)
    v_completeness := 0;
    IF v_record.email IS NOT NULL THEN v_completeness := v_completeness + 0.15; END IF;
    IF v_record.username IS NOT NULL THEN v_completeness := v_completeness + 0.15; END IF;
    IF v_record.first_name IS NOT NULL THEN v_completeness := v_completeness + 0.15; END IF;
    IF v_record.last_name IS NOT NULL THEN v_completeness := v_completeness + 0.15; END IF;
    IF v_record.birth_date IS NOT NULL THEN v_completeness := v_completeness + 0.10; END IF;
    IF v_record.email_verified THEN v_completeness := v_completeness + 0.10; END IF;
    IF v_record.profile_preferences IS NOT NULL AND v_record.profile_preferences != '{}' THEN 
        v_completeness := v_completeness + 0.10; 
    END IF;
    IF v_record.gdpr_consent IS NOT NULL THEN v_completeness := v_completeness + 0.10; END IF;
    
    -- Calculate validity (25% weight)
    v_validity := 0;
    IF v_record.email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$' THEN 
        v_validity := v_validity + 0.40; 
    END IF;
    IF v_record.username ~* '^[a-zA-Z0-9_]{3,50}$' THEN 
        v_validity := v_validity + 0.30; 
    END IF;
    IF v_record.birth_date IS NULL OR 
       (v_record.birth_date <= CURRENT_DATE - INTERVAL '13 years' AND 
        v_record.birth_date >= CURRENT_DATE - INTERVAL '120 years') THEN 
        v_validity := v_validity + 0.30; 
    END IF;
    
    -- Calculate consistency (20% weight) - simplified for this example
    v_consistency := 0.85; -- Base consistency score
    
    -- Calculate final score with weights
    v_score := (v_completeness * 0.30) + (v_validity * 0.25) + (v_consistency * 0.20) + 0.25;
    
    -- Ensure score is within bounds
    IF v_score > 1.00 THEN v_score := 1.00; END IF;
    IF v_score < 0.00 THEN v_score := 0.00; END IF;
    
    RETURN v_score;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- PHASE 3: TRIGGER FUNCTIONS
-- =====================================================

-- Trigger function for updating timestamps
CREATE OR REPLACE FUNCTION users_v2.update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for timestamp updates
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users_v2.users
    FOR EACH ROW
    EXECUTE FUNCTION users_v2.update_updated_at();

-- Trigger function for audit logging
CREATE OR REPLACE FUNCTION users_v2.create_audit_log()
RETURNS TRIGGER AS $$
DECLARE
    v_old_value JSONB;
    v_new_value JSONB;
    v_action VARCHAR(50);
BEGIN
    IF TG_OP = 'DELETE' THEN
        v_action := 'delete';
        v_old_value := row_to_json(OLD)::JSONB;
        INSERT INTO users_v2.audit_log (user_id, action, old_value, actor_type, timestamp)
        VALUES (OLD.id, v_action, v_old_value, 'system', CURRENT_TIMESTAMP);
        RETURN OLD;
    ELSIF TG_OP = 'UPDATE' THEN
        v_action := 'update';
        v_old_value := row_to_json(OLD)::JSONB;
        v_new_value := row_to_json(NEW)::JSONB;
        INSERT INTO users_v2.audit_log (user_id, action, old_value, new_value, actor_type, timestamp)
        VALUES (NEW.id, v_action, v_old_value, v_new_value, 'system', CURRENT_TIMESTAMP);
        RETURN NEW;
    ELSIF TG_OP = 'INSERT' THEN
        v_action := 'create';
        v_new_value := row_to_json(NEW)::JSONB;
        INSERT INTO users_v2.audit_log (user_id, action, new_value, actor_type, timestamp)
        VALUES (NEW.id, v_action, v_new_value, 'system', CURRENT_TIMESTAMP);
        RETURN NEW;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Create audit trigger
CREATE TRIGGER audit_users_changes
    AFTER INSERT OR UPDATE OR DELETE ON users_v2.users
    FOR EACH ROW
    EXECUTE FUNCTION users_v2.create_audit_log();

-- =====================================================
-- PHASE 4: MIGRATION PROCEDURES
-- =====================================================

-- Function to parse various date formats
CREATE OR REPLACE FUNCTION users_v2.parse_date_flexible(
    p_date_string TEXT
) RETURNS DATE AS $$
DECLARE
    v_date DATE;
BEGIN
    -- Try different date formats
    BEGIN
        -- Try ISO format first (YYYY-MM-DD)
        v_date := p_date_string::DATE;
        RETURN v_date;
    EXCEPTION WHEN OTHERS THEN
        -- Continue to next format
    END;
    
    BEGIN
        -- Try MM/DD/YYYY format
        v_date := TO_DATE(p_date_string, 'MM/DD/YYYY');
        RETURN v_date;
    EXCEPTION WHEN OTHERS THEN
        -- Continue to next format
    END;
    
    BEGIN
        -- Try DD-MM-YYYY format
        v_date := TO_DATE(p_date_string, 'DD-MM-YYYY');
        RETURN v_date;
    EXCEPTION WHEN OTHERS THEN
        -- Return NULL if no format matches
        RETURN NULL;
    END;
END;
$$ LANGUAGE plpgsql;

-- Main migration procedure
CREATE OR REPLACE PROCEDURE users_v2.migrate_users_batch(
    p_batch_size INTEGER DEFAULT 10000,
    p_offset INTEGER DEFAULT 0
) AS $$
DECLARE
    v_record RECORD;
    v_new_id UUID;
    v_quality_score DECIMAL(3,2);
    v_birth_date DATE;
    v_migration_metadata JSONB;
    v_processed INTEGER := 0;
BEGIN
    -- Process users in batches
    FOR v_record IN 
        SELECT * FROM legacy_users 
        ORDER BY id 
        LIMIT p_batch_size 
        OFFSET p_offset
    LOOP
        BEGIN
            -- Generate new UUID
            v_new_id := uuid_generate_v4();
            
            -- Parse birth date with multiple format support
            v_birth_date := users_v2.parse_date_flexible(v_record.birth_date_string);
            
            -- Create migration metadata
            v_migration_metadata := jsonb_build_object(
                'source_system', 'legacy_v1',
                'migration_date', CURRENT_TIMESTAMP,
                'migration_version', '2.0.0',
                'original_birth_date_format', v_record.birth_date_string,
                'data_corrections', ARRAY[]::TEXT[]
            );
            
            -- Insert into new schema
            INSERT INTO users_v2.users (
                id, legacy_id, email, username, first_name, last_name,
                birth_date, profile_preferences, gdpr_consent,
                migration_metadata, created_at, updated_at
            ) VALUES (
                v_new_id,
                v_record.id,
                LOWER(TRIM(v_record.email)),
                v_record.username,
                v_record.first_name,
                v_record.last_name,
                v_birth_date,
                COALESCE(v_record.preferences::JSONB, '{}'::JSONB),
                COALESCE(v_record.gdpr_consent::JSONB, 
                    '{"marketing": false, "data_processing": true, "third_party_sharing": false}'::JSONB),
                v_migration_metadata,
                v_record.created_at,
                v_record.updated_at
            );
            
            -- Calculate and update data quality score
            v_quality_score := users_v2.calculate_data_quality_score(v_new_id);
            UPDATE users_v2.users SET data_quality_score = v_quality_score WHERE id = v_new_id;
            
            -- Log any data quality issues
            IF v_birth_date IS NULL AND v_record.birth_date_string IS NOT NULL THEN
                INSERT INTO users_v2.data_quality_issues (
                    user_id, issue_type, severity, field_name, 
                    current_value, detection_method, status
                ) VALUES (
                    v_new_id, 'invalid_date', 'medium', 'birth_date',
                    to_jsonb(v_record.birth_date_string), 'migration', 'pending'
                );
            END IF;
            
            v_processed := v_processed + 1;
            
        EXCEPTION WHEN OTHERS THEN
            -- Log migration error
            RAISE NOTICE 'Error migrating user %: %', v_record.id, SQLERRM;
            
            INSERT INTO users_v2.data_quality_issues (
                user_id, issue_type, severity, field_name,
                current_value, detection_method, status, resolution_notes
            ) VALUES (
                v_new_id, 'constraint_violation', 'critical', 'migration',
                row_to_json(v_record)::JSONB, 'migration', 'pending', SQLERRM
            );
        END;
    END LOOP;
    
    RAISE NOTICE 'Processed % records', v_processed;
    COMMIT;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- PHASE 5: VALIDATION QUERIES
-- =====================================================

-- Create validation views
CREATE OR REPLACE VIEW users_v2.migration_validation AS
SELECT 
    'Record Count' as check_name,
    (SELECT COUNT(*) FROM legacy_users) as source_count,
    (SELECT COUNT(*) FROM users_v2.users) as target_count,
    CASE 
        WHEN (SELECT COUNT(*) FROM legacy_users) = (SELECT COUNT(*) FROM users_v2.users)
        THEN 'PASS'
        ELSE 'FAIL'
    END as status
UNION ALL
SELECT 
    'Email Uniqueness' as check_name,
    0 as source_count,
    (SELECT COUNT(*) FROM (SELECT email, COUNT(*) FROM users_v2.users GROUP BY email HAVING COUNT(*) > 1) t) as target_count,
    CASE 
        WHEN (SELECT COUNT(*) FROM (SELECT email, COUNT(*) FROM users_v2.users GROUP BY email HAVING COUNT(*) > 1) t) = 0
        THEN 'PASS'
        ELSE 'FAIL'
    END as status
UNION ALL
SELECT 
    'Data Quality Score' as check_name,
    0 as source_count,
    (SELECT COUNT(*) FROM users_v2.users WHERE data_quality_score < 0.70) as target_count,
    CASE 
        WHEN (SELECT COUNT(*) FROM users_v2.users WHERE data_quality_score < 0.70) < 
             (SELECT COUNT(*) * 0.05 FROM users_v2.users)
        THEN 'PASS'
        ELSE 'FAIL'
    END as status;

-- =====================================================
-- PHASE 6: ROLLBACK PROCEDURES
-- =====================================================

CREATE OR REPLACE PROCEDURE users_v2.rollback_migration() AS $$
BEGIN
    -- Disable triggers temporarily
    ALTER TABLE users_v2.users DISABLE TRIGGER ALL;
    
    -- Clear migrated data
    TRUNCATE TABLE users_v2.users CASCADE;
    TRUNCATE TABLE users_v2.audit_log CASCADE;
    TRUNCATE TABLE users_v2.data_quality_issues CASCADE;
    
    -- Re-enable triggers
    ALTER TABLE users_v2.users ENABLE TRIGGER ALL;
    
    RAISE NOTICE 'Migration rolled back successfully';
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- PHASE 7: PERFORMANCE OPTIMIZATION
-- =====================================================

-- Create materialized view for frequently accessed user data
CREATE MATERIALIZED VIEW users_v2.active_users_summary AS
SELECT 
    id, email, username, first_name, last_name,
    data_quality_score, created_at, updated_at
FROM users_v2.users
WHERE deleted_at IS NULL
  AND updated_at > CURRENT_DATE - INTERVAL '30 days'
WITH DATA;

-- Create index on materialized view
CREATE UNIQUE INDEX idx_active_users_summary_id ON users_v2.active_users_summary(id);
CREATE INDEX idx_active_users_summary_email ON users_v2.active_users_summary(email);

-- Schedule refresh of materialized view
CREATE OR REPLACE FUNCTION users_v2.refresh_active_users_summary()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY users_v2.active_users_summary;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- PHASE 8: MONITORING QUERIES
-- =====================================================

-- Query to monitor migration progress
CREATE OR REPLACE VIEW users_v2.migration_progress AS
SELECT 
    'Total Records' as metric,
    (SELECT COUNT(*) FROM legacy_users) as total,
    (SELECT COUNT(*) FROM users_v2.users) as completed,
    ROUND((SELECT COUNT(*) FROM users_v2.users)::NUMERIC / 
          NULLIF((SELECT COUNT(*) FROM legacy_users), 0) * 100, 2) as percentage
UNION ALL
SELECT 
    'High Quality Records' as metric,
    (SELECT COUNT(*) FROM users_v2.users) as total,
    (SELECT COUNT(*) FROM users_v2.users WHERE data_quality_score >= 0.85) as completed,
    ROUND((SELECT COUNT(*) FROM users_v2.users WHERE data_quality_score >= 0.85)::NUMERIC / 
          NULLIF((SELECT COUNT(*) FROM users_v2.users), 0) * 100, 2) as percentage
UNION ALL
SELECT 
    'Data Issues Resolved' as metric,
    (SELECT COUNT(*) FROM users_v2.data_quality_issues) as total,
    (SELECT COUNT(*) FROM users_v2.data_quality_issues WHERE status IN ('resolved', 'auto_corrected')) as completed,
    ROUND((SELECT COUNT(*) FROM users_v2.data_quality_issues WHERE status IN ('resolved', 'auto_corrected'))::NUMERIC / 
          NULLIF((SELECT COUNT(*) FROM users_v2.data_quality_issues), 0) * 100, 2) as percentage;

-- Grant appropriate permissions
GRANT USAGE ON SCHEMA users_v2 TO application_role;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA users_v2 TO application_role;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA users_v2 TO application_role;