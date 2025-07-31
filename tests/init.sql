-- SingleStore Test Database Initialization
-- This file is automatically executed when the container starts

-- Create test database if it doesn't exist
CREATE DATABASE IF NOT EXISTS test_db;

-- Create additional test databases for examples
CREATE DATABASE IF NOT EXISTS test_example;
CREATE DATABASE IF NOT EXISTS test_example_async;

-- Switch to test database and verify setup
USE test_db;

-- Create a simple test table to verify database is working
CREATE TABLE IF NOT EXISTS health_check (
    id INT AUTO_INCREMENT PRIMARY KEY,
    status VARCHAR(50) DEFAULT 'ready',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert a test record
INSERT INTO health_check (status) VALUES ('initialized');

-- Show databases to confirm setup
SELECT 'Database initialization complete' as status;