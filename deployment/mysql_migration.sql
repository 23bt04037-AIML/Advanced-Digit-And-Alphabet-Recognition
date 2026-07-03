-- ============================================================
-- DigitAI  –  MySQL / phpMyAdmin Migration Script
-- Run this in phpMyAdmin on your XAMPP installation.
-- ============================================================

CREATE DATABASE IF NOT EXISTS digit_recognition
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE digit_recognition;

-- ── Users ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    id            INT          NOT NULL AUTO_INCREMENT,
    username      VARCHAR(64)  NOT NULL UNIQUE,
    password_hash VARCHAR(256) NOT NULL,
    role          VARCHAR(10)  NOT NULL DEFAULT 'client',
    full_name     VARCHAR(128),
    email         VARCHAR(256),
    is_active     TINYINT(1)   NOT NULL DEFAULT 1,
    created_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_login    DATETIME,
    PRIMARY KEY (id),
    UNIQUE KEY uq_users_username (username),
    INDEX ix_users_username (username)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Default admin account  (password: Admin@123)
-- SHA-256 of "Admin@123"
INSERT IGNORE INTO users (username, password_hash, role, full_name, email)
VALUES (
  'admin',
  'e86f78a8a3caf0b60d8e74e5942aa6d86dc150cd3c03338aef25b7d2d7e3acc7',
  'admin',
  'Administrator',
  'admin@digitai.local'
);

-- ── OTP Records (for Forgot Password) ───────────────────────
CREATE TABLE IF NOT EXISTS otp_records (
    id         INT         NOT NULL AUTO_INCREMENT,
    user_id    INT         NOT NULL,
    otp_code   VARCHAR(6)  NOT NULL,
    expires_at DATETIME    NOT NULL,
    used       TINYINT(1)  NOT NULL DEFAULT 0,
    created_at DATETIME    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    INDEX ix_otp_user_id (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ── Prediction Logs ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS prediction_logs (
    id                  INT          NOT NULL AUTO_INCREMENT,
    session_id          VARCHAR(64)  NOT NULL,
    user_id             INT,
    username            VARCHAR(64),
    input_type          VARCHAR(20)  NOT NULL,
    image_path          VARCHAR(512),
    model_used          VARCHAR(50)  NOT NULL,
    predicted_digit     INT          NOT NULL,
    confidence          FLOAT        NOT NULL,
    top3_predictions    JSON,
    all_probabilities   JSON,
    gradcam_path        VARCHAR(512),
    lime_path           VARCHAR(512),
    saliency_path       VARCHAR(512),
    processing_time_ms  FLOAT,
    is_correct          TINYINT(1),
    true_label          INT,
    created_at          DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    INDEX ix_prediction_logs_session_id (session_id),
    INDEX ix_prediction_logs_user_id    (user_id),
    INDEX ix_prediction_logs_username   (username),
    CONSTRAINT fk_user_logs FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ── Audit Logs ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS audit_logs (
    id          INT          NOT NULL AUTO_INCREMENT,
    action      VARCHAR(100) NOT NULL,
    entity_type VARCHAR(50),
    entity_id   INT,
    user_agent  VARCHAR(512),
    ip_address  VARCHAR(45),
    details     JSON,
    created_at  DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ── Runtime Logs ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS runtime_logs (
    id         INT          NOT NULL AUTO_INCREMENT,
    level      VARCHAR(10)  NOT NULL,
    module     VARCHAR(100) NOT NULL,
    message    TEXT         NOT NULL,
    extra      JSON,
    created_at DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ── Model Metrics ────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS model_metrics (
    id                    INT         NOT NULL AUTO_INCREMENT,
    model_name            VARCHAR(50) NOT NULL UNIQUE,
    accuracy              FLOAT,
    precision_score       FLOAT,
    recall_score          FLOAT,
    f1_score              FLOAT,
    auc_roc               FLOAT,
    confusion_matrix      JSON,
    training_epochs       INT,
    training_time_seconds FLOAT,
    parameters            INT,
    created_at            DATETIME    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at            DATETIME    ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============================================================
-- To migrate existing SQLite data use the Python helper:
--   python deployment/migrate_sqlite_to_mysql.py
-- ============================================================

