# Real-Time State Management Architecture

## Overview

This document describes the architecture for the real-time state management system that provides visibility into instance lifecycle transitions with explicit AWS confirmation flow.

## System Components

```
models-ml/
├── backend/                    # Backend API and state management
│   ├── app.py                 # Main FastAPI application
│   ├── state_machine.py       # Instance state machine logic
│   ├── websocket_manager.py   # WebSocket event broadcasting
│   ├── models.py              # Database models (SQLAlchemy)
│   ├── schemas.py             # Pydantic schemas for API
│   ├── api/                   # API endpoints
│   │   ├── instances.py       # Instance management endpoints
│   │   ├── actions.py         # Action endpoints (launch, terminate)
│   │   └── events.py          # Event history endpoints
│   ├── services/              # Business logic
│   │   ├── instance_service.py
│   │   ├── confirmation_service.py
│   │   └── validation_service.py
│   └── config.py              # Configuration
│
├── agent/                      # AWS agent for instance management
│   ├── agent.py               # Main agent application
│   ├── aws_client.py          # AWS SDK integration (boto3)
│   ├── command_handler.py     # Command execution logic
│   ├── polling.py             # AWS state polling
│   ├── confirmation.py        # Confirmation event logic
│   └── config.py              # Agent configuration
│
├── frontend/                   # React frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── InstancesView.tsx      # Main instances view
│   │   │   ├── InstanceRow.tsx        # Individual instance row
│   │   │   ├── StatusBadge.tsx        # Status badge with colors
│   │   │   ├── ActionButtons.tsx      # Action buttons
│   │   │   └── HistoryView.tsx        # History view
│   │   ├── hooks/
│   │   │   ├── useWebSocket.ts        # WebSocket connection
│   │   │   ├── useInstances.ts        # Instance state management
│   │   │   └── useOptimisticUpdate.ts # Optimistic updates
│   │   ├── store/
│   │   │   └── instanceStore.ts       # Zustand store
│   │   ├── types/
│   │   │   └── instance.ts            # TypeScript types
│   │   └── App.tsx
│   ├── package.json
│   └── tsconfig.json
│
├── database/                   # Database schema and migrations
│   ├── migrations/            # Alembic migrations
│   │   ├── 001_initial_schema.sql
│   │   ├── 002_add_launching_terminating_states.sql
│   │   └── 003_add_timestamp_columns.sql
│   ├── schema.sql             # Complete schema
│   └── seed_data.sql          # Test data
│
├── docs/                       # Documentation
│   ├── API.md                 # API documentation
│   ├── WEBSOCKET_EVENTS.md    # WebSocket event reference
│   ├── STATE_MACHINE.md       # State machine documentation
│   ├── DEPLOYMENT.md          # Deployment guide
│   └── MONITORING.md          # Monitoring and observability
│
├── tests/                      # Test suite
│   ├── backend/
│   ├── agent/
│   ├── frontend/
│   └── integration/
│
└── docker/                     # Docker configuration
    ├── backend.Dockerfile
    ├── agent.Dockerfile
    ├── frontend.Dockerfile
    └── docker-compose.yml
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND (React)                        │
│  ┌────────────────────────────────────────────────────────┐   │
│  │  InstancesView Component                                │   │
│  │  - Optimistic UI updates                                │   │
│  │  - WebSocket listener                                   │   │
│  │  - Status badges (LAUNCHING, RUNNING, TERMINATING...)  │   │
│  └────────────────────────────────────────────────────────┘   │
└───────────────────────┬─────────────────────────────────────────┘
                        │ WebSocket + REST API
                        │
┌───────────────────────▼─────────────────────────────────────────┐
│                    BACKEND (FastAPI)                            │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  State Machine                                           │  │
│  │  - Immediate action execution                           │  │
│  │  - State validation                                     │  │
│  │  - WebSocket event broadcasting                         │  │
│  │  - Confirmation handlers (first wins)                   │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Database (PostgreSQL)                                   │  │
│  │  - instances_runtime (status, timestamps)               │  │
│  │  - instance_lifecycle_events                            │  │
│  │  - instance_runs (historical)                           │  │
│  └─────────────────────────────────────────────────────────┘  │
└───────────────────────┬─────────────────────────────────────────┘
                        │ Command Queue (High Priority)
                        │
┌───────────────────────▼─────────────────────────────────────────┐
│                      AGENT (Python)                             │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Command Handler                                         │  │
│  │  - Immediate execution (no deferral)                    │  │
│  │  - AWS API calls (RunInstances, TerminateInstances)    │  │
│  │  - Send ACTION_RESULT                                   │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  AWS Polling                                             │  │
│  │  - Poll DescribeInstances for confirmation             │  │
│  │  - Send LAUNCH_CONFIRMED / TERMINATE_CONFIRMED         │  │
│  │  - Timeout handling                                     │  │
│  └─────────────────────────────────────────────────────────┘  │
└───────────────────────┬─────────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────────┐
│                       AWS EC2 API                               │
│  - RunInstances                                                 │
│  - TerminateInstances                                           │
│  - DescribeInstances                                            │
│  - DescribeInstanceStatus                                       │
└─────────────────────────────────────────────────────────────────┘
```

## State Machine

### States

- **LAUNCHING**: Instance requested, AWS creating it (not ready yet)
- **RUNNING**: Instance confirmed running by AWS, passes health checks
- **PROMOTING**: Replica being promoted to primary
- **TERMINATING**: Terminate command issued, awaiting AWS confirmation
- **TERMINATED**: AWS confirmed instance is terminated (moves to History)
- **OFFLINE**: Instance not responding to health checks

### Transitions

#### Launch Flow
```
NONE → LAUNCHING → RUNNING
```

1. User clicks "Launch" button
2. Frontend optimistically shows LAUNCHING status
3. Backend sends LAUNCH_INSTANCE command to agent
4. Backend updates DB: status=LAUNCHING, launch_requested_at=now
5. Backend broadcasts WebSocket event: INSTANCE_LAUNCHING
6. Agent calls AWS RunInstances API
7. Agent sends ACTION_RESULT to backend
8. Agent polls DescribeInstances until state=running
9. Agent sends LAUNCH_CONFIRMED to backend
10. Backend updates DB: status=RUNNING, launch_confirmed_at=now
11. Backend broadcasts WebSocket event: INSTANCE_RUNNING
12. Frontend updates status badge to RUNNING

#### Terminate Flow
```
RUNNING/ZOMBIE/REPLICA/PRIMARY → TERMINATING → TERMINATED
```

1. User clicks "Terminate" button
2. Frontend optimistically shows TERMINATING status
3. Backend sends TERMINATE_INSTANCE command to agent
4. Backend updates DB: status=TERMINATING, termination_requested_at=now
5. Backend broadcasts WebSocket event: INSTANCE_TERMINATING
6. Agent calls AWS TerminateInstances API
7. Agent sends ACTION_RESULT to backend
8. Agent polls DescribeInstances until state=terminated
9. Agent sends TERMINATE_CONFIRMED to backend
10. Backend updates DB: status=TERMINATED, termination_confirmed_at=now
11. Backend broadcasts WebSocket event: INSTANCE_TERMINATED
12. Frontend moves instance to History view

### Invariants

1. **Launch**: Must pass through LAUNCHING before RUNNING for new instances
2. **Terminate**: Must pass through TERMINATING before TERMINATED for system-initiated terminations
3. **No Skip**: Cannot skip transitional states (LAUNCHING, TERMINATING)
4. **Confirmation Required**: Backend never marks TERMINATED without explicit confirmation from AWS
5. **First Wins**: If multiple confirmation sources report, first one wins (idempotent)

## Data Flow

### Action Execution (Immediate)

```
User Action → Frontend (optimistic) → Backend (state update) → Agent (immediate execution)
                                    → WebSocket broadcast
```

**Key Points:**
- Actions execute immediately, NOT deferred to next cycle
- Frontend gets immediate feedback (optimistic update)
- Backend broadcasts state change via WebSocket
- Agent receives command on high-priority queue
- Agent executes immediately in separate thread

### Confirmation Flow

```
Agent polls AWS → Confirms state change → Sends event to Backend → Backend updates DB → WebSocket broadcast → Frontend updates
```

**Confirmation Sources:**
1. Agent AWS poll (DescribeInstances)
2. Backend direct AWS check
3. New primary confirmation (for terminate)

**First confirmation wins** - subsequent confirmations are idempotent.

## Database Schema

### instances_runtime

Primary table for real-time instance state.

```sql
CREATE TABLE instances_runtime (
    instance_id VARCHAR(255) PRIMARY KEY,
    status VARCHAR(50) NOT NULL,  -- LAUNCHING, RUNNING, PROMOTING, TERMINATING, TERMINATED, OFFLINE
    cloud_instance_id VARCHAR(255),
    instance_type VARCHAR(50),
    availability_zone VARCHAR(50),
    launch_requested_at TIMESTAMP NULL,
    launch_confirmed_at TIMESTAMP NULL,
    termination_requested_at TIMESTAMP NULL,
    termination_confirmed_at TIMESTAMP NULL,
    launch_duration_seconds INT NULL,
    termination_duration_seconds INT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    INDEX idx_status (status),
    INDEX idx_status_requested_at (status, launch_requested_at),
    INDEX idx_cloud_instance_id (cloud_instance_id)
);
```

### instance_lifecycle_events

Event log for all lifecycle transitions.

```sql
CREATE TABLE instance_lifecycle_events (
    event_id BIGINT PRIMARY KEY AUTO_INCREMENT,
    instance_id VARCHAR(255) NOT NULL,
    event_type VARCHAR(50) NOT NULL,  -- LAUNCH_REQUESTED, LAUNCH_CONFIRMED, TERMINATE_REQUESTED, TERMINATE_CONFIRMED
    source VARCHAR(50) NOT NULL,      -- agent, backend, new_primary_confirmation, external
    reason VARCHAR(100) NULL,         -- manual, auto_switch, emergency, cleanup, cost_optimization
    confirmation_method VARCHAR(100) NULL,  -- agent_aws_poll, backend_aws_check, new_primary_health_check
    cloud_instance_id VARCHAR(255) NULL,
    metadata JSON NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_instance_id (instance_id),
    INDEX idx_event_type (event_type),
    INDEX idx_created_at (created_at)
);
```

### instance_runs

Historical data for completed instance lifecycles (primary DB).

```sql
CREATE TABLE instance_runs (
    run_id BIGINT PRIMARY KEY AUTO_INCREMENT,
    instance_id VARCHAR(255) NOT NULL,
    cloud_instance_id VARCHAR(255),
    instance_type VARCHAR(50),
    availability_zone VARCHAR(50),
    launched_at TIMESTAMP NULL,
    terminated_at TIMESTAMP NULL,
    termination_confirmed_at TIMESTAMP NULL,
    launch_duration_seconds INT NULL,
    termination_duration_seconds INT NULL,
    total_runtime_seconds INT NULL,
    termination_reason VARCHAR(100) NULL,

    INDEX idx_instance_id (instance_id),
    INDEX idx_launched_at (launched_at),
    INDEX idx_terminated_at (terminated_at)
);
```

## WebSocket Events

### Event Types

#### INSTANCE_LAUNCHING
```json
{
  "event_type": "INSTANCE_LAUNCHING",
  "instance_id": "inst-12345",
  "cloud_instance_id": null,
  "status": "LAUNCHING",
  "launch_requested_at": "2025-11-27T10:30:00Z",
  "timestamp": "2025-11-27T10:30:00Z"
}
```

#### INSTANCE_RUNNING
```json
{
  "event_type": "INSTANCE_RUNNING",
  "instance_id": "inst-12345",
  "cloud_instance_id": "i-0abc123",
  "status": "RUNNING",
  "launch_confirmed_at": "2025-11-27T10:32:15Z",
  "launch_duration_seconds": 135,
  "timestamp": "2025-11-27T10:32:15Z"
}
```

#### INSTANCE_TERMINATING
```json
{
  "event_type": "INSTANCE_TERMINATING",
  "instance_id": "inst-12345",
  "cloud_instance_id": "i-0abc123",
  "status": "TERMINATING",
  "termination_requested_at": "2025-11-27T11:00:00Z",
  "timestamp": "2025-11-27T11:00:00Z"
}
```

#### INSTANCE_TERMINATED
```json
{
  "event_type": "INSTANCE_TERMINATED",
  "instance_id": "inst-12345",
  "cloud_instance_id": "i-0abc123",
  "status": "TERMINATED",
  "termination_confirmed_at": "2025-11-27T11:02:30Z",
  "termination_duration_seconds": 150,
  "timestamp": "2025-11-27T11:02:30Z"
}
```

## API Endpoints

### GET /api/v1/instances
Get all active instances.

**Response:**
```json
{
  "instances": [
    {
      "instance_id": "inst-12345",
      "status": "RUNNING",
      "cloud_instance_id": "i-0abc123",
      "instance_type": "t3.medium",
      "availability_zone": "ap-south-1a",
      "launch_requested_at": "2025-11-27T10:30:00Z",
      "launch_confirmed_at": "2025-11-27T10:32:15Z",
      "launch_duration_seconds": 135
    }
  ]
}
```

### POST /api/v1/instances/{instance_id}/launch
Launch a new instance.

**Request:**
```json
{
  "instance_type": "t3.medium",
  "availability_zone": "ap-south-1a"
}
```

**Response:**
```json
{
  "instance_id": "inst-12345",
  "status": "LAUNCHING",
  "launch_requested_at": "2025-11-27T10:30:00Z"
}
```

### POST /api/v1/instances/{instance_id}/terminate
Terminate an instance.

**Request:**
```json
{
  "reason": "manual"
}
```

**Response:**
```json
{
  "instance_id": "inst-12345",
  "status": "TERMINATING",
  "termination_requested_at": "2025-11-27T11:00:00Z"
}
```

### GET /api/v1/instances/{instance_id}/history
Get lifecycle event history for an instance.

**Response:**
```json
{
  "events": [
    {
      "event_id": 1,
      "event_type": "LAUNCH_REQUESTED",
      "source": "backend",
      "reason": "manual",
      "created_at": "2025-11-27T10:30:00Z"
    },
    {
      "event_id": 2,
      "event_type": "LAUNCH_CONFIRMED",
      "source": "agent",
      "confirmation_method": "agent_aws_poll",
      "cloud_instance_id": "i-0abc123",
      "created_at": "2025-11-27T10:32:15Z"
    }
  ]
}
```

### WebSocket Connection
```
ws://backend-host/ws
```

Clients connect and receive real-time state updates for all instances.

## Technology Stack

### Backend
- **Framework**: FastAPI 0.100+
- **WebSocket**: FastAPI WebSockets
- **Database**: PostgreSQL 14+
- **ORM**: SQLAlchemy 2.0
- **Validation**: Pydantic v2
- **Async**: asyncio, asyncpg

### Agent
- **Language**: Python 3.11+
- **AWS SDK**: boto3
- **Async**: asyncio
- **Queue**: High-priority command queue
- **Communication**: WebSocket client or HTTP polling

### Frontend
- **Framework**: React 18+
- **Language**: TypeScript 5+
- **State**: Zustand
- **WebSocket**: native WebSocket API
- **HTTP**: Axios
- **UI**: TailwindCSS, Headless UI

### Database
- **Primary**: PostgreSQL 14+
- **Migrations**: Alembic
- **Connection Pool**: asyncpg

## Deployment

### Development
```bash
# Start all services
docker-compose up -d

# Backend: http://localhost:8000
# Frontend: http://localhost:3000
# Database: localhost:5432
```

### Production
- Backend: Deploy on AWS ECS or Kubernetes
- Frontend: Deploy on S3 + CloudFront or Vercel
- Agent: Deploy on EC2 or ECS
- Database: AWS RDS PostgreSQL

## Monitoring

### Metrics
- `launch_duration_p50`, `launch_duration_p99`
- `termination_duration_p50`, `termination_duration_p99`
- `launch_timeout_rate`, `termination_timeout_rate`
- `websocket_connections_active`
- `state_transition_rate`

### Alerts
- Launch duration exceeds 5 minutes
- Termination duration exceeds 10 minutes
- Instance stuck in LAUNCHING > 10 minutes
- Instance stuck in TERMINATING > 15 minutes
- High rate of launch/terminate failures

### Logging
- Structured JSON logs
- Log aggregation: CloudWatch Logs or ELK
- Trace IDs for request tracking
- Log levels: DEBUG, INFO, WARNING, ERROR

## Security

- **Authentication**: JWT tokens
- **Authorization**: Role-based access control (RBAC)
- **WebSocket**: Authenticated connections only
- **AWS**: IAM roles with least privilege
- **Database**: Encrypted at rest and in transit
- **API**: Rate limiting, input validation

## Next Steps

1. Implement database schema and migrations
2. Build backend state machine and API
3. Develop agent with AWS polling
4. Create frontend components
5. Write comprehensive tests
6. Set up monitoring and observability
7. Deploy to production

---

**Document Version**: 1.0
**Last Updated**: 2025-11-27
**Status**: Initial Implementation
