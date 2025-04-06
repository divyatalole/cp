# Tech Industry Competency Assessment Engine

A sophisticated assessment system for evaluating technical competencies across different roles in the tech industry.

## Features

- Support for multiple tech roles (Software Engineer, Data Scientist, DevOps Engineer)
- Adaptive questioning system
- Comprehensive competency scoring across 10 key skill dimensions
- Role-specific question sets with fixed lengths
- Detailed performance breakdown and analysis

## Technical Stack

- Backend: FastAPI + Python
- Database: PostgreSQL
- ORM: SQLAlchemy

## Competency Scoring System

- 0-20: Beginner level
- 21-40: Foundational level
- 41-60: Intermediate level
- 61-80: Advanced level
- 81-100: Expert level

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure PostgreSQL:
   - Create a database named 'competency_assessment'
   - Update database credentials in app/database.py

4. Run the application:
   ```bash
   uvicorn app.main:app --reload
   ```

## API Endpoints

- POST `/assessments/start`: Start a new assessment
- GET `/questions/{session_id}`: Get the next question
- POST `/responses/{session_id}`: Submit an answer
- GET `/results/{session_id}`: Get assessment results

## Project Structure

```
.
├── app/
│   ├── main.py          # FastAPI application
│   ├── models.py        # SQLAlchemy models
│   ├── schemas.py       # Pydantic schemas
│   └── database.py      # Database configuration
├── requirements.txt     # Project dependencies
└── README.md           # Project documentation
```
