from app.models.base import get_db
from app.models.assessment import Question
from app.data.sample_questions import ROLES, SKILL_DIMENSIONS
from app.main import init_question_bank

def test_db():
    db = next(get_db())
    try:
        # Initialize the question bank
        init_question_bank(db)
        
        # Get total questions
        total = db.query(Question).count()
        print(f'Total questions in DB: {total}\n')
        
        # Get questions by role
        print('Questions by role:')
        for role in ROLES.values():
            count = db.query(Question).filter(Question.role == role).count()
            print(f'{role}: {count}')
        
        # Get questions by skill dimension
        print('\nQuestions by skill dimension:')
        for skill_dim in SKILL_DIMENSIONS.values():
            count = db.query(Question).filter(Question.skill_dimension == skill_dim).count()
            print(f'{skill_dim}: {count}')
            
    finally:
        db.close()

if __name__ == "__main__":
    test_db()
