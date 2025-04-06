from app.data.sample_questions import get_questions_for_role, get_question_by_difficulty, get_question_by_skill
from app.models import Role, SkillDimension

def test_questions():
    # Test getting all questions for software engineer
    se_questions = get_questions_for_role(Role.software_engineer)
    print(f"\nTotal Software Engineer questions: {len(se_questions)}")
    print(f"First question: {se_questions[0]['text']}")
    
    # Test getting questions by difficulty
    hard_questions = get_question_by_difficulty(Role.software_engineer, 4)
    print(f"\nDifficulty 4 questions: {len(hard_questions)}")
    if hard_questions:
        print(f"Example hard question: {hard_questions[0]['text']}")
    
    # Test getting questions by skill
    algo_questions = get_question_by_skill(Role.software_engineer, SkillDimension.algorithm_knowledge)
    print(f"\nAlgorithm questions: {len(algo_questions)}")
    if algo_questions:
        print(f"Example algorithm question: {algo_questions[0]['text']}")
    
    # Print unique skill dimensions
    skill_dimensions = set(q['skill_dimension'] for q in se_questions)
    print(f"\nUnique skill dimensions: {skill_dimensions}")
    
    # Print difficulty distribution
    difficulty_dist = {}
    for q in se_questions:
        difficulty_dist[q['difficulty']] = difficulty_dist.get(q['difficulty'], 0) + 1
    print(f"\nDifficulty distribution: {difficulty_dist}")

if __name__ == "__main__":
    test_questions()
