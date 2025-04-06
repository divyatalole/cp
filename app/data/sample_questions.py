# Role and skill dimension constants
ROLES = {
    'SOFTWARE_ENGINEER': 'software_engineer',
    'DATA_SCIENTIST': 'data_scientist',
    'DEVOPS_ENGINEER': 'devops_engineer'
}

SKILL_DIMENSIONS = {
    'ALGORITHM_KNOWLEDGE': 'algorithm_knowledge',
    'CODING_PROFICIENCY': 'coding_proficiency',
    'SYSTEM_DESIGN': 'system_design',
    'DATABASE_MANAGEMENT': 'database_management',
    'TESTING_QA': 'testing_qa',
    'DEVOPS_KNOWLEDGE': 'devops_knowledge',
    'CLOUD_SERVICES': 'cloud_services',
    'COMMUNICATION': 'communication',
    'DATA_ANALYSIS': 'data_analysis',
    'MACHINE_LEARNING': 'machine_learning'
}


def get_questions_for_role(role: str):
    """Retrieve all questions for a specific role."""
    return SAMPLE_QUESTIONS.get(role, [])


def get_question_by_difficulty(role: str, difficulty: int):
    """Get questions of a specific difficulty level for a role."""
    questions = SAMPLE_QUESTIONS.get(role, [])
    return [q for q in questions if q["difficulty"] == difficulty]


def get_question_by_skill(role: str, skill_dimension: str):
    """Get questions for a specific skill dimension within a role."""
    questions = SAMPLE_QUESTIONS.get(role, [])
    return [q for q in questions if q["skill_dimension"] == skill_dimension]


SAMPLE_QUESTIONS = {
    ROLES['SOFTWARE_ENGINEER']: [
        # Algorithm Knowledge Questions (8 questions)
        {
            "text": "What is the time complexity of QuickSort in the average case?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['ALGORITHM_KNOWLEDGE'],
            "options": {"a": "O(n)", "b": "O(n log n)", "c": "O(n²)", "d": "O(log n)"},
            "correct_answer": "b"
        },
        {
            "text": "Explain the concept of dynamic programming and when it should be used.",
            "difficulty": 4,
            "skill_dimension": SKILL_DIMENSIONS['ALGORITHM_KNOWLEDGE'],
            "options": {
                "a": "For problems with optimal substructure and overlapping subproblems",
                "b": "Only for sorting algorithms",
                "c": "For linear time algorithms only",
                "d": "When recursion is not possible"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is the time complexity of binary search?",
            "difficulty": 2,
            "skill_dimension": SKILL_DIMENSIONS['ALGORITHM_KNOWLEDGE'],
            "options": {"a": "O(n)", "b": "O(n log n)", "c": "O(n²)", "d": "O(log n)"},
            "correct_answer": "d"
        },
        {
            "text": "What algorithm would you use to find the shortest path in a weighted graph?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['ALGORITHM_KNOWLEDGE'],
            "options": {
                "a": "Breadth-First Search",
                "b": "Depth-First Search",
                "c": "Dijkstra's Algorithm",
                "d": "Bubble Sort"
            },
            "correct_answer": "c"
        },
        {
            "text": "What is the space complexity of a recursive implementation of the Fibonacci sequence?",
            "difficulty": 4,
            "skill_dimension": SKILL_DIMENSIONS['ALGORITHM_KNOWLEDGE'],
            "options": {"a": "O(1)", "b": "O(n)", "c": "O(n²)", "d": "O(2^n)"},
            "correct_answer": "b"
        },
        {
            "text": "Which data structure would you use to implement a priority queue most efficiently?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['ALGORITHM_KNOWLEDGE'],
            "options": {
                "a": "Array",
                "b": "Linked List",
                "c": "Binary Heap",
                "d": "Hash Table"
            },
            "correct_answer": "c"
        },
        {
            "text": "What's the primary difference between Greedy and Dynamic Programming algorithms?",
            "difficulty": 4,
            "skill_dimension": SKILL_DIMENSIONS['ALGORITHM_KNOWLEDGE'],
            "options": {
                "a": "Greedy makes locally optimal choices, DP considers all possible choices",
                "b": "Greedy is always faster than DP",
                "c": "DP is always more accurate than Greedy",
                "d": "Greedy uses memoization, DP doesn't"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is the time complexity of inserting an element into a balanced binary search tree?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['ALGORITHM_KNOWLEDGE'],
            "options": {"a": "O(1)", "b": "O(log n)", "c": "O(n)", "d": "O(n log n)"},
            "correct_answer": "b"
        },
        
        # Coding Proficiency Questions (8 questions)
        {
            "text": "Which algorithm is best suited to detect a cycle in a linked list?",
            "difficulty": 4,
            "skill_dimension": SKILL_DIMENSIONS['CODING_PROFICIENCY'],
            "options": {
                "a": "Floyd's cycle-finding algorithm (fast/slow pointers)",
                "b": "Binary search",
                "c": "Depth-first search",
                "d": "Linear search"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is the best way to implement a thread-safe singleton pattern in Python?",
            "difficulty": 5,
            "skill_dimension": SKILL_DIMENSIONS['CODING_PROFICIENCY'],
            "options": {
                "a": "Use metaclass with threading lock",
                "b": "Use global variable",
                "c": "Use class variable",
                "d": "Use static method"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is the purpose of the 'yield' keyword in Python?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['CODING_PROFICIENCY'],
            "options": {
                "a": "To create a generator function",
                "b": "To exit a function",
                "c": "To import modules",
                "d": "To define a class"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is a closure in JavaScript?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['CODING_PROFICIENCY'],
            "options": {
                "a": "A function that maintains access to its lexical scope",
                "b": "A way to close the browser window",
                "c": "A method to shut down a server",
                "d": "A type of loop"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is the difference between '==' and '===' operators in JavaScript?",
            "difficulty": 2,
            "skill_dimension": SKILL_DIMENSIONS['CODING_PROFICIENCY'],
            "options": {
                "a": "No difference, they are aliases",
                "b": "'===' checks both value and type, '==' only checks value",
                "c": "'==' is deprecated",
                "d": "'===' is only used in TypeScript"
            },
            "correct_answer": "b"
        },
        {
            "text": "What is the purpose of the 'virtual' keyword in C++?",
            "difficulty": 4,
            "skill_dimension": SKILL_DIMENSIONS['CODING_PROFICIENCY'],
            "options": {
                "a": "Memory optimization",
                "b": "Enable polymorphism in inheritance",
                "c": "Declare a variable without initialization",
                "d": "Create a virtual machine"
            },
            "correct_answer": "b"
        },
        {
            "text": "What's the difference between shallow copy and deep copy?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['CODING_PROFICIENCY'],
            "options": {
                "a": "Shallow copy duplicates references, deep copy duplicates values",
                "b": "Deep copy is always faster",
                "c": "Shallow copy works only on primitive types",
                "d": "There is no difference"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is dependency injection?",
            "difficulty": 4,
            "skill_dimension": SKILL_DIMENSIONS['CODING_PROFICIENCY'],
            "options": {
                "a": "A design pattern where dependencies are provided to objects",
                "b": "A method to inject code at runtime",
                "c": "A way to add libraries to a project",
                "d": "A technique to make code run faster"
            },
            "correct_answer": "a"
        },
        
        # System Design Questions (6 questions)
        {
            "text": "Design considerations for a distributed cache system:",
            "difficulty": 5,
            "skill_dimension": SKILL_DIMENSIONS['SYSTEM_DESIGN'],
            "options": {
                "a": "Consistency model",
                "b": "Cache invalidation strategy",
                "c": "Network latency",
                "d": "All of the above"
            },
            "correct_answer": "d"
        },
        {
            "text": "Best practices for designing a microservices architecture:",
            "difficulty": 4,
            "skill_dimension": SKILL_DIMENSIONS['SYSTEM_DESIGN'],
            "options": {
                "a": "Service discovery",
                "b": "API gateway",
                "c": "Circuit breakers",
                "d": "All of the above"
            },
            "correct_answer": "d"
        },
        {
            "text": "What are the key components of a URL shortening service design?",
            "difficulty": 4,
            "skill_dimension": SKILL_DIMENSIONS['SYSTEM_DESIGN'],
            "options": {
                "a": "Hashing algorithm, database storage, redirection mechanism",
                "b": "Only a good database",
                "c": "AI-powered URL analyzer",
                "d": "Social media integration"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is the CAP theorem in distributed systems?",
            "difficulty": 4,
            "skill_dimension": SKILL_DIMENSIONS['SYSTEM_DESIGN'],
            "options": {
                "a": "A system can have at most 2 of: Consistency, Availability, Partition tolerance",
                "b": "A method to measure system performance",
                "c": "A security protocol",
                "d": "A database design pattern"
            },
            "correct_answer": "a"
        },
        {
            "text": "How would you design a system that handles millions of concurrent users?",
            "difficulty": 5,
            "skill_dimension": SKILL_DIMENSIONS['SYSTEM_DESIGN'],
            "options": {
                "a": "Horizontal scaling with load balancing",
                "b": "Single powerful server",
                "c": "Only use caching",
                "d": "Blockchain technology"
            },
            "correct_answer": "a"
        },
        {
            "text": "What design pattern would you use for an event-driven architecture?",
            "difficulty": 4,
            "skill_dimension": SKILL_DIMENSIONS['SYSTEM_DESIGN'],
            "options": {
                "a": "Observer pattern",
                "b": "Singleton pattern",
                "c": "Factory pattern",
                "d": "Adapter pattern"
            },
            "correct_answer": "a"
        },
        
        # Database Management Questions (6 questions)
        {
            "text": "Explain ACID properties in database transactions.",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['DATABASE_MANAGEMENT'],
            "options": {
                "a": "Atomicity, Consistency, Isolation, Durability",
                "b": "Availability, Consistency, Isolation, Distribution",
                "c": "Atomicity, Concurrency, Integrity, Distribution",
                "d": "None of the above"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is database normalization?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['DATABASE_MANAGEMENT'],
            "options": {
                "a": "Process of organizing data to reduce redundancy",
                "b": "Process of converting SQL to NoSQL",
                "c": "Method to speed up queries",
                "d": "Technique to compress database size"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is the purpose of an index in a database?",
            "difficulty": 2,
            "skill_dimension": SKILL_DIMENSIONS['DATABASE_MANAGEMENT'],
            "options": {
                "a": "To speed up queries",
                "b": "To store backup data",
                "c": "To encrypt sensitive data",
                "d": "To connect to external systems"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is a NoSQL database best used for?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['DATABASE_MANAGEMENT'],
            "options": {
                "a": "Unstructured or semi-structured data",
                "b": "Financial transactions",
                "c": "Small datasets",
                "d": "Single-user applications"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is sharding in database systems?",
            "difficulty": 4,
            "skill_dimension": SKILL_DIMENSIONS['DATABASE_MANAGEMENT'],
            "options": {
                "a": "Horizontal partitioning of data",
                "b": "Encryption technique",
                "c": "A type of join operation",
                "d": "Database backup strategy"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is the difference between a left join and an inner join?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['DATABASE_MANAGEMENT'],
            "options": {
                "a": "Left join includes unmatched rows from the left table, inner join only includes matches",
                "b": "Inner join is faster than left join",
                "c": "Left join works only in MySQL",
                "d": "They are different names for the same operation"
            },
            "correct_answer": "a"
        },
        
        # Testing/QA Questions (4 questions)
        {
            "text": "What is the difference between unit testing and integration testing?",
            "difficulty": 2,
            "skill_dimension": SKILL_DIMENSIONS['TESTING_QA'],
            "options": {
                "a": "Unit tests are faster",
                "b": "Unit tests isolate components, integration tests check component interactions",
                "c": "Integration tests are always automated",
                "d": "Unit tests require more setup"
            },
            "correct_answer": "b"
        },
        {
            "text": "What is Test-Driven Development (TDD)?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['TESTING_QA'],
            "options": {
                "a": "Writing tests before implementation code",
                "b": "Testing only after development is complete",
                "c": "Using AI to generate tests",
                "d": "Outsourcing testing to third parties"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is a mocking framework used for?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['TESTING_QA'],
            "options": {
                "a": "To simulate dependencies in unit tests",
                "b": "To generate random test data",
                "c": "To create UI mockups",
                "d": "To test performance"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is the purpose of code coverage in testing?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['TESTING_QA'],
            "options": {
                "a": "To measure what percentage of code is executed by tests",
                "b": "To measure code quality",
                "c": "To count lines of code",
                "d": "To generate documentation"
            },
            "correct_answer": "a"
        },
        
        # DevOps Knowledge Questions (3 questions)
        {
            "text": "What is Continuous Integration?",
            "difficulty": 2,
            "skill_dimension": SKILL_DIMENSIONS['DEVOPS_KNOWLEDGE'],
            "options": {
                "a": "Practice of merging code changes frequently",
                "b": "Deploying to production continuously",
                "c": "Writing code without breaks",
                "d": "A testing methodology"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is the purpose of Docker containers?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['DEVOPS_KNOWLEDGE'],
            "options": {
                "a": "Package applications with dependencies",
                "b": "Replace virtual machines completely",
                "c": "Run only database systems",
                "d": "Store data securely"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is infrastructure as code?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['DEVOPS_KNOWLEDGE'],
            "options": {
                "a": "Managing infrastructure through code and automation",
                "b": "Writing code about infrastructure",
                "c": "Using physical servers",
                "d": "A coding methodology"
            },
            "correct_answer": "a"
        },
        
        # Cloud Services Questions (2 questions)
        {
            "text": "What is the difference between IaaS, PaaS, and SaaS?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['CLOUD_SERVICES'],
            "options": {
                "a": "Different levels of managed services",
                "b": "Different cloud providers",
                "c": "Different pricing models",
                "d": "Different security levels"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is the purpose of a CDN?",
            "difficulty": 2,
            "skill_dimension": SKILL_DIMENSIONS['CLOUD_SERVICES'],
            "options": {
                "a": "Delivering content to users with reduced latency",
                "b": "Storing database backups",
                "c": "Running serverless functions",
                "d": "Managing DNS records"
            },
            "correct_answer": "a"
        },
        
        # Communication Questions (3 questions)
        {
            "text": "How would you explain a complex technical concept to a non-technical stakeholder?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['COMMUNICATION'],
            "options": {
                "a": "Use analogies and real-world examples",
                "b": "Use technical jargon for precision",
                "c": "Send a detailed technical document",
                "d": "Recommend they learn programming"
            },
            "correct_answer": "a"
        },
        {
            "text": "What information should be included in a technical requirements document?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['COMMUNICATION'],
            "options": {
                "a": "Functional requirements, constraints, acceptance criteria",
                "b": "Only the technical implementation details",
                "c": "Just the deadline and budget",
                "d": "Programming language preferences"
            },
            "correct_answer": "a"
        },
        {
            "text": "How would you handle receiving contradicting requirements from different stakeholders?",
            "difficulty": 4,
            "skill_dimension": SKILL_DIMENSIONS['COMMUNICATION'],
            "options": {
                "a": "Facilitate a meeting to reach consensus",
                "b": "Implement both requirements",
                "c": "Choose the one from the highest-ranking stakeholder",
                "d": "Ignore both and implement what you think is best"
            },
            "correct_answer": "a"
        }
    ],

    ROLES['DATA_SCIENTIST']: [
        
        {
            "text": "What is the correct interpretation of the following correlation matrix output?\npython\n# Correlation matrix output\nimport pandas as pd\nimport numpy as np\n\ncorr_matrix = pd.DataFrame({\n    'feature1': [1.0, 0.92, 0.15],\n    'feature2': [0.92, 1.0, 0.21],\n    'feature3': [0.15, 0.21, 1.0]\n}, index=['feature1', 'feature2', 'feature3'])\nprint(corr_matrix)\n",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['DATA_ANALYSIS'],
            "options": {
                "a": "feature1 and feature2 show strong multicollinearity",
                "b": "All features are independent enough for modeling",
                "c": "feature3 has strong correlation with feature1",
                "d": "The matrix shows no concerning relationships"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is the purpose of feature scaling in machine learning?",
            "difficulty": 2,
            "skill_dimension": SKILL_DIMENSIONS['DATA_ANALYSIS'],
            "options": {
                "a": "To make features comparable",
                "b": "To improve model convergence",
                "c": "To prevent feature dominance",
                "d": "All of the above"
            },
            "correct_answer": "d"
        },
        {
            "text": "What is the difference between correlation and causation?",
            "difficulty": 2,
            "skill_dimension": SKILL_DIMENSIONS['DATA_ANALYSIS'],
            "options": {
                "a": "Correlation means one variable causes the other",
                "b": "Correlation shows relationship, causation indicates one variable causes change in another",
                "c": "They are the same concept",
                "d": "Causation is always stronger than correlation"
            },
            "correct_answer": "b"
        },
        {
            "text": "What is the purpose of outlier detection?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['DATA_ANALYSIS'],
            "options": {
                "a": "Identify anomalous data points",
                "b": "Increase model accuracy",
                "c": "Understand data distribution",
                "d": "All of the above"
            },
            "correct_answer": "d"
        },
        {
            "text": "What is the central limit theorem?",
            "difficulty": 4,
            "skill_dimension": SKILL_DIMENSIONS['DATA_ANALYSIS'],
            "options": {
                "a": "Distribution of sample means approaches normal distribution",
                "b": "All data eventually becomes normally distributed",
                "c": "The center of any distribution contains the most data",
                "d": "A method to find the central tendency"
            },
            "correct_answer": "a"
        },
        {
            "text": "What technique would you use to identify the most important features in a dataset?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['DATA_ANALYSIS'],
            "options": {
                "a": "Feature importance from tree-based models",
                "b": "Chi-square test",
                "c": "Correlation analysis",
                "d": "All of the above"
            },
            "correct_answer": "d"
        },
        {
            "text": "What is the purpose of A/B testing?",
            "difficulty": 2,
            "skill_dimension": SKILL_DIMENSIONS['DATA_ANALYSIS'],
            "options": {
                "a": "Compare two variants to determine which performs better",
                "b": "Test if data follows normal distribution",
                "c": "Verify database integrity",
                "d": "Measure algorithm efficiency"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is the difference between parametric and non-parametric statistical tests?",
            "difficulty": 4,
            "skill_dimension": SKILL_DIMENSIONS['DATA_ANALYSIS'],
            "options": {
                "a": "Parametric tests assume specific distribution, non-parametric don't",
                "b": "Non-parametric tests are always more accurate",
                "c": "Parametric tests work only with categorical data",
                "d": "They differ only in computational complexity"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is the purpose of dimensionality reduction?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['DATA_ANALYSIS'],
            "options": {
                "a": "Reduce computational complexity",
                "b": "Address curse of dimensionality",
                "c": "Improve visualization",
                "d": "All of the above"
            },
            "correct_answer": "d"
        },
        {
            "text": "What is the difference between Type I and Type II errors?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['DATA_ANALYSIS'],
            "options": {
                "a": "Type I is false positive, Type II is false negative",
                "b": "Type I is more serious than Type II",
                "c": "Type II is a software error",
                "d": "They are the same, just different names"
            },
            "correct_answer": "a"
        },
        {
            "text": "What would be the issue with the following model training code?\npython\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.linear_model import LogisticRegression\n\n# Split data\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)\n\n# Scale features\nscaler = StandardScaler()\nX_train = scaler.fit_transform(X_train)\nX_test = scaler.fit_transform(X_test)  # Issue is here\n\n# Train model\nmodel = LogisticRegression()\nmodel.fit(X_train, y_train)\nscore = model.score(X_test, y_test)\n",
            "difficulty": 4,
            "skill_dimension": SKILL_DIMENSIONS['MACHINE_LEARNING'],
            "options": {
                "a": "The test data is being fit with a new scaler, causing data leakage",
                "b": "LogisticRegression is not appropriate for this data",
                "c": "The test size is too small",
                "d": "The code is missing cross-validation"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is the purpose of dropout in neural networks?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['MACHINE_LEARNING'],
            "options": {
                "a": "Prevent overfitting",
                "b": "Speed up training",
                "c": "Reduce model size",
                "d": "All of the above"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is gradient descent?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['MACHINE_LEARNING'],
            "options": {
                "a": "Optimization algorithm to minimize loss function",
                "b": "A type of neural network",
                "c": "A method to clean data",
                "d": "A sampling technique"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is the difference between supervised and unsupervised learning?",
            "difficulty": 2,
            "skill_dimension": SKILL_DIMENSIONS['MACHINE_LEARNING'],
            "options": {
                "a": "Supervised uses labeled data, unsupervised doesn't",
                "b": "Supervised is always more accurate",
                "c": "Unsupervised requires more data",
                "d": "They use different algorithms"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is the purpose of cross-validation?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['MACHINE_LEARNING'],
            "options": {
                "a": "Evaluate model performance on unseen data",
                "b": "Speed up training",
                "c": "Reduce model complexity",
                "d": "Ensure data quality"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is a neural network activation function?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['MACHINE_LEARNING'],
            "options": {
                "a": "Introduces non-linearity to the model",
                "b": "Controls how many neurons are active",
                "c": "Determines learning rate",
                "d": "Sets network topology"
            },
            "correct_answer": "a"
        },
        {
            "text": "How does a Random Forest algorithm work?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['MACHINE_LEARNING'],
            "options": {
                "a": "Builds multiple decision trees and merges results",
                "b": "Randomly selects data points",
                "c": "Creates a single optimized tree",
                "d": "Uses random hyperparameters"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is the purpose of the ROC curve?",
            "difficulty": 4,
            "skill_dimension": SKILL_DIMENSIONS['MACHINE_LEARNING'],
            "options": {
                "a": "Evaluate classifier performance across thresholds",
                "b": "Determine optimal learning rate",
                "c": "Visualize data distribution",
                "d": "Optimize network architecture"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is transfer learning?",
            "difficulty": 4,
            "skill_dimension": SKILL_DIMENSIONS['MACHINE_LEARNING'],
            "options": {
                "a": "Using pre-trained models for new tasks",
                "b": "Transferring data between databases",
                "c": "Moving models to production",
                "d": "Converting models between frameworks"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is a confusion matrix used for?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['MACHINE_LEARNING'],
            "options": {
                "a": "Evaluate classification model performance",
                "b": "Visualize neural network layers",
                "c": "Show feature correlations",
                "d": "Determine optimal cluster count"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is the difference between bagging and boosting?",
            "difficulty": 4,
            "skill_dimension": SKILL_DIMENSIONS['MACHINE_LEARNING'],
            "options": {
                "a": "Bagging trains in parallel, boosting trains sequentially",
                "b": "Boosting is always more accurate",
                "c": "Bagging works only with decision trees",
                "d": "They are different names for the same concept"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is regularization in machine learning?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['MACHINE_LEARNING'],
            "options": {
                "a": "Technique to prevent overfitting",
                "b": "Method to normalize data",
                "c": "Process to speed up training",
                "d": "Procedure to initialize weights"
            },
            "correct_answer": "a"
        },
        {
            "text": "What will be the output of the following pandas code?\npython\nimport pandas as pd\n\ndf = pd.DataFrame({\n    'A': [1, 2, 3, 4, 5],\n    'B': [10, 20, 30, 40, 50]\n})\n\nresult = df.apply(lambda x: x.max() - x.min())\nprint(result)\n",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['CODING_PROFICIENCY'],
            "options": {
                "a": "A    4\nB   40\ndtype: int64",
                "b": "0    9\n1   18\n2   27\n3   36\n4   45\ndtype: int64",
                "c": "4",
                "d": "An error, because apply cannot be used this way"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is the difference between map and reduce functions?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['CODING_PROFICIENCY'],
            "options": {
                "a": "Map transforms elements, reduce aggregates elements",
                "b": "Map is faster than reduce",
                "c": "Reduce can only be used on numeric data",
                "d": "They are interchangeable"
            },
            "correct_answer": "a"
        },
        {
            "text": "What are Python generators used for?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['CODING_PROFICIENCY'],
            "options": {
                "a": "Memory-efficient iteration",
                "b": "Creating GUI elements",
                "c": "Generating random numbers",
                "d": "Parallel processing"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is vectorization in numerical computing?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['CODING_PROFICIENCY'],
            "options": {
                "a": "Optimizing operations to work on arrays instead of loops",
                "b": "Creating vector graphics",
                "c": "Converting text to vectors",
                "d": "A form of parallelization"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is the difference between .loc and .iloc in pandas?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['CODING_PROFICIENCY'],
            "options": {
                "a": ".loc uses labels, .iloc uses integer positions",
                "b": ".iloc is deprecated",
                "c": ".loc is faster than .iloc",
                "d": "They are identical in functionality"
            },
            "correct_answer": "a"
        },
        {
            "text": "What would this SQL query return when executed against a typical e-commerce database?\n```sql\nSELECT \n    c.customer_name,\n    COUNT(o.order_id) as order_count,\n    SUM(o.total_amount) as total_spent\nFROM customers c\nLEFT JOIN orders o ON c.customer_id = o.customer_id\nGROUP BY c.customer_id, c.customer_name\nHAVING COUNT(o.order_id) > 0\nORDER BY total_spent DESC\nLIMIT 10;\n```",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['DATABASE_MANAGEMENT'],
            "options": {
                "a": "Top 10 customers by total spending who made at least one order",
                "b": "All customers who made at least one order",
                "c": "Top 10 customers by order count",
                "d": "First 10 customers in the database"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is the purpose of database indexing in data science applications?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['DATABASE_MANAGEMENT'],
            "options": {
                "a": "Speed up query performance",
                "b": "Reduce storage size",
                "c": "Improve data visualization",
                "d": "Enable machine learning capabilities"
            },
            "correct_answer": "a"
        },
        {
            "text": "Which database type is most suitable for storing unstructured data?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['DATABASE_MANAGEMENT'],
            "options": {
                "a": "NoSQL databases",
                "b": "Relational databases",
                "c": "In-memory databases",
                "d": "Time-series databases"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is the purpose of database normalization?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['DATABASE_MANAGEMENT'],
            "options": {
                "a": "Minimize redundancy and dependency",
                "b": "Maximize query performance",
                "c": "Enable parallel processing",
                "d": "Simplify database creation"
            },
            "correct_answer": "a"
        },
        {
            "text": "Which visualization code would be most appropriate for communicating the relationship between customer age groups and purchase amounts to business stakeholders?\n```python\n# Option A\nplt.figure(figsize=(10, 6))\nsns.boxplot(x='age_group', y='purchase_amount', data=customer_data)\nplt.title('Purchase Amount Distribution by Age Group')\nplt.xlabel('Customer Age Group')\nplt.ylabel('Purchase Amount ($)')\nplt.xticks(rotation=0)\n\n# Option B\nplt.figure(figsize=(10, 6))\ncustomer_data.groupby('age_group')['purchase_amount'].mean().plot(kind='bar')\nplt.title('Average Purchase Amount by Age Group')\nplt.xlabel('Customer Age Group')\nplt.ylabel('Avg Purchase Amount ($)')\nplt.xticks(rotation=0)\n\n# Option C\nplt.figure(figsize=(10, 6))\nsns.heatmap(customer_data.pivot_table(index='age_group', \n                                    columns='purchase_category', \n                                    values='purchase_amount'), \n          annot=True, cmap='viridis')\nplt.title('Purchase Amount by Age Group and Category')\n\n# Option D\nplt.figure(figsize=(10, 6))\nsns.lineplot(x='purchase_timestamp', y='purchase_amount', \n            hue='age_group', data=customer_data)\nplt.title('Purchase Trends Over Time by Age Group')\nplt.xlabel('Time')\nplt.ylabel('Purchase Amount ($)')\n```",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['COMMUNICATION'],
            "options": {
                "a": "Option A - Boxplot showing distribution details",
                "b": "Option B - Bar chart showing averages",
                "c": "Option C - Heatmap showing category breakdown",
                "d": "Option D - Line plot showing time trends"
            },
            "correct_answer": "b"
        },
        {
            "text": "How would you explain a complex machine learning model to non-technical stakeholders?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['COMMUNICATION'],
            "options": {
                "a": "Use analogies and visualizations",
                "b": "Show code examples",
                "c": "Use technical terms for precision",
                "d": "Skip the explanation and focus on results"
            },
            "correct_answer": "a"
        },
        {
            "text": "What information should be included in a data science project report?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['COMMUNICATION'],
            "options": {
                "a": "Problem statement, methodology, results, limitations",
                "b": "Only positive findings",
                "c": "Technical details without context",
                "d": "Code samples and raw data"
            },
            "correct_answer": "a"
        },
        {
            "text": "How would you communicate model uncertainty to business stakeholders?",
            "difficulty": 4,
            "skill_dimension": SKILL_DIMENSIONS['COMMUNICATION'],
            "options": {
                "a": "Use confidence intervals and scenario analysis",
                "b": "Avoid mentioning uncertainty",
                "c": "Use technical statistical terminology",
                "d": "Display p-values without explanation"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is the most effective way to present technical findings to executives?",
            "difficulty": 4,
            "skill_dimension": SKILL_DIMENSIONS['COMMUNICATION'],
            "options": {
                "a": "Focus on business impact and actionable insights",
                "b": "Show all technical details of the analysis",
                "c": "Present as many data visualizations as possible",
                "d": "Explain the code implementation"
            },
            "correct_answer": "a"
        },
        {
            "text": "What issue might occur with the following Dockerfile for a machine learning service?\ndockerfile\nFROM python:3.9\n\nWORKDIR /app\n\nCOPY requirements.txt .\nRUN pip install -r requirements.txt\n\nCOPY . .\n\nCMD [\"python\", \"model_server.py\"]\n",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['DEVOPS_KNOWLEDGE'],
            "options": {
                "a": "Inefficient layer caching - code changes will trigger reinstalling dependencies",
                "b": "The base image is inappropriate for ML workloads",
                "c": "Missing EXPOSE statement for the server port",
                "d": "The WORKDIR should not be set to /app"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is MLOps?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['DEVOPS_KNOWLEDGE'],
            "options": {
                "a": "Practices for ML lifecycle management and deployment",
                "b": "A machine learning framework",
                "c": "A specific model architecture",
                "d": "A data storage solution"
            },
            "correct_answer": "a"
        },
        {
            "text": "What would this AWS CLI command accomplish?\nbash\naws s3 cp large_dataset.csv s3://my-ml-bucket/data/ \\\n    && aws sagemaker create-training-job \\\n    --training-job-name \"model-training-job\" \\\n    --algorithm-specification \\\n        TrainingImage=123456789012.dkr.ecr.us-west-2.amazonaws.com/xgboost:1.0-1 \\\n    --role-arn arn:aws:iam::123456789012:role/SageMakerRole \\\n    --input-data-config \\\n        ChannelName=train,DataSource={S3DataSource={S3Uri=s3://my-ml-bucket/data/}} \\\n    --output-data-config S3OutputPath=s3://my-ml-bucket/output/ \\\n    --resource-config InstanceType=ml.m5.xlarge,InstanceCount=1 \\\n    --stopping-condition MaxRuntimeInSeconds=3600\n",
            "difficulty": 4,
            "skill_dimension": SKILL_DIMENSIONS['CLOUD_SERVICES'],
            "options": {
                "a": "Upload data to S3 and start a SageMaker training job using XGBoost",
                "b": "Create a copy of an existing training job",
                "c": "Deploy a pre-trained model to a SageMaker endpoint",
                "d": "Export training results from SageMaker to S3"
            },
            "correct_answer": "a"
        },
        {
            "text": "What are the benefits of cloud-based machine learning services?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['CLOUD_SERVICES'],
            "options": {
                "a": "Scalability, managed infrastructure, pre-trained models",
                "b": "Always cheaper than on-premises",
                "c": "Perfect security",
                "d": "No need for data preparation"
            },
            "correct_answer": "a"
        }
    ],

    ROLES['DEVOPS_ENGINEER']: [
        
        {
            "text": "Explain blue-green deployment strategy:",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['DEVOPS_KNOWLEDGE'],
            "options": {
                "a": "Load balancing technique",
                "b": "Zero-downtime deployment method",
                "c": "Testing strategy",
                "d": "Monitoring solution"
            },
            "correct_answer": "b"
        },
        {
            "text": "What is the purpose of Kubernetes StatefulSets?",
            "difficulty": 4,
            "skill_dimension": SKILL_DIMENSIONS['DEVOPS_KNOWLEDGE'],
            "options": {
                "a": "Managing stateful applications",
                "b": "Providing stable network identities",
                "c": "Ordered deployment and scaling",
                "d": "All of the above"
            },
            "correct_answer": "d"
        },
        {
            "text": "What is the difference between Docker containers and virtual machines?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['DEVOPS_KNOWLEDGE'],
            "options": {
                "a": "Containers share OS kernel, VMs have complete OS",
                "b": "Containers are always faster",
                "c": "VMs are deprecated technology",
                "d": "Containers work on any infrastructure"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is the purpose of a service mesh?",
            "difficulty": 4,
            "skill_dimension": SKILL_DIMENSIONS['DEVOPS_KNOWLEDGE'],
            "options": {
                "a": "Manage service-to-service communication",
                "b": "Connect to external APIs",
                "c": "Optimize database performance",
                "d": "Create web interfaces"
            },
            "correct_answer": "a"
        },
        {
            "text": "What's wrong with this Dockerfile for a Node.js application?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['DEVOPS_KNOWLEDGE'],
            "options": {
                "a": "The FROM instruction is incorrect",
                "b": "npm install is run before copying package.json",
                "c": "The EXPOSE instruction is unnecessary",
                "d": "The CMD instruction format is wrong"
            },
            "code": "FROM node:14\nWORKDIR /app\nRUN npm install\nCOPY . .\nEXPOSE 3000\nCMD [\"npm\", \"start\"]",
            "correct_answer": "b"
        },
        {
            "text": "What is a Kubernetes Ingress controller?",
            "difficulty": 4,
            "skill_dimension": SKILL_DIMENSIONS['DEVOPS_KNOWLEDGE'],
            "options": {
                "a": "Manages external access to services",
                "b": "Controls cluster auto-scaling",
                "c": "Manages pod scheduling",
                "d": "Handles data persistence"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is GitOps?",
            "difficulty": 4,
            "skill_dimension": SKILL_DIMENSIONS['DEVOPS_KNOWLEDGE'],
            "options": {
                "a": "Infrastructure and app management using Git as source of truth",
                "b": "A Git feature for operations teams",
                "c": "GUI tools for Git",
                "d": "A GitHub subscription service"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is the purpose of Terraform state?",
            "difficulty": 4,
            "skill_dimension": SKILL_DIMENSIONS['DEVOPS_KNOWLEDGE'],
            "options": {
                "a": "Track managed infrastructure and configuration",
                "b": "Monitor application health",
                "c": "Test infrastructure components",
                "d": "Replace version control"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is canary deployment?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['DEVOPS_KNOWLEDGE'],
            "options": {
                "a": "Gradually releasing to subset of users",
                "b": "Deploying exclusively to test environments",
                "c": "Releasing only critical security updates",
                "d": "Complete system replacement"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is the difference between a Dockerfile and docker-compose.yml?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['DEVOPS_KNOWLEDGE'],
            "options": {
                "a": "Dockerfile defines a single container, compose defines multi-container apps",
                "b": "Compose is deprecated in favor of Dockerfile",
                "c": "Dockerfile is for development, compose is for production",
                "d": "They serve the same purpose with different syntax"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is chaos engineering?",
            "difficulty": 4,
            "skill_dimension": SKILL_DIMENSIONS['DEVOPS_KNOWLEDGE'],
            "options": {
                "a": "Deliberately injecting failures to test resilience",
                "b": "Writing code without planning",
                "c": "Using random configurations",
                "d": "Testing without documentation"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is a sidecar container pattern?",
            "difficulty": 4,
            "skill_dimension": SKILL_DIMENSIONS['DEVOPS_KNOWLEDGE'],
            "options": {
                "a": "Helper container that extends main container functionality",
                "b": "Backup container for failover",
                "c": "Container for legacy applications",
                "d": "Database container configuration"
            },
            "correct_answer": "a"
        },
        {
            "text": "Best practices for AWS S3 bucket security:",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['CLOUD_SERVICES'],
            "options": {
                "a": "Enable encryption",
                "b": "Use bucket policies",
                "c": "Enable versioning",
                "d": "All of the above"
            },
            "correct_answer": "d"
        },
        {
            "text": "When to use AWS Lambda vs. ECS:",
            "difficulty": 4,
            "skill_dimension": SKILL_DIMENSIONS['CLOUD_SERVICES'],
            "options": {
                "a": "Lambda for event-driven, ECS for long-running",
                "b": "Lambda is always cheaper",
                "c": "ECS is always more reliable",
                "d": "They serve the same purpose"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is cloud-native architecture?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['CLOUD_SERVICES'],
            "options": {
                "a": "Designed to exploit cloud capabilities and services",
                "b": "Running any application in the cloud",
                "c": "Using only AWS services",
                "d": "Running containerized applications"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is the difference between IaaS, PaaS, and SaaS?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['CLOUD_SERVICES'],
            "options": {
                "a": "Different levels of managed services and abstraction",
                "b": "Different cloud providers",
                "c": "Different pricing models",
                "d": "Different security levels"
            },
            "correct_answer": "a"
        },
        {
            "text": "Which AWS CLI command correctly creates an auto scaling group with minimum 2 and maximum 5 instances?",
            "difficulty": 4,
            "skill_dimension": SKILL_DIMENSIONS['CLOUD_SERVICES'],
            "options": {
                "a": "aws autoscaling create-auto-scaling-group --auto-scaling-group-name my-asg --min-size 2 --max-size 5 --launch-configuration-name my-lc",
                "b": "aws ec2 create-auto-scaling-group --name my-asg --min-size 2 --max-size 5 --launch-configuration my-lc",
                "c": "aws asg create --name my-asg --min 2 --max 5 --launch-config my-lc",
                "d": "aws autoscaling create --group-name my-asg --min-instances 2 --max-instances 5 --launch-config my-lc"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is a CDN and when would you use it?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['CLOUD_SERVICES'],
            "options": {
                "a": "Distributed servers that deliver content based on user location",
                "b": "Container deployment network",
                "c": "Cloud database network",
                "d": "Continuous delivery nodes"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is the shared responsibility model in cloud computing?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['CLOUD_SERVICES'],
            "options": {
                "a": "Division of security responsibilities between provider and customer",
                "b": "Sharing cloud resources among multiple customers",
                "c": "Cost-sharing arrangement",
                "d": "Resource allocation methodology"
            },
            "correct_answer": "a"
        },
        {
            "text": "What's the difference between vertical and horizontal scaling?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['CLOUD_SERVICES'],
            "options": {
                "a": "Vertical adds more power, horizontal adds more instances",
                "b": "Horizontal is always more expensive",
                "c": "Vertical is always more reliable",
                "d": "They are different terms for the same concept"
            },
            "correct_answer": "a"
        },
        {
            "text": "Design considerations for a CI/CD pipeline:",
            "difficulty": 4,
            "skill_dimension": SKILL_DIMENSIONS['SYSTEM_DESIGN'],
            "options": {
                "a": "Automated testing",
                "b": "Environment consistency",
                "c": "Rollback capability",
                "d": "All of the above"
            },
            "correct_answer": "d"
        },
        {
            "text": "What architectural pattern would you use for a highly available system?",
            "difficulty": 4,
            "skill_dimension": SKILL_DIMENSIONS['SYSTEM_DESIGN'],
            "options": {
                "a": "Multi-region deployment with failover",
                "b": "Single powerful server",
                "c": "Microservices without redundancy",
                "d": "Only focusing on database replication"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is the Circuit Breaker pattern?",
            "difficulty": 4,
            "skill_dimension": SKILL_DIMENSIONS['SYSTEM_DESIGN'],
            "options": {
                "a": "Prevents cascading failures in distributed systems",
                "b": "A type of electrical protection",
                "c": "A network security feature",
                "d": "A database backup strategy"
            },
            "correct_answer": "a"
        },
        {
            "text": "What are key considerations for designing a scalable system?",
            "difficulty": 4,
            "skill_dimension": SKILL_DIMENSIONS['SYSTEM_DESIGN'],
            "options": {
                "a": "Stateless architecture, caching, load balancing",
                "b": "Using a single database server",
                "c": "Avoiding cloud services",
                "d": "Using monolithic design"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is the purpose of an API Gateway?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['SYSTEM_DESIGN'],
            "options": {
                "a": "Single entry point for client requests to backend services",
                "b": "Connecting databases only",
                "c": "User authentication only",
                "d": "Load balancing only"
            },
            "correct_answer": "a"
        },
        {
            "text": "Which code snippet correctly publishes a message to an AWS SQS queue?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['SYSTEM_DESIGN'],
            "options": {
                "a": "import boto3\nsqs = boto3.client('sqs')\nresponse = sqs.send_message(\n    QueueUrl='https://sqs.us-east-1.amazonaws.com/123456789012/my-queue',\n    MessageBody='Hello World'\n)",
                "b": "import boto3\nsqs = boto3.client('sqs')\nresponse = sqs.publish_message(\n    Queue='https://sqs.us-east-1.amazonaws.com/123456789012/my-queue',\n    Message='Hello World'\n)",
                "c": "import boto3\nsqs = boto3.resource('sqs')\nqueue = sqs.get_queue_by_name(QueueName='my-queue')\nresponse = queue.publish(\n    MessageBody='Hello World'\n)",
                "d": "import boto3\nsqs = boto3.resource('sqs')\nqueue = sqs.get_queue('my-queue')\nresponse = queue.send(\n    Body='Hello World'\n)"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is database sharding?",
            "difficulty": 4,
            "skill_dimension": SKILL_DIMENSIONS['DATABASE_MANAGEMENT'],
            "options": {
                "a": "Horizontal partitioning of data across multiple databases",
                "b": "Database compression technique",
                "c": "Database security feature",
                "d": "A type of database backup"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is the CAP theorem?",
            "difficulty": 4,
            "skill_dimension": SKILL_DIMENSIONS['DATABASE_MANAGEMENT'],
            "options": {
                "a": "Cannot simultaneously guarantee consistency, availability, partition tolerance",
                "b": "Cost, applications, performance balance principle",
                "c": "Core architecture principles",
                "d": "Cloud architecture principles"
            },
            "correct_answer": "a"
        },
        {
            "text": "When would you use a NoSQL database over a relational database?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['DATABASE_MANAGEMENT'],
            "options": {
                "a": "Unstructured data, high write loads, horizontal scaling needs",
                "b": "All financial applications",
                "c": "When SQL knowledge is limited",
                "d": "For all web applications"
            },
            "correct_answer": "a"
        },
        {
            "text": "Which SQL statement correctly creates an index on the 'email' column of the 'users' table?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['DATABASE_MANAGEMENT'],
            "options": {
                "a": "CREATE INDEX idx_email ON users (email);",
                "b": "ADD INDEX idx_email ON users (email);",
                "c": "ALTER TABLE users ADD INDEX ON email;",
                "d": "INDEX CREATE idx_email ON users.email;"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is database replication?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['DATABASE_MANAGEMENT'],
            "options": {
                "a": "Copying data from one database to another",
                "b": "Compressing database size",
                "c": "Converting between database types",
                "d": "Debugging database errors"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is the purpose of smoke testing?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['TESTING_QA'],
            "options": {
                "a": "Verify critical functionality works",
                "b": "Test for fire hazards",
                "c": "Comprehensive regression testing",
                "d": "Performance benchmarking"
            },
            "correct_answer": "a"
        },
        {
            "text": "What is the difference between unit testing and integration testing?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['TESTING_QA'],
            "options": {
                "a": "Unit tests test individual components, integration tests test interactions",
                "b": "Unit tests are always automated",
                "c": "Integration tests are always manual",
                "d": "They are the same with different names"
            },
            "correct_answer": "a"
        },
        {
            "text": "What's wrong with this shell script for automated testing?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['TESTING_QA'],
            "options": {
                "a": "The `set -e` flag exits on error, making the if statement redundant",
                "b": "The script doesn't specify which tests to run",
                "c": "The exit code should be 0 for failures",
                "d": "There's no proper error handling"
            },
            "code": "#!/bin/bash\nset -e\nnpm test\nif [ $? -eq 0 ]; then\n  echo \"Tests passed\"\nelse\n  echo \"Tests failed\"\n  exit 1\nfi",
            "correct_answer": "a"
        },
        {
            "text": "What is the purpose of load testing?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['TESTING_QA'],
            "options": {
                "a": "Assess system behavior under expected and peak loads",
                "b": "Test weight limitations",
                "c": "Test only database performance",
                "d": "Check for memory leaks"
            },
            "correct_answer": "a"
        },
        {
            "text": "How would you explain a complex infrastructure issue to non-technical stakeholders?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['COMMUNICATION'],
            "options": {
                "a": "Use analogies and focus on business impact",
                "b": "Use detailed technical terminology",
                "c": "Show code and logs",
                "d": "Avoid explaining and just fix it"
            },
            "correct_answer": "a"
        },
        {
            "text": "What should be included in an incident post-mortem report?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['COMMUNICATION'],
            "options": {
                "a": "Timeline, root cause, impact, preventive measures",
                "b": "Only technical details",
                "c": "Who to blame for the incident",
                "d": "Just resolution steps"
            },
            "correct_answer": "a"
        },
        {
            "text": "How would you handle conflicting requirements from security and development teams?",
            "difficulty": 4,
            "skill_dimension": SKILL_DIMENSIONS['COMMUNICATION'],
            "options": {
                "a": "Facilitate discussion to find compromise balancing both concerns",
                "b": "Always prioritize security",
                "c": "Always prioritize development speed",
                "d": "Escalate to management without attempting resolution"
            },
            "correct_answer": "a"
        },
        {
            "text": "Which Terraform code snippet correctly sets up an AWS S3 bucket with versioning enabled?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['CODING_PROFICIENCY'],
            "options": {
                "a": "resource \"aws_s3_bucket\" \"example\" {\n  bucket = \"my-bucket\"\n  versioning = true\n}",
                "b": "resource \"aws_s3_bucket\" \"example\" {\n  bucket = \"my-bucket\"\n  versioning {\n    enabled = true\n  }\n}",
                "c": "resource \"aws_s3_bucket\" \"example\" {\n  bucket = \"my-bucket\"\n}\nresource \"aws_s3_bucket_versioning\" \"versioning\" {\n  bucket = aws_s3_bucket.example.id\n  versioning_configuration {\n    status = \"Enabled\"\n  }\n}",
                "d": "resource \"aws_s3_bucket\" \"example\" {\n  bucket = \"my-bucket\"\n  enable_versioning = true\n}"
            },
            "correct_answer": "c"
        },
        {
            "text": "Which shell script correctly backs up all files in the current directory to a timestamped archive?",
            "difficulty": 3,
            "skill_dimension": SKILL_DIMENSIONS['CODING_PROFICIENCY'],
            "options": {
                "a": "#!/bin/bash\nBACKUP_DIR=\"/backups\"\nTIMESTAMP=$(date +%Y%m%d_%H%M%S)\ntar -czf $BACKUP_DIR/backup_$TIMESTAMP.tar.gz .",
                "b": "#!/bin/bash\nBACKUP_DIR=\"/backups\"\nTIMESTAMP=`date +%Y%m%d_%H%M%S`\ntar -czf $BACKUP_DIR/backup_$TIMESTAMP.tar.gz *",
                "c": "#!/bin/bash\nBACKUP_DIR=\"/backups\"\nTIMESTAMP=$(date +%Y%m%d_%H%M%S)\nmkdir -p $BACKUP_DIR\ntar -czf $BACKUP_DIR/backup_$TIMESTAMP.tar.gz .",
                "d": "#!/bin/bash\ndate=`date +%Y%m%d`\nzip -r backup_$date.zip ."
            },
            "correct_answer": "c"
        }
    ]
 }