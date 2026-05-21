# test_agent.py
from agent import run

test_questions = [
    "What is 3847 divided by 47 and what is the square root of that result?",
    "How does RAG reduce hallucinations?",
    "What are the latest developments in AI agents?",
]

for q in test_questions:
    print(f"\n{'='*60}")
    print(f"Question: {q}")
    print('='*60)
    result = run(q)
    print(f"\nFinal Answer:\n{result['answer']}")
    print(f"\nCompleted in {result['steps']} steps")
    print(f"\nReasoning trace:")
    for step in result['trace']:
        print(f"  Step {step['step']}: {step['action'] or 'ANSWER'}")
        if step['thought']:
            print(f"    Thought: {step['thought'][:100]}...")