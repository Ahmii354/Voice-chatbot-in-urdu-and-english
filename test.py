from app import query_rag
from langchain_community.llms.ollama import Ollama

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""


def test_author():
    assert query_and_validate(
        question="Who is the author? (Answer with 2 words)",
        expected_response="Imran Khan",
    )

def test_Birth():
    assert query_and_validate(
        question="In which year was Imran Khan born? (Answer with the number only)",
        expected_response="1952",
    )

def test_Jemima():
    assert query_and_validate(
        question="Who was the wife of Imran Khan? (Answer with no more than 2 words)",
        expected_response="Jemima Goldsmith",
    )

def test_party():
    assert query_and_validate(
        question="What was the name of Imran Khan's political party? (Answer with no more than 4 words)",
        expected_response="Tehreek-e-Insaf",
    )

def test_cricket(): # negative test case
    assert not query_and_validate(
        question="How long was Imran Khan's cricket career? (Answer with no more than 4 words)",
        expected_response="5 years long",
    )

def query_and_validate(question: str, expected_response: str):
    response_text = query_rag(question)

    # Ensure response_text is a string 
    if isinstance(response_text, tuple):
        response_text = response_text[0]  # Assuming the actual response is the first element of the tuple

    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )

    model = Ollama(model="mistral")
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    print(prompt)

    if "true" in evaluation_results_str_cleaned:
        # Print response in Green if it is correct.
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        # Print response in Red if it is incorrect.
        print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        ) 
