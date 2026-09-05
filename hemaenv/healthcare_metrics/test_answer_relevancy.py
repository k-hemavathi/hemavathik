from deepeval.evaluate import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import AnswerRelevancyMetric
from app.healthcare_chatbot import chatbot


def test_answer_relevancy_metrics():
    prompt = "what is COVID-19 how effect is on me?"
    response = chatbot(prompt)

    test_case = LLMTestCase(
        input= prompt,
        actual_output= response
    )

    answer_relevancy = AnswerRelevancyMetric(threshold=0.7)

    result = evaluate([test_case], [answer_relevancy])
    print(result)