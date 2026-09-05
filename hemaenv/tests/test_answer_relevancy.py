from deepeval.evaluate import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import AnswerRelevancyMetric, GEval
from app.banking_chatbot import chatbot

def test_answer_relevancy_metrics():
    prompt = "What is Fixed Dipoasit"
    response = chatbot(prompt)

    test_case = LLMTestCase(
        input = prompt,
        actual_output= response
    )

    answer_relevancy = GEval(
        name = "Answer Relevancy",
        criteria= "Check if the answer is relevant",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT
        ],

        evaluation_steps=[
            "Check if the answer is relevent to the query",
        ],
        threshold=0.8
        
    )

    reult = ([test_case], [answer_relevancy])