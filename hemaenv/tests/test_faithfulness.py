from deepeval.evaluate import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import FaithfulnessMetric, GEval
from app.banking_chatbot import chatbot

retrival_context = [
    "Reset using ATM Machine",
    "ATM PIN Can be reset  via internetBanking",
    "Registered mobile number is required for OTP Verification",
    "You can not connect with customer support to change the PIN"

]

def test_faithfulness_metrics():
    prompt = "How can i reset my ATM PIN"
    output = chatbot(prompt)
    print(output)

    test_case = LLMTestCase(
        input= prompt,
        actual_output= output,
        retrieval_context=retrival_context,
        context=retrival_context

    )

    faithfulness_metrics = GEval(
        name = "Faithfullness",
        criteria= "Validate if all the information inside the output available in the context ",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT
        ],

        evaluation_steps=[
            "validate if all the information inside the output available in the context given"
        ],

        

    ) 

    faithfulness_metrics = FaithfulnessMetric(threshold=0.7) 
    result = evaluate([test_case], [faithfulness_metrics])
    print(result)