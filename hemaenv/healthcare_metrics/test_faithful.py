from deepeval.evaluate import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric
from app.healthcare_chatbot import chatbot

retrieval_context = [
        "Dont drink water",
        "we can use the first aid box in the vehicals or schools or ect..",
        "It will be using if any emergency is there first we can use it before going to the hospital",
        "its wont work for fevers or any simple medical issues "
]


def test_faithfulness_metrics():

    query = "i have headache how to reduce it is there any quick medication"
    output = chatbot(query)
    print(output)

    test_case = LLMTestCase(
        input=query,
        actual_output=output,
        retrieval_context=retrieval_context,
    )

    faith = FaithfulnessMetric(
        threshold=0.7
    )
    #hal = HallucinationMetric(threshold=0.2)


    result = evaluate([test_case], [faith])
    print(result)