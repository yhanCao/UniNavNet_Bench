import openai

class LLMQwQ():
    def __init__(self):
        self.history = []
        self.client = openai.Client(
        base_url="http://192.168.100.24:8055/v1", 
        api_key="EMPTY")
    def scoring_eqa(self,question,gt_answer,answer):
        from llm.scoring import SYSTEM_PROMPT
        response = self.client.chat.completions.create(
            model="default",
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': f"Question: {question}\nAnswer: {gt_answer}\nResponse: {answer}"}
            ],
            temperature=0.8,
            max_tokens=1024,
        )
        print(response.choices[0].message.content)
        return response.choices[0].message.content