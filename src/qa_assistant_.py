import os
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv

load_dotenv(".env")

class AdmissionConsultant:
    def __init__(self):
        self.vector_db = self._setup_vector_db()
        self.llm = ChatMistralAI(
            model="mistral-medium",
            temperature=0.2,
            api_key=os.getenv("MISTRAL_KEY")
        )
        self.agent = self._setup_agent()

    def _setup_vector_db(self):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        return FAISS.load_local("vector_db", embeddings, allow_dangerous_deserialization=True)

    def _setup_tools(self):
        def program_search(query: str) -> str:
            docs = self.vector_db.similarity_search(query, k=2)
            return "\n".join([f"📄 {doc.metadata['source']}:\n{doc.page_content[:300]}..." for doc in docs])

        return [
            Tool(
                name="ProgramSearch",
                func=program_search,
                description="Поиск информации о программах магистратуры"
            )
        ]

    def _setup_agent(self):
        tools = self._setup_tools()

        prompt_template = """Ты консультант для абитуриентов магистратуры ИТМО. Отвечай на русском языке. 
        Важно:  отвечай только на релевантные вопросы по обучению в магистратурах Искусственный интеллект и Управление ИИ-продуктами/AI Product в ИТМО.
    Всегда строго придерживайся следующего формата:
    Question: входной вопрос
    Thought: анализ вопроса и определение нужного действия
    Action: инструмент (один из [{tools}]) или "Final Answer"
    Action Input: входные данные для инструмента (если нужен)
    Observation: результат выполнения инструмента
    ... (этот цикл может повторяться несколько раз)
    Thought: окончательный вывод
    Final Answer: итоговый ответ на русском языке
    Пример 1 (с инструментом):
    Question: Какие курсы есть по ИИ?
    Thought: Нужно найти информацию о курсах по искусственному интеллекту
    Action: ProgramSearch
    Action Input: курсы по искусственному интеллекту
    Observation: 📄 ИИ.txt: 1. Нейронные сети 2. Компьютерное зрение...
    Thought: Нашел нужную информацию
    Final Answer: Вот курсы по ИИ: 1. Нейронные сети 2. Компьютерное зрение...
    Пример 2 (без инструмента):
    Question: Привет
    Thought: Это приветствие, можно ответить напрямую
    Final Answer: Здравствуйте! Чем могу помочь?
    Текущий диалог:
    {chat_history}
    Question: {input}
    {agent_scratchpad}"""
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["input", "chat_history", "agent_scratchpad"],
            partial_variables={
                "tool_names": ", ".join([t.name for t in tools])
            }
        )

        agent = create_react_agent(
            llm=self.llm,
            tools=tools,
            prompt=prompt
        )
        return AgentExecutor(
            agent=agent,
            tools=tools,
            memory=ConversationBufferMemory(memory_key="chat_history"),
            verbose=True,
            max_iterations=3,
            handle_parsing_errors=True
        )


    def ask(self, query):
        try:
            result = self.agent.invoke({"input": query})
            return result["output"]
        except Exception as e:
            return f"Ошибка: {str(e)}"

if __name__ == "__main__":
    try:
        consultant = AdmissionConsultant()
        print("🎓 Консультант магистратуры ИТМО готов к работе")
        print("Примеры вопросов:")
        print("- Какие курсы есть по ИИ?")
        print("- Как поступить на магистратуру?")
        print("- Расскажи о программе по компьютерному зрению")
        
        while True:
            query = input("\n❓ Ваш вопрос (или 'выход'): ").strip()
            if query.lower() in ('exit', 'выход'):
                break
                
            response = consultant.ask(query)
            print(f"\n💡 Ответ: {response}")
            
    except Exception as e:
        print(f"\n⚠️ Ошибка: {str(e)}")