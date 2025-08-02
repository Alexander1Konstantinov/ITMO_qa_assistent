import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import Tool, AgentExecutor, initialize_agent, create_react_agent
from langchain import hub
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import AgentType
from langchain.chains import LLMMathChain
from langchain.memory import ConversationBufferMemory


# Загрузка переменных окружения
load_dotenv(".env")

class RAGAssistant:
    def __init__(self, docs_folder="parsed_pages", vector_db_path="vector_db"):
        self.docs_folder = docs_folder
        self.vector_db_path = vector_db_path
        self.vector_db = None
        self.agent = None
        self.embeddings = None
        self.llm = None

        if not os.path.exists(self.docs_folder):
            raise FileNotFoundError(f"Папка с документами '{self.docs_folder}' не найдена")

        # Настройка компонентов
        self._setup_embeddings()
        self._setup_llm()
        self._setup_vector_db()
        self._setup_agent()  # Инициализация агента вместо QA-цепи

    def _setup_embeddings(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={"device": "cpu"},
        )
        print("🔍 Модель эмбеддингов загружена")

    def _setup_llm(self):
        api_key = os.getenv("MISTRAL_KEY")
        self.llm = ChatMistralAI(
            model="mistral-medium-2505", temperature=0, max_retries=2, api_key=api_key
        )
        print("💬 Языковая модель загружена")

    def _load_documents(self):
        documents = []
        for filename in os.listdir(self.docs_folder):
            if filename.endswith(".txt"):
                filepath = os.path.join(self.docs_folder, filename)
                try:
                    loader = TextLoader(filepath, encoding="utf-8")
                    docs = loader.load()
                    documents.extend(docs)
                    print(f"✅ Успешно загружен: {filename}")
                except Exception as e:
                    print(f"❌ Ошибка загрузки {filename}: {str(e)}")
        return documents

    def _setup_vector_db(self):
        if os.path.exists(self.vector_db_path):
            try:
                print(f"🔄 Загрузка векторной базы из {self.vector_db_path}")
                self.vector_db = FAISS.load_local(
                    self.vector_db_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                return
            except Exception as e:
                print(f"❌ Ошибка загрузки векторной базы: {str(e)}")
        
        # Создание новой базы при необходимости
        print("🧠 Создание новой векторной базы...")
        documents = self._load_documents()
        if not documents:
            raise ValueError("❌ Нет документов для обработки")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)
        print(f"📚 Создано {len(chunks)} фрагментов текста")

        self.vector_db = FAISS.from_documents(chunks, self.embeddings)
        self.vector_db.save_local(self.vector_db_path)
        print(f"💾 Векторная база сохранена в {self.vector_db_path}")

    def _setup_tools(self):
        """Создание инструментов для агента"""
        tools = []
        
        # Инструмент 1: Поиск по документам (RAG)
        def document_search(query: str) -> str:
            """Поиск в локальных документах"""
            docs = self.vector_db.similarity_search(query, k=2)
            return "\n\n".join([f"📄 {os.path.basename(doc.metadata['source'])}:\n{doc.page_content[:300]}..." for doc in docs])
        
        tools.append(Tool(
            name="DocumentSearch",
            func=document_search,
            description="Используй для поиска информации в локальных документах и файлах"
        ))
        
        # Инструмент 2: Калькулятор
        math_chain = LLMMathChain.from_llm(llm=self.llm)
        tools.append(Tool(
            name="Calculator",
            func=math_chain.run,
            description="Используй для математических расчетов и вычислений"
        ))
        
        # Инструмент 3: Поиск в Wikipedia
        wikipedia = WikipediaAPIWrapper()
        tools.append(Tool(
            name="WikipediaSearch",
            func=wikipedia.run,
            description="Используй для поиска актуальной информации в Wikipedia"
        ))
        
        return tools

    def _setup_agent(self):
        """Инициализация агента с инструментами и памятью"""
        tools = self._setup_tools()
        
        # Загружаем промпт
        prompt = hub.pull("hwchase17/react-chat")
        
        # Добавляем память для истории диалога
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Создаём агента с поддержкой памяти
        agent = create_react_agent(
            llm=self.llm,
            tools=tools,
            prompt=prompt
        )
        
        self.agent = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,  # Добавляем память
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )
        print("🤖 Агент с инструментами и памятью инициализирован")

    def ask(self, query):
        """
        Задать вопрос агенту
        
        :param query: Текст вопроса
        :return: Ответ агента и использованные инструменты
        """
        try:
            # Новый метод invoke вместо run
            result = self.agent.invoke({"input": query})
            
            return {
                "answer": result["output"],
                "sources": self._extract_sources(result)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _extract_sources(self, result):
        """Извлечение источников из лога агента"""
        sources = []
        
        # Анализируем шаги выполнения агента
        for step in result.get("intermediate_steps", []):
            # Шаг содержит (действие, результат наблюдения)
            action, observation = step
            
            # Для инструмента поиска документов
            if action.tool == "DocumentSearch":
                # Разбираем результаты наблюдения
                for part in observation.split("\n\n"):
                    if part.startswith("📄"):
                        source, content = part.split(":", 1)
                        sources.append({
                            "source": source.strip("📄 "),
                            "content": content.strip()[:200] + "..."
                        })
        
        return sources

    def interactive_mode(self):
        print("\n" + "=" * 50)
        print("🤖 RAG-ассистент готов к работе (Агентский режим)")
        print("=" * 50 + "\n")

        while True:
            query = input("❓ Ваш вопрос (или 'exit' для выхода): ").strip()
            if query.lower() in ['exit', 'выход']:
                break

            response = self.ask(query)

            if "error" in response:
                print(f"\n⚠️ Ошибка: {response['error']}\n")
            else:
                print(f"\n💡 Ответ: {response['answer']}")

                if response["sources"]:
                    print("\n🔍 Источники информации:")
                    for i, source in enumerate(response["sources"], 1):
                        print(f"{i}. Файл: {source['source']}")
                        print(f"   Фрагмент: {source['content']}")

                print("\n" + "-" * 50)


def main():
    try:
        assistant = RAGAssistant(docs_folder="parsed_pages", vector_db_path="vector_db")
        assistant.interactive_mode()

    except Exception as e:
        print(f"\n🛑 Критическая ошибка: {str(e)}")
        print("Программа завершена")


if __name__ == "__main__":
    print("Запуск RAG-ассистента в агентском режиме...")
    main()