import os
import asyncio
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import Message
from qa_assistant import RAGAssistant
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

class TelegramRAGBot:
    def __init__(self):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not self.bot_token:
            raise ValueError("❌ TELEGRAM_BOT_TOKEN не установлен в .env файле")
        
        self.bot = Bot(token=self.bot_token)
        self.dp = Dispatcher()
        self.rag_assistant = None
        
        # Регистрация обработчиков
        self.dp.message(Command("start"))(self.start_command)
        self.dp.message(Command("help"))(self.help_command)
        # self.dp.message(Command("reindex"))(self.reindex_command)
        self.dp.message()(self.handle_message)
    
    async def initialize_assistant(self):
        """Инициализация RAG ассистента"""
        print("🔄 Инициализация RAG ассистента...")
        self.rag_assistant = RAGAssistant(
            docs_folder="parsed_pages",
            vector_db_path="vector_db"
        )
        print("✅ RAG ассистент готов")
    
    async def start_command(self, message: Message):
        """Обработка команды /start"""
        await message.answer(
            "🤖 Привет! Я RAG-ассистент, обученный на ваших документах.\n\n"
            "Задайте мне вопрос по содержимому документов, и я постараюсь найти ответ!\n\n"
            "Используйте /help для справки"
        )
    
    async def help_command(self, message: Message):
        """Обработка команды /help"""
        help_text = (
            "📚 Доступные команды:\n"
            "/start - Начать работу с ботом\n"
            "/help - Получить справку\n"
            "/reindex - Пересоздать векторную базу данных\n\n"
            "💡 Просто задайте вопрос в чате, и я поищу ответ в документах!"
        )
        await message.answer(help_text)
    
    async def handle_message(self, message: Message):
        """Обработка обычных сообщений с вопросами"""
        # # Пропускаем команды
        # if message.text.startswith('/'):
        #     return
            
        # Показываем статус "печатает..."
        await message.bot.send_chat_action(message.chat.id, "typing")
        
        try:
            # Инициализация ассистента при первом запросе
            if not self.rag_assistant:
                await self.initialize_assistant()
            
            # Получаем ответ от RAG системы
            response = self.rag_assistant.ask(message.text)
            
            if "error" in response:
                await message.answer(f"⚠️ Ошибка: {response['error']}")
                return
            
            # Форматируем ответ с помощью доступных функций
            answer_text = "💡 Ответ:" + f"\n{response['answer']}"
            sources_text = ""
            
            # if response["sources"]:
            #     sources_text = hitalic("\n\n🔍 Источники информации:") + "\n"
            #     for i, source in enumerate(response["sources"], 1):
            #         source_name = source["source"]
            #         content_preview = source["content"]
            #         sources_text += f"{i}. {hbold(source_name)}\n{content_preview}\n"
            
            # Отправляем ответ
            full_response = f"{answer_text}{sources_text}"
            await message.answer(full_response)
            
        except Exception as e:
            await message.answer(f"⚠️ Произошла ошибка: {str(e)}")
    
    async def start(self):
        """Запуск бота"""
        await self.dp.start_polling(self.bot)

if __name__ == "__main__":
    # Для Windows
    # if os.name == 'nt':
    #     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    print("🚀 Запуск Telegram бота...")
    bot = TelegramRAGBot()
    
    try:
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        print("Бот остановлен")