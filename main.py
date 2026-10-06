import os
import sys
import asyncio
import hashlib
import logging
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher
from aiogram.types import (
    InlineQuery,
    InlineQueryResultArticle,
    InputTextMessageContent,
    Message
)
from aiogram.filters import CommandStart
from openai import AsyncOpenAI

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Загружаем настройки из .env
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
AI_API_KEY = os.getenv("AI_API_KEY", "").strip()
AI_BASE_URL = os.getenv("AI_BASE_URL", "https://api.intelligence.io.solutions/api/v1").strip()
AI_MODEL = os.getenv("AI_MODEL", "google/gemma-4-26b-a4b-it").strip()

if not TELEGRAM_BOT_TOKEN:
    logger.error("ОШИБКА: TELEGRAM_BOT_TOKEN не указан в .env файле!")
    sys.exit(1)

bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()

ai_client = AsyncOpenAI(
    api_key=AI_API_KEY,
    base_url=AI_BASE_URL
)

def format_telegram_message(question: str, raw_answer: str) -> str:
    """Преобразует Markdown нейросети в аккуратный безопасный HTML для Telegram."""
    import re
    import html

    # Заменяем звездочки списков (* пункт) и дефисы (- пункт) на аккуратные точки (• пункт)
    clean_answer = re.sub(r'^[ \t]*[\*\-][ \t]+', '• ', raw_answer, flags=re.MULTILINE)

    # Экранируем специальные HTML-символы (<, >, &)
    safe_q = html.escape(question)
    safe_a = html.escape(clean_answer)

    # Превращаем **жирный текст** в <b>жирный текст</b>
    safe_a = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', safe_a)

    return f"❓ <b>{safe_q}</b>\n\n🤖 {safe_a}"

async def ask_ai(prompt: str) -> str:
    """Запрос к модели Gemma через OpenAI-совместимый API io.net."""
    try:
        response = await ai_client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты краткий, умный и полезный AI-помощник. "
                        "Отвечай емко, понятно и по существу на русском языке. "
                        "Для списков используй символ «•» вместо звездочек. "
                        "Твой ответ будет сразу отправлен пользователем в чат."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Ошибка запроса к AI API: {e}")
        return f"Не удалось получить ответ: {e}"

@dp.message(CommandStart())
async def cmd_start(message: Message):
    """Приветственное сообщение при открытии бота в ЛС."""
    await message.answer(
        "👋 <b>Привет! Я инлайн-бот с моделью Gemma 4.</b>\n\n"
        "Ты можешь использовать меня в <b>любом</b> чате Telegram!\n\n"
        "Просто напиши в строке ввода любого чата:\n"
        "<code>@dmgemmabot твой вопрос?</code>\n\n"
        "Например:\n"
        "<code>@dmgemmabot что такое танк?</code>\n"
        "<code>@dmgemmabot напиши формулу фотосинтеза</code>",
        parse_mode="HTML"
    )

@dp.inline_query()
async def handle_inline_query(query: InlineQuery):
    user_text = query.query.strip()

    # Если запрос слишком короткий — показываем подсказку
    if len(user_text) < 3:
        hint_id = hashlib.md5("hint".encode()).hexdigest()
        item = InlineQueryResultArticle(
            id=hint_id,
            title="💡 Введите вопрос",
            description="Например: @dmgemmabot что такое танк?",
            input_message_content=InputTextMessageContent(
                message_text="Подсказка: напишите @dmgemmabot и ваш вопрос в любом чате."
            )
        )
        await query.answer([item], cache_time=2, is_personal=True)
        return

    logger.info(f"Запрос от @{query.from_user.username or query.from_user.id}: {user_text}")

    # Запрашиваем ответ у нейросети
    ai_answer = await ask_ai(user_text)

    # Формируем итоговое красивое сообщение (HTML)
    formatted_message = format_telegram_message(user_text, ai_answer)

    # Превью для карточки (без лишних символов)
    preview_desc = ai_answer[:90].replace("\n", " ").replace("*", "").strip() + "..."
    result_id = hashlib.md5(f"{user_text}_{query.id}".encode()).hexdigest()

    item = InlineQueryResultArticle(
        id=result_id,
        title=f"Ответ: {user_text[:35]}",
        description=preview_desc,
        input_message_content=InputTextMessageContent(
            message_text=formatted_message,
            parse_mode="HTML"
        )
    )

    await query.answer([item], cache_time=3, is_personal=True)

async def main():
    logger.info("Запуск бота @dmgemmabot...")
    # Удаляем вебхуки, если были
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Бот остановлен.")
