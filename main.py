import os
import sys
import time
import asyncio
import hashlib
import logging
import re
import html
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher
from aiogram.types import (
    InlineQuery,
    InlineQueryResultArticle,
    InputTextMessageContent,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    ChosenInlineResult,
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
    # Заменяем списки со звездочками (* пункт) и дефисы (- пункт) на точки (• пункт)
    clean_answer = re.sub(r'^[ \t]*[\*\-][ \t]+', '• ', raw_answer, flags=re.MULTILINE)

    # Экранируем HTML-символы (<, >, &)
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
        return f"Не удалось получить ответ от нейросети: {e}"

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
    """Мгновенная выдача подсказки (без задержки на генерацию ИИ)."""
    user_text = query.query.strip()

    if len(user_text) < 2:
        hint_id = hashlib.md5("hint".encode()).hexdigest()
        item = InlineQueryResultArticle(
            id=hint_id,
            title="💡 Введите вопрос нейросети",
            description="Например: @dmgemmabot что такое танк?",
            input_message_content=InputTextMessageContent(
                message_text="Подсказка: напишите @dmgemmabot и ваш вопрос в любом чате."
            )
        )
        await query.answer([item], cache_time=1, is_personal=True)
        return

    # Заглушка, которая мгновенно отправляется в чат
    initial_text = f"❓ <b>{html.escape(user_text)}</b>\n\n⏳ <i>Нейросеть генерирует ответ...</i>"

    # Кнопка под сообщением (необходима для получения inline_message_id в Telegram)
    thinking_kb = InlineKeyboardMarkup(
        inline_keyboard=[[
            InlineKeyboardButton(text="⏳ Генерация ответа...", callback_data="thinking")
        ]]
    )

    result_id = hashlib.md5(f"{user_text}_{time.time()}".encode()).hexdigest()
    item = InlineQueryResultArticle(
        id=result_id,
        title=f"Задать вопрос: {user_text[:35]}",
        description="Нажмите, чтобы отправить вопрос в чат",
        reply_markup=thinking_kb,
        input_message_content=InputTextMessageContent(
            message_text=initial_text,
            parse_mode="HTML"
        )
    )

    # cache_time=1 позволяет сразу обновлять запросы при изменении текста
    await query.answer([item], cache_time=1, is_personal=True)

@dp.chosen_inline_result()
async def handle_chosen_inline_result(chosen: ChosenInlineResult):
    """Срабатывает в момент, когда пользователь нажал на карточку и сообщение улетело в чат."""
    query_text = chosen.query.strip()
    inline_msg_id = chosen.inline_message_id

    if not inline_msg_id or not query_text:
        return

    logger.info(f"Сообщение отправлено пользователем. Запрос: '{query_text}'. Генерируем ответ...")

    # Получаем ответ от Gemma
    ai_answer = await ask_ai(query_text)

    # Формируем итоговый красивый HTML
    final_text = format_telegram_message(query_text, ai_answer)

    # Кнопка под готовым сообщением для быстрого вызова бота другими участниками чата
    ready_kb = InlineKeyboardMarkup(
        inline_keyboard=[[
            InlineKeyboardButton(text="✨ Спросить у @dmgemmabot", switch_inline_query_current_chat="")
        ]]
    )

    try:
        # Редактируем уже отправленное сообщение в чате
        await bot.edit_message_text(
            inline_message_id=inline_msg_id,
            text=final_text,
            parse_mode="HTML",
            reply_markup=ready_kb
        )
        logger.info("Сообщение успешно обновлено на готовый ответ!")
    except Exception as e:
        logger.error(f"Ошибка при редактировании сообщения: {e}")

async def main():
    logger.info("Запуск бота @dmgemmabot (режим мгновенной отправки + редактирования)...")
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Бот остановлен.")
