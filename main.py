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

def format_telegram_message(question: str, raw_answer: str, mode: str = "normal") -> str:
    """Преобразует Markdown нейросети в аккуратный безопасный HTML для Telegram."""
    # Заменяем списки со звездочками (* пункт) и дефисы (- пункт) на точки (• пункт)
    clean_answer = re.sub(r'^[ \t]*[\*\-][ \t]+', '• ', raw_answer, flags=re.MULTILINE)

    # Экранируем HTML-символы (<, >, &)
    safe_q = html.escape(question)
    safe_a = html.escape(clean_answer)

    # Превращаем **жирный текст** в <b>жирный текст</b>
    safe_a = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', safe_a)

    icons = {
        "short": "⚡",
        "detailed": "📚",
        "normal": "❓"
    }
    icon = icons.get(mode, "❓")

    return f"{icon} <b>{safe_q}</b>\n\n🤖 {safe_a}"

async def ask_ai(prompt: str, mode: str = "normal") -> str:
    """Запрос к модели Gemma с настройкой стиля ответа."""
    if mode == "short":
        system_instruction = (
            "Ты лаконичный AI-помощник. Ответь максимально кратко (1-3 емких предложения), "
            "прямо по сути вопроса, без долгих вступлений и лишней воды на русском языке."
        )
        max_tokens = 300
    elif mode == "detailed":
        system_instruction = (
            "Ты экспертный AI-помощник. Дай подробный, всесторонний и глубоко структурированный ответ на русском языке. "
            "Разбей ответ на логические блоки, приведи ключевые факты и детали. "
            "Для списков используй символ «•» вместо звездочек."
        )
        max_tokens = 1200
    else:  # normal
        system_instruction = (
            "Ты умный и полезный AI-помощник. Отвечай сбалансированно, емко и по существу на русском языке. "
            "Для списков используй символ «•» вместо звездочек."
        )
        max_tokens = 700

    try:
        response = await ai_client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Ошибка запроса к AI API ({mode}): {e}")
        return f"Не удалось получить ответ от нейросети: {e}"

@dp.message(CommandStart())
async def cmd_start(message: Message):
    """Приветственное сообщение при открытии бота в ЛС."""
    await message.answer(
        "👋 <b>Привет! Я инлайн-бот с моделью Gemma 4.</b>\n\n"
        "Ты можешь использовать меня в <b>любом</b> чате Telegram!\n\n"
        "Просто напиши в строке ввода любого чата:\n"
        "<code>@dmgemmabot твой вопрос?</code>\n\n"
        "Тебе будут доступны 3 режима ответа:\n"
        "💬 <b>Обычный</b> — сбалансированный ответ\n"
        "⚡ <b>Краткий</b> — 1-2 предложения без воды\n"
        "📚 <b>Подробный</b> — развернутый анализ с фактами",
        parse_mode="HTML"
    )

@dp.inline_query()
async def handle_inline_query(query: InlineQuery):
    """Выдает 3 карточки на выбор: Обычный, Краткий, Подробный."""
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

    safe_q = html.escape(user_text)
    timestamp = int(time.time() * 1000)

    # Клавиатура со статусом ожидания
    thinking_kb = InlineKeyboardMarkup(
        inline_keyboard=[[
            InlineKeyboardButton(text="⏳ Генерация ответа...", callback_data="thinking")
        ]]
    )

    # 1. Обычный
    card_normal = InlineQueryResultArticle(
        id=f"normal:{timestamp}",
        title="💬 Обычный ответ",
        description=f"Сбалансированный ответ: {user_text[:35]}",
        reply_markup=thinking_kb,
        input_message_content=InputTextMessageContent(
            message_text=f"❓ <b>{safe_q}</b>\n\n⏳ <i>Нейросеть генерирует ответ...</i>",
            parse_mode="HTML"
        )
    )

    # 2. Краткий
    card_short = InlineQueryResultArticle(
        id=f"short:{timestamp}",
        title="⚡ Краткий ответ",
        description="1–3 предложения по сути, без лишней воды",
        reply_markup=thinking_kb,
        input_message_content=InputTextMessageContent(
            message_text=f"⚡ <b>{safe_q}</b>\n\n⏳ <i>Генерирую краткий ответ...</i>",
            parse_mode="HTML"
        )
    )

    # 3. Подробный
    card_detailed = InlineQueryResultArticle(
        id=f"detailed:{timestamp}",
        title="📚 Подробный ответ",
        description="Развернутый анализ с деталями, фактами и пунктами",
        reply_markup=thinking_kb,
        input_message_content=InputTextMessageContent(
            message_text=f"📚 <b>{safe_q}</b>\n\n⏳ <i>Генерирую подробный разбор...</i>",
            parse_mode="HTML"
        )
    )

    # Отправляем 3 карточки пользователю
    await query.answer([card_normal, card_short, card_detailed], cache_time=1, is_personal=True)

@dp.chosen_inline_result()
async def handle_chosen_inline_result(chosen: ChosenInlineResult):
    """Срабатывает при выборе одной из 3 карточек."""
    query_text = chosen.query.strip()
    inline_msg_id = chosen.inline_message_id

    if not inline_msg_id or not query_text:
        return

    # Определяем выбранный режим по префиксу id карточки
    result_id = chosen.result_id or ""
    if result_id.startswith("short:"):
        mode = "short"
    elif result_id.startswith("detailed:"):
        mode = "detailed"
    else:
        mode = "normal"

    logger.info(f"Выбран режим '{mode}' для запроса: '{query_text}'. Генерируем...")

    # Запрашиваем ответ у Gemma
    ai_answer = await ask_ai(query_text, mode=mode)

    # Форматируем итоговое сообщение
    final_text = format_telegram_message(query_text, ai_answer, mode=mode)

    ready_kb = InlineKeyboardMarkup(
        inline_keyboard=[[
            InlineKeyboardButton(text="✨ Спросить у @dmgemmabot", switch_inline_query_current_chat="")
        ]]
    )

    try:
        await bot.edit_message_text(
            inline_message_id=inline_msg_id,
            text=final_text,
            parse_mode="HTML",
            reply_markup=ready_kb
        )
        logger.info(f"Сообщение [{mode}] успешно обновлено!")
    except Exception as e:
        logger.error(f"Ошибка при редактировании сообщения: {e}")

async def main():
    logger.info("Запуск бота @dmgemmabot (3 режима: Обычный / Краткий / Подробный)...")
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Бот остановлен.")
