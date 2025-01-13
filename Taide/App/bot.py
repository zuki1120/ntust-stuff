import os
import logging
from telegram import Update
from dotenv import load_dotenv
from query_engine import process_query
from telegram.ext import (ApplicationBuilder, 
                          CommandHandler, 
                          MessageHandler, 
                          filters, 
                          ContextTypes)
from telegram.constants import ChatAction  # 用於發送聊天動作

# 設定日誌
logging.basicConfig(
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level = logging.INFO,
)
logger = logging.getLogger(__name__)

# 開始命令處理
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("歡迎使用醫療諮詢系統！請直接輸入您的問題。")

# 回答處理
async def handle_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.message.text
    logger.info(f"收到用戶查詢: {query}")
    try:
        # 發送「正在輸入中」提示
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

        response = process_query(query)
        if not response:
            logger.warning("回應為空")
            response = "抱歉，我目前無法提供答案。"
        await update.message.reply_text(response)
    except Exception as e:
        logger.error(f"錯誤: {e}")
        await update.message.reply_text("系統處理您的請求時出現問題，請稍後再試。")

# 主函數
def main():
    load_dotenv()
    TOKEN = os.getenv("TELEGRAM_BOT_API_TOKEN")

    # 創建應用程序
    application = ApplicationBuilder().token(TOKEN).build()

    # 註冊處理器
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_query))

    # 運行 Bot
    application.run_polling()

if __name__ == "__main__":
    main()