from telegram.ext import *
from io import BytesIO
import numpy as np
import cv2
import os
from preprocessing import image_preparation, finding_contours
from recognition_service import recognize_img, for_exponentiation, for_sqrt



with open('token.txt', 'r') as f:
    TOKEN = f.read()

def start(update, context):
    update.message.reply_text("Welcome!")

def help(update, context):
    update.message.reply_text(
        """
        /start - Starts conversation
        /help - Shows this message
        /check - Recognition of an image
        """
    )

def check(update, context):
    pass


def handle_message(update, context):
    update.message.reply_text("Отправьте картинку с математической формулой")

def handle_photo(update, context):
    file = context.bot.get_file(update.message.photo[-1].file_id)
    f = BytesIO(file.download_as_bytearray())
    file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)

    img_arr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    pic, gray = image_preparation(img_arr)
    objects = finding_contours(pic, img_arr, gray)
    print(objects)
    amount_of_nums = len(objects)

    items = []
    first = True
    for i in range(amount_of_nums):
        img = cv2.imread(f"num_{i}.png")
        try:
            if objects[i][4] == True and objects[i-1][4] == False:
                items.append("^")
        except:
            update.message.reply_text(f"Что-то пошло не так...")
        finally:
            if recognize_img(img) == 'multiply':
                items.append('*')
            else:
                items.append(recognize_img(img))
            os.remove(f"num_{i}.png")
    changable_text = "".join(items)
    text = for_sqrt(for_exponentiation(changable_text))


    update.message.reply_text(f"На этом изображении я вижу {text}")



updater = Updater(TOKEN, use_context=True)
dp = updater.dispatcher

dp.add_handler(CommandHandler("start", start))
dp.add_handler(CommandHandler("help", help))
#dp.add_handler(CommandHandler("train", train))
dp.add_handler(MessageHandler(Filters.text, handle_message))
dp.add_handler(MessageHandler(Filters.photo, handle_photo))

updater.start_polling()
updater.idle()