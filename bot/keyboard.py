from aiogram.types import KeyboardButton, ReplyKeyboardMarkup


button_hi = KeyboardButton('Hi 👋')

greet_kb = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
greet_kb.add(button_hi)

