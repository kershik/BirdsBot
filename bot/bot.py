from aiogram import Bot, Dispatcher, executor, types
from PIL import Image
import torch
import os
from torchvision import transforms
import keyboard as kb
from model_26 import model, transform
import pandas as pd

API_TOKEN = '5336164984:AAGVvcXrchOYqIRVuMBXpnlW3TiMYbIL5-I' # example how to place token
FILE = 'test.jpg'
counter = 0

birds = pd.read_csv('birds.csv')

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

@dp.message_handler(commands=['start'])
async def process_start_command(message: types.Message):
    await message.reply("H3LL0!", reply_markup=kb.greet_kb)

@dp.message_handler(content_types=['photo'])
async def handle_photo(message):
    await message.reply("L0AD1NG...")
    await message.photo[-1].download(FILE)
    image = Image.open(FILE)
    image.load()

    transformed_img = transform(image)
    transformed_img = transformed_img.unsqueeze(0) 
    model.load_state_dict(torch.load('best_model_26.pth', map_location=torch.device('cpu')))
    model.eval()
    out = model(transformed_img)
    out = torch.nn.functional.softmax(out, dim=-1)
    _, pred = torch.max(out.data, 1)
    res = birds['b_class'][pred.numpy()].to_string()
    await message.reply(res)
    os.remove(FILE)



if __name__ == '__main__':
    executor.start_polling(dp)
