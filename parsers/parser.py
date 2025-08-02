import requests
from bs4 import BeautifulSoup
import PyPDF2
import os

# Список URL-адресов для парсинга
urls = [
    "https://abit.itmo.ru/program/master/ai",
    "https://abit.itmo.ru/program/master/ai_product"
]

# Функция для парсинга текста с веб-страницы
def parse_text_from_url(url):
    try:
        
        # Получаем HTML-код страницы
        response = requests.get(url)
        response.raise_for_status()  # Проверяем, что запрос прошел успешно

        # Создаем объект BeautifulSoup для парсинга HTML
        soup = BeautifulSoup(response.text, 'html.parser')

        # Извлекаем текст из HTML
        text = soup.get_text(separator=' ', strip=True)

        return text
    except requests.RequestException as e:
        print(f"Ошибка при получении данных с {url}: {e}")
        return None



def parse_pdf_with_pypdf2(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

folder = 'parsed_pages'
os.makedirs(folder, exist_ok=True)
filename = f"document.txt"
filepath = os.path.join(folder, filename)

file_path_ai = 'C:\qa_assistant_ITMO\parsed_pages\plan_ai.pdf'
file_path_manage = 'C:\qa_assistant_ITMO\parsed_pages\plan_manage.pdf'
text =''
text += parse_pdf_with_pypdf2(file_path_ai)
text += parse_pdf_with_pypdf2(file_path_manage)
with open(filepath, 'a', encoding='utf-8') as f:
    f.write(text)


for url in urls:
    print(f"Парсинг текста с {url}...")
    text = parse_text_from_url(url)
    print(text)
    if text:
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(f"URL: {url}\n")
            f.write(text)
