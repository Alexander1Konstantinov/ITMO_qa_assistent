import requests
import os
from bs4 import BeautifulSoup
from urllib.parse import urlparse

def parse_and_save_page(url, folder="parsed_pages"):
    try:
        # Создаем папку, если не существует
        os.makedirs(folder, exist_ok=True)
        
        # Загрузка страницы
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        
        # Парсинг HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Извлечение заголовка страницы
        title = soup.title.string if soup.title else "Без заголовка"
        
        # Извлечение основного текстового контента
        # Удаляем ненужные элементы (скрипты, стили и т.д.)
        for element in soup(["script", "style", "meta", "link", "nav", "footer"]):
            element.decompose()
        
        # Получаем чистый текст
        text = soup.get_text(separator='\n', strip=True)
        
        # Генерируем имя файла из домена
        domain = urlparse(url).netloc.replace("www.", "")
        filename = f"{domain}.txt"
        filepath = os.path.join(folder, filename)
        
        # Сохраняем в файл
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(f"URL: {url}\n")
            f.write(f"Заголовок: {title}\n\n")
            f.write(text)
        
        return f"Сохранено: {filepath} ({len(text)} символов)"
    
    except Exception as e:
        return f"Ошибка: {url} - {str(e)}"

# Анализ двух страниц
pages = [
    "https://habr.com/ru/articles/881372/",
    "https://habr.com/ru/articles/779526/"
]

for url in pages:
    result = parse_and_save_page(url)
    print(result)