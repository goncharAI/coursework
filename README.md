Telegram Data Analyzer
определение вероятного места жительства и временного профиля
Требования:
Python 3.12+ 
pip

Запуск:
git clone https://github.com/goncharAI/coursework
cd coursework
pip install -r requirements.txt
streamlit run ui.py

можно запустить скриптом run.bat

При загрузке больших json и csv файлов проявите терпение во время лемматизации и фильтрации. 
Рекомендуется загружать одиночные чаты во избежание долгого времени работы (450000 сообщений обрабатываются примерно за час)
Загружать необходимо JSON-файлы, которые создаются Telegram-ом при использовании функций "export chat history" и "expoet telegram data"
