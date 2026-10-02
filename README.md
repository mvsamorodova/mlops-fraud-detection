# Потоковый скоринг фродовых транзакций

Kafka `transactions` → препроцессинг → CatBoost → Kafka `scores` → PostgreSQL → Streamlit.

Проект использует [код семинара №4](https://github.com/NikitaMalykhin/mts25_mlops_hw2_real_time_fraud_detection) и собственную обученную модель. Сервис выполняет только inference на CPU. Модель и параметры препроцессинга находятся в `fraud_detector/models/`.

Чтение и отправка сообщений, препроцессинг и скоринг разделены на файлы `fraud_detector/app/app.py`, `fraud_detector/src/preprocessing.py` и `fraud_detector/src/scorer.py`.

Отдельный сервис `score_writer` читает топик `scores` с полями `transaction_id`, `score`, `fraud_flag` и записывает результаты в PostgreSQL. Контейнеры запускаются в одной сети через `docker-compose.yml`.

## Запуск

Нужны Docker и Docker Compose. Запустите Docker и выполните из корня репозитория:

```bash
docker compose up -d --build
docker compose ps -a
```

Kafka и PostgreSQL должны быть `healthy`, остальные рабочие сервисы — `Up`. Для `kafka-setup` статус `Exited (0)` означает успешное создание топиков.

Приложение: [http://localhost:8501](http://localhost:8501).

## Проверка

1. Скачайте `test.csv` из [соревнования Kaggle](https://www.kaggle.com/t/1918f3f6435300327d38d6c596f97394).
2. Загрузите `test.csv` в приложение и нажмите «Отправить test.csv». Дождитесь статуса «Отправлен».
3. Нажмите «Посмотреть результаты». Интерфейс показывает:
   - 10 последних транзакций из PostgreSQL с `fraud_flag = 1`, если они есть;
   - гистограмму скоров последних 100 транзакций; если в базе меньше 100, используются все имеющиеся записи.

Проверка записи результатов в PostgreSQL:

```bash
docker compose exec postgres psql -U fraud -d fraud -c \
"SELECT COUNT(*) AS total, COUNT(*) FILTER (WHERE fraud_flag = 1) AS fraud FROM transaction_scores;"
```
