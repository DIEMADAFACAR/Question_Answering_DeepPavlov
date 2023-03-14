from flask import Flask, request, render_template
from deeppavlov import build_model, configs
import json

app = Flask(__name__)
model = build_model(configs.squad.squad_ru_bert, download=True)

@app.route('/')
def home():
    label = 'Введите текст и вопрос, чтобы получить ответ'
    return render_template("index.html", label=label, answer='')

@app.route('/answer', methods=['POST'])
def get_answer():
    context = request.form['context']
    question = request.form['question']
    result = model([context], [question])
    answer = result[0][0]
    return json.dumps(answer, ensure_ascii=False)

if __name__ == '__main__':
    app.run(debug=True)


