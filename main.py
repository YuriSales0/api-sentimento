from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import openai
import os
from dotenv import load_dotenv

# Carrega sua chave OpenAI
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
print("Chave OpenAI carregada:", openai.api_key)

app = FastAPI()

# Carrega o modelo pré-treinado de sentimento
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)

class Review(BaseModel):
    texto: str

@app.get("/")
def home():
    return {"status": "API rodando com sucesso!"}

@app.post("/analise_sentimento")
async def analisar_sentimento(review: Review):
    resultado = sentiment_pipeline(review.texto)[0]
    sentimento = resultado['label']
    score = round(resultado['score'], 2)

    prompt_resumo = (f"Resuma em português claro e amigável esta avaliação de cliente:\n\n"
                     f"{review.texto}\n\n"
                     f"Sentimento detectado: {sentimento} com confiança {score}.")

    resumo_gpt = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt_resumo}],
        max_tokens=100
    )

    resumo = resumo_gpt.choices[0].message.content.strip()

    # Extrair termos-chave
    prompt_keywords = (f"Extraia 3 principais termos-chave ou expressões curtas "
                       f"que definem o sentimento ou pontos fortes/fracos do seguinte texto:\n\n"
                       f"{review.texto}\n\n"
                       f"Apresente apenas como uma lista separada por vírgulas:")

    keywords_gpt = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt_keywords}],
        max_tokens=50
    )

    termos_chave = keywords_gpt.choices[0].message.content.strip().split(", ")

    # NOVA FUNÇÃO: Relatório detalhado
    prompt_relatorio = (f"Crie um relatório breve e amigável para o gerente dessa empresa, "
                        f"explicando o sentimento do cliente e sugerindo ações práticas. "
                        f"Use uma abordagem positiva e objetiva.\n\n"
                        f"Avaliação do cliente: {review.texto}\n"
                        f"Sentimento: {sentimento} com confiança {score}.")

    relatorio_gpt = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt_relatorio}],
        max_tokens=200
    )

    relatorio = relatorio_gpt.choices[0].message.content.strip()

    return {
        "sentimento": sentimento,
        "score": score,
        "resumo": resumo,
        "termos_chave": termos_chave,
        "relatorio": relatorio
    }