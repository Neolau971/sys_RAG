from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

load_dotenv()

# ----------------------------
# CONFIG
# ----------------------------

FAISS_FOLDER = Path("data/langchain_faiss")

EMBEDDING_MODEL = "mistral-embed"
CHAT_MODEL = "mistral-small-2603"

TOP_K = 20
TEMPERATURE = 0.2


def format_documents(documents) -> str:
    """
    Transforme les Documents renvoyés par FAISS en contexte RAG lisible
    par le modèle de chat.
    """
    formatted_parts = []

    for position, document in enumerate(documents, start=1):
        metadata = document.metadata

        title = metadata.get("title") or "Sans titre"
        city = metadata.get("city") or "Ville non renseignée"
        start_date = metadata.get("firstdate_begin") or "Date non renseignée"
        source_url = metadata.get("source_url") or "URL non renseignée"

        formatted_parts.append(
            f"[Source {position}]\n"
            f"Titre : {title}\n"
            f"Ville : {city}\n"
            f"Date de début : {start_date}\n"
            f"Description : {document.page_content}\n"
            f"URL : {source_url}"
        )

    return "\n\n".join(formatted_parts)


def load_vector_store() -> FAISS:
    """Charge l'index FAISS créé avec LangChain."""
    if not FAISS_FOLDER.exists():
        raise FileNotFoundError(
            f"Index introuvable : {FAISS_FOLDER}. "
            "Exécute d'abord : python build_langchain_faiss.py"
        )

    embeddings = MistralAIEmbeddings(
        model=EMBEDDING_MODEL,
        max_concurrent_requests=1,
    )

    # Attention : autorisé seulement pour un index que TU as créé et dont tu
    # contrôles le fichier index.pkl. Ne charge jamais un pickle inconnu.
    return FAISS.load_local(
        str(FAISS_FOLDER),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def build_rag_chain(vector_store: FAISS):
    """
    Assemble une chaîne LCEL :
    question -> retriever FAISS -> format contexte -> prompt -> LLM -> texte.
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "Tu es un assistant spécialisé dans les événements culturels "
                    "à Montpellier. Réponds en français de façon claire, courte et "
                    "factuelle. Utilise exclusivement le contexte fourni. "
                    "Ne crée aucune date, URL, lieu, artiste ou événement. "
                    "Si le contexte ne permet pas une réponse fiable, dis exactement : "
                    "\"Je ne dispose pas d'assez d'informations dans les événements indexés.\" "
                    "Pour chaque événement mentionné, cite son titre, sa date si elle "
                    "est disponible, et son URL source."
                ),
            ),
            (
                "human",
                "Contexte récupéré :\n{context}\n\nQuestion : {question}",
            ),
        ]
    )

    llm = ChatMistralAI(
        model=CHAT_MODEL,
        temperature=TEMPERATURE,
        max_retries=2,
        timeout=60,
    )

    chain = (
        {
            "context": retriever | format_documents,
            "question": lambda question: question,
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def main():
    vector_store = load_vector_store()
    rag_chain, retriever = build_rag_chain(vector_store)

    question = "Quels sont les concerts et festivals de musique à Montpellier cet été ?"

    # Facultatif mais très utile en développement :
    # inspecter les chunks récupérés AVANT la génération LLM.
    documents = retriever.invoke(question)

    print("\n--- Chunks récupérés ---\n")
    for index, document in enumerate(documents, start=1):
        print(f"{index}. {document.metadata.get('title', 'Sans titre')}")
        print(f"   Score non retourné par ce retriever par défaut")
        print(f"   Texte : {document.page_content[:250]}...")
        print()

    print("--- Réponse RAG ---\n")
    answer = rag_chain.invoke(question)
    print(answer)


if __name__ == "__main__":
    main()