from fastapi import FastAPI, File, UploadFile
from app.function_definitions import functions
from app.functions import api_functions, create_pizzas
from app.handler import OpenAIHandler
from app.models import Interaction, Conversation, Message
from app.db import Base, engine
from app.prompts import system_message
import os
from app.store import create_store
from app.db import Session, Review, Order
from app.audio_handler import AudioHandler


app = FastAPI()
handler = OpenAIHandler(api_functions, functions, system_message)
audio_handler = AudioHandler()


@app.on_event("startup")
async def startup_event():
    Base.metadata.create_all(bind=engine)
    create_pizzas()
    if not os.path.exists("vectorstore.pkl"):
        create_store()


@app.on_event("shutdown")
async def shutdown_event():
    engine.dispose()
    try:
        os.remove("pizzadb.db")
        os.remove("vectorstore.pkl")
    except FileNotFoundError:
        pass  
    except PermissionError as e:
        print(f"Error deleting database file: {e}")


@app.post("/voice_conversation")
async def voice_query_endpoint(file: UploadFile = File(...)):
    try:
        # Read and process the audio file
        audio_bytes = await file.read()
        wav_audio = audio_handler.convert_audio_to_wav(audio_bytes)
        query_text = audio_handler.transcribe_audio(wav_audio)

        # Create the required Conversation and Interaction objects
        conversation = Conversation(messages=[Message(role="user", content=query_text)])
        interaction = Interaction(conversation=conversation, query=query_text)

        # Process the query and return the response
        response = handler.send_response(interaction.query)
        return {"response": response, "query_text": query_text}
    except ValueError as ve:
        return {"error": str(ve)}
    except ConnectionError as ce:
        return {"error": str(ce)}


@app.post("/conversation")
async def query_endpoint(interaction: Interaction):
    response = handler.send_response(interaction.query)
    return {"response": response}


@app.get("/reviews")
async def get_all_reviews():
    session = Session()
    reviews = session.query(Review).all()
    session.close()
    return reviews


@app.get("/orders")
async def get_all_orders():
    session = Session()
    orders = session.query(Order).all()
    session.close()
    return orders
