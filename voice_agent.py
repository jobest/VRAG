from openai import AsyncOpenAI
import streamlit as st

COLLECTION_NAME = "voice-rag-agent"

async def process_query(query, client, embedder, openai_api_key, voice):
    st.info("Searching vector database...")
    vector = list(embedder.embed([query]))[0]

    search = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        limit=3,
        with_payload=True
    )

    context = "Based on the following documentation:\n\n"
    if search:
        for i, hit in enumerate(search):
            content = hit.payload.get("content", "")
            source = hit.payload.get("file_name", "Unknown")
            context += f"--- Source {i+1}: {source} ---\n{content}\n\n"
    else:
        context += "No relevant content found.\n"

    context += f"\nUser Question: {query}\n\n"
    context += "Please provide a spoken response based on the above."

    system = (
        "You are a helpful assistant. Read and analyze the documents provided.\n"
        "Answer clearly. Be concise and conversational. Include filenames as needed."
    )

    openai_client = AsyncOpenAI(api_key=openai_api_key)

    try:
        st.info("Generating answer...")
        res = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": context}
            ],
            temperature=0.7
        )

        text = res.choices[0].message.content
        st.markdown(f"**AI Response:**\n\n{text}")

        st.info("Generating audio...")
        audio_response = await openai_client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
            response_format="wav"
        )
        st.audio(audio_response.content, format="audio/wav")

    except Exception as e:
        st.error(f"AI processing failed: {e}")